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
#define SMEM_EXCH_OFF 228416
#define SMEM_EXCH_STAGE_BYTES 3072
#define SMEM_EXCH_STRIDE 3072
#define SMEM_SMEM_V15_OFF 1024
#define SMEM_SMEM_V15_STAGE_BYTES 6144
#define SMEM_SMEM_V15_STRIDE 38912
#define SMEM_SMEM_V16_OFF 1072
#define SMEM_SMEM_V16_STAGE_BYTES 6144
#define SMEM_SMEM_V16_STRIDE 38912
#define SMEM_SMEM_V17_OFF 1120
#define SMEM_SMEM_V17_STAGE_BYTES 6144
#define SMEM_SMEM_V17_STRIDE 38912
#define SMEM_SMEM_V18_OFF 1040
#define SMEM_SMEM_V18_STAGE_BYTES 6144
#define SMEM_SMEM_V18_STRIDE 38912
#define SMEM_SMEM_V19_OFF 1088
#define SMEM_SMEM_V19_STAGE_BYTES 6144
#define SMEM_SMEM_V19_STRIDE 38912
#define SMEM_SMEM_V20_OFF 1136
#define SMEM_SMEM_V20_STAGE_BYTES 6144
#define SMEM_SMEM_V20_STRIDE 38912
#define SMEM_SMEM_V21_OFF 1056
#define SMEM_SMEM_V21_STAGE_BYTES 6144
#define SMEM_SMEM_V21_STRIDE 38912
#define SMEM_SMEM_V22_OFF 1104
#define SMEM_SMEM_V22_STAGE_BYTES 6144
#define SMEM_SMEM_V22_STRIDE 38912
#define SMEM_SMEM_V23_OFF 1024
#define SMEM_SMEM_V23_STAGE_BYTES 6144
#define SMEM_SMEM_V23_STRIDE 38912
#define SMEM_SMEM_V24_OFF 17408
#define SMEM_SMEM_V24_STAGE_BYTES 6144
#define SMEM_SMEM_V24_STRIDE 38912
#define SMEM_SMEM_V25_OFF 17456
#define SMEM_SMEM_V25_STAGE_BYTES 6144
#define SMEM_SMEM_V25_STRIDE 38912
#define SMEM_SMEM_V26_OFF 17504
#define SMEM_SMEM_V26_STAGE_BYTES 6144
#define SMEM_SMEM_V26_STRIDE 38912
#define SMEM_SMEM_V27_OFF 17424
#define SMEM_SMEM_V27_STAGE_BYTES 6144
#define SMEM_SMEM_V27_STRIDE 38912
#define SMEM_SMEM_V28_OFF 17472
#define SMEM_SMEM_V28_STAGE_BYTES 6144
#define SMEM_SMEM_V28_STRIDE 38912
#define SMEM_SMEM_V29_OFF 17520
#define SMEM_SMEM_V29_STAGE_BYTES 6144
#define SMEM_SMEM_V29_STRIDE 38912
#define SMEM_SMEM_V30_OFF 17440
#define SMEM_SMEM_V30_STAGE_BYTES 6144
#define SMEM_SMEM_V30_STRIDE 38912
#define SMEM_SMEM_V31_OFF 17488
#define SMEM_SMEM_V31_STAGE_BYTES 6144
#define SMEM_SMEM_V31_STRIDE 38912
#define SMEM_SMEM_V32_OFF 17408
#define SMEM_SMEM_V32_STAGE_BYTES 6144
#define SMEM_SMEM_V32_STRIDE 38912
#define SMEM_TOTAL 231552
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 5
#define GROUP_M 64
#define WORK_STAGES 4
#define WORK_CONSUMERS 1058
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


__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
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

__global__ __launch_bounds__(640, 1) __cluster_dims__(2,1,1) void
kernel_cake_minimax_h3_nvfp4_pre_attention_10a6b21d3c5085b7f12d(CakeTensorMap const* A, CakeTensorMap const* B, CakeTensorMap const* SFA, CakeTensorMap const* SFB, float* __restrict__ alpha, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, float* __restrict__ out_global_scale, uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, CakeTensorMap const* OUTQ, unsigned int* __restrict__ qkv_words, unsigned int* __restrict__ debug_q_words, unsigned int* __restrict__ debug_k_words, int write_debug, float eps, int M, int m_tiles, int HEADS_PER_DESTINATION, int ROWS_PER_DESTINATION, int SCALE_STRIDE)
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
    float* exch = reinterpret_cast<float*>(smem_raw + 228416);
    const int exch_addr = smem + 228416;
    uint8_t* smem_v15 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v15_addr = smem + 1024;
    uint8_t* smem_v16 = reinterpret_cast<uint8_t*>(smem_raw + 1072);
    const int smem_v16_addr = smem + 1072;
    uint8_t* smem_v17 = reinterpret_cast<uint8_t*>(smem_raw + 1120);
    const int smem_v17_addr = smem + 1120;
    uint8_t* smem_v18 = reinterpret_cast<uint8_t*>(smem_raw + 1040);
    const int smem_v18_addr = smem + 1040;
    uint8_t* smem_v19 = reinterpret_cast<uint8_t*>(smem_raw + 1088);
    const int smem_v19_addr = smem + 1088;
    uint8_t* smem_v20 = reinterpret_cast<uint8_t*>(smem_raw + 1136);
    const int smem_v20_addr = smem + 1136;
    uint8_t* smem_v21 = reinterpret_cast<uint8_t*>(smem_raw + 1056);
    const int smem_v21_addr = smem + 1056;
    uint8_t* smem_v22 = reinterpret_cast<uint8_t*>(smem_raw + 1104);
    const int smem_v22_addr = smem + 1104;
    uint8_t* smem_v23 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v23_addr = smem + 1024;
    uint8_t* smem_v24 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v24_addr = smem + 17408;
    uint8_t* smem_v25 = reinterpret_cast<uint8_t*>(smem_raw + 17456);
    const int smem_v25_addr = smem + 17456;
    uint8_t* smem_v26 = reinterpret_cast<uint8_t*>(smem_raw + 17504);
    const int smem_v26_addr = smem + 17504;
    uint8_t* smem_v27 = reinterpret_cast<uint8_t*>(smem_raw + 17424);
    const int smem_v27_addr = smem + 17424;
    uint8_t* smem_v28 = reinterpret_cast<uint8_t*>(smem_raw + 17472);
    const int smem_v28_addr = smem + 17472;
    uint8_t* smem_v29 = reinterpret_cast<uint8_t*>(smem_raw + 17520);
    const int smem_v29_addr = smem + 17520;
    uint8_t* smem_v30 = reinterpret_cast<uint8_t*>(smem_raw + 17440);
    const int smem_v30_addr = smem + 17440;
    uint8_t* smem_v31 = reinterpret_cast<uint8_t*>(smem_raw + 17488);
    const int smem_v31_addr = smem + 17488;
    uint8_t* smem_v32 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v32_addr = smem + 17408;

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
            // epilogue_done: 1 barriers, init_count=32
            mbarrier_init(smem + 88, 32);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // work_empty: 4 barriers, init_count=1058
            mbarrier_init(smem + 128, 1058);
            mbarrier_init(smem + 136, 1058);
            mbarrier_init(smem + 144, 1058);
            mbarrier_init(smem + 152, 1058);
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

    // ---- Warpgroup: 0 ----
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
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
        // ---- Role: mma ----
        } else if (warp == 1) {
            { // mma_main
                unsigned int mma_tma_stage = 0;
                unsigned int acc_stage = 0;
                unsigned int work_stage_1 = 0;
                unsigned int _phase_tma_full = 0;
                unsigned int _phase_epilogue_done = 1;
                unsigned int _phase_work_full_1 = 0;
                if (cta_rank == 0) {
                    #pragma unroll 1
                    for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles; _tile_iter_1++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432 + 32)));
                        }
                        mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
                        #pragma unroll 1
                        for (int super_k = 0; super_k < 7; super_k++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p0 = mma_tma_stage;
                            int init_flag = ((super_k == 0) ? 1 : 0);
                            if (elect_sync()) {
                                if (super_k != 0) {
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
                                int _mma_a_lo_0 = (((smem_v15_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_b_lo_0 = (((smem_v24_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_a_next_lo_0 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_b_next_lo_0 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0 | ((uint64_t)((uint32_t)_mma_a_next_lo_0 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0 | ((uint64_t)((uint32_t)_mma_b_next_lo_0 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1));
                                }
                                int _mma_a_lo_1 = (((smem_v16_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_b_lo_1 = (((smem_v25_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_a_next_lo_1 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_b_next_lo_1 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1 | ((uint64_t)((uint32_t)_mma_a_next_lo_1 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1 | ((uint64_t)((uint32_t)_mma_b_next_lo_1 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1);
                                }
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p1 = mma_tma_stage;
                            if (elect_sync()) {
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
                                int _mma_a_lo_2 = (((smem_v17_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_b_lo_2 = (((smem_v26_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_a_next_lo_2 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_b_next_lo_2 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2 | ((uint64_t)((uint32_t)_mma_a_next_lo_2 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2 | ((uint64_t)((uint32_t)_mma_b_next_lo_2 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1);
                                }
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p0) * 8, (uint16_t)(3));
                            if (elect_sync()) {
                                int _mma_a_lo_3 = (((smem_v18_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_b_lo_3 = (((smem_v27_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_a_next_lo_3 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_b_next_lo_3 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3 | ((uint64_t)((uint32_t)_mma_a_next_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3 | ((uint64_t)((uint32_t)_mma_b_next_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + 32 + 0, 1);
                                }
                                int _mma_a_lo_4 = (((smem_v19_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_b_lo_4 = (((smem_v28_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_a_next_lo_4 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_b_next_lo_4 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4 | ((uint64_t)((uint32_t)_mma_a_next_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4 | ((uint64_t)((uint32_t)_mma_b_next_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb + 48 + 0, 1);
                                }
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p2 = mma_tma_stage;
                            if (elect_sync()) {
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
                                int _mma_a_lo_5 = (((smem_v20_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_b_lo_5 = (((smem_v29_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_a_next_lo_5 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_b_next_lo_5 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5 | ((uint64_t)((uint32_t)_mma_a_next_lo_5 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5 | ((uint64_t)((uint32_t)_mma_b_next_lo_5 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb + 56 + 0, 1);
                                }
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p1) * 8, (uint16_t)(3));
                            if (elect_sync()) {
                                int _mma_a_lo_6 = (((smem_v21_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_b_lo_6 = (((smem_v30_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_a_next_lo_6 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_b_next_lo_6 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6 | ((uint64_t)((uint32_t)_mma_a_next_lo_6 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6 | ((uint64_t)((uint32_t)_mma_b_next_lo_6 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb + 72 + 0, 1);
                                }
                                int _mma_a_lo_7 = (((smem_v22_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_b_lo_7 = (((smem_v31_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_a_next_lo_7 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_b_next_lo_7 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7 | ((uint64_t)((uint32_t)_mma_a_next_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7 | ((uint64_t)((uint32_t)_mma_b_next_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb + 80 + 0, 1);
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
        // ---- Role: padding ----
        } else if (warp >= 2 && warp <= 3) {
            // idle — no tasks assigned
        }
    }
    // ---- Role: epilogue_a ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // epilogue_a_main
            unsigned int acc_stage_1 = 0;
            unsigned int work_stage_2 = 0;
            const int epi_index = warp - 4;
            const int epi_warp = epi_index % 4;
            const int head_half = epi_index / 4;
            const int local_row = epi_warp * 32 + lane;
            int exch_slot = (head_half * 128 + local_row) * 3;
            int pair_bar = 1 + head_half * 4 + epi_warp;
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
                unsigned int words[32];
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], lane_addr);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_tmem_load_0[2 * j] * alpha_v, _tmem_load_0[2 * j + 1] * alpha_v));
                    words[j] = __as_u32(_bf16x2_0);
                }
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], lane_addr + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_tmem_load_1[2 * j_1] * alpha_v, _tmem_load_1[2 * j_1 + 1] * alpha_v));
                    words[8 + j_1] = __as_u32(_bf16x2_1);
                }
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], lane_addr + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_tmem_load_2[2 * j_2] * alpha_v, _tmem_load_2[2 * j_2 + 1] * alpha_v));
                    words[16 + j_2] = __as_u32(_bf16x2_2);
                }
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], lane_addr + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_tmem_load_3[2 * j_3] * alpha_v, _tmem_load_3[2 * j_3 + 1] * alpha_v));
                    words[24 + j_3] = __as_u32(_bf16x2_3);
                }
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                {
                    int destination = head / HEADS_PER_DESTINATION;
                    int local_head = head % HEADS_PER_DESTINATION;
                    int head_kind_local = local_head * 3 + kind;
                    int row_in_destination = token * rows_per_token + head_kind_local;
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
                    asm volatile("barrier.sync %0, 256;" :: "r"(store_bar) : "memory");
                    if (kind < 2) {
                        float partial[4];
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            float sum_lo = 0.0f;
                            #pragma unroll
                            for (int j_4 = 0; j_4 < 4; j_4++) {
                                float v_lo = __uint_as_float(words[i * 8 + j_4] << 16);
                                float v_hi = __uint_as_float(words[i * 8 + j_4] & 4294901760);
                                float _fma_0 = __fmaf_rn(v_lo, v_lo, sum_lo);
                                sum_lo = _fma_0;
                                float _fma_1 = __fmaf_rn(v_hi, v_hi, sum_lo);
                                sum_lo = _fma_1;
                            }
                            float sum_hi = 0.0f;
                            #pragma unroll
                            for (int j_5 = 4; j_5 < 8; j_5++) {
                                float v_lo2 = __uint_as_float(words[i * 8 + j_5] << 16);
                                float v_hi2 = __uint_as_float(words[i * 8 + j_5] & 4294901760);
                                float _fma_2 = __fmaf_rn(v_lo2, v_lo2, sum_hi);
                                sum_hi = _fma_2;
                                float _fma_3 = __fmaf_rn(v_hi2, v_hi2, sum_hi);
                                sum_hi = _fma_3;
                            }
                            partial[i] = sum_lo + sum_hi;
                        }
                        float s01 = partial[0] + partial[1];
                        exch[exch_slot] = s01;
                        exch[exch_slot + 1] = partial[2];
                        exch[exch_slot + 2] = partial[3];
                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                        float p2 = exch[exch_slot];
                        float p5 = exch[exch_slot + 1];
                        float s67 = exch[exch_slot + 2];
                        float s23 = p2 + partial[2];
                        float s45 = partial[3] + p5;
                        float sum_sq = s01 + s23 + (s45 + s67);
                        float _fdiv_rn_0 = __fdiv_rn(sum_sq, 128.0f);
                        float mean_sq = _fdiv_rn_0;
                        float _rsqrt_0 = rsqrtf(mean_sq + eps);
                        float rstd = _rsqrt_0;
                        float2 _f2_0 = make_float2(rstd, rstd);
                        unsigned int normalized[32];
                        unsigned int weight_words[8];
                        #pragma unroll
                        for (int h = 0; h < 2; h++) {
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight : k_norm_weight) + h * 8);
                                uint4* _vdst_0 = reinterpret_cast<uint4*>(&weight_words[4 * h]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_0[_blk] = _vptr_0[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_6 = 0; j_6 < 8; j_6++) {
                            float2 _f2_1 = make_float2(__uint_as_float(words[j_6] << 16), __uint_as_float(words[j_6] & 4294901760));
                            float2 _mul_f32x2_0;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_1), "l"(*(const unsigned long long*)&_f2_0));
                            float2 _f2_2 = make_float2(__uint_as_float(weight_words[j_6] << 16), __uint_as_float(weight_words[j_6] & 4294901760));
                            float2 _mul_f32x2_1;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_mul_f32x2_0), "l"(*(const unsigned long long*)&_f2_2));
                            __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(_mul_f32x2_1.x, _mul_f32x2_1.y));
                            normalized[j_6] = __as_u32(_bf16x2_4);
                        }
                        unsigned int weight_words_0[8];
                        #pragma unroll
                        for (int h_1 = 0; h_1 < 2; h_1++) {
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight : k_norm_weight) + 16 + h_1 * 8);
                                uint4* _vdst_1 = reinterpret_cast<uint4*>(&weight_words_0[4 * h_1]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_1[_blk] = _vptr_1[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_7 = 0; j_7 < 8; j_7++) {
                            float2 _f2_3 = make_float2(__uint_as_float(words[8 + j_7] << 16), __uint_as_float(words[8 + j_7] & 4294901760));
                            float2 _mul_f32x2_2;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_3), "l"(*(const unsigned long long*)&_f2_0));
                            float2 _f2_4 = make_float2(__uint_as_float(weight_words_0[j_7] << 16), __uint_as_float(weight_words_0[j_7] & 4294901760));
                            float2 _mul_f32x2_3;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_mul_f32x2_2), "l"(*(const unsigned long long*)&_f2_4));
                            __nv_bfloat162 _bf16x2_5 = __float22bfloat162_rn(make_float2(_mul_f32x2_3.x, _mul_f32x2_3.y));
                            normalized[8 + j_7] = __as_u32(_bf16x2_5);
                        }
                        unsigned int weight_words_1[8];
                        #pragma unroll
                        for (int h_2 = 0; h_2 < 2; h_2++) {
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight : k_norm_weight) + 48 + h_2 * 8);
                                uint4* _vdst_2 = reinterpret_cast<uint4*>(&weight_words_1[4 * h_2]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_2[_blk] = _vptr_2[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_8 = 0; j_8 < 8; j_8++) {
                            float2 _f2_5 = make_float2(__uint_as_float(words[16 + j_8] << 16), __uint_as_float(words[16 + j_8] & 4294901760));
                            float2 _mul_f32x2_4;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_5), "l"(*(const unsigned long long*)&_f2_0));
                            float2 _f2_6 = make_float2(__uint_as_float(weight_words_1[j_8] << 16), __uint_as_float(weight_words_1[j_8] & 4294901760));
                            float2 _mul_f32x2_5;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_mul_f32x2_4), "l"(*(const unsigned long long*)&_f2_6));
                            __nv_bfloat162 _bf16x2_6 = __float22bfloat162_rn(make_float2(_mul_f32x2_5.x, _mul_f32x2_5.y));
                            normalized[16 + j_8] = __as_u32(_bf16x2_6);
                        }
                        unsigned int weight_words_2[8];
                        #pragma unroll
                        for (int h_3 = 0; h_3 < 2; h_3++) {
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight : k_norm_weight) + 64 + h_3 * 8);
                                uint4* _vdst_3 = reinterpret_cast<uint4*>(&weight_words_2[4 * h_3]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_3[_blk] = _vptr_3[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_9 = 0; j_9 < 8; j_9++) {
                            float2 _f2_7 = make_float2(__uint_as_float(words[24 + j_9] << 16), __uint_as_float(words[24 + j_9] & 4294901760));
                            float2 _mul_f32x2_6;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&_f2_7), "l"(*(const unsigned long long*)&_f2_0));
                            float2 _f2_8 = make_float2(__uint_as_float(weight_words_2[j_9] << 16), __uint_as_float(weight_words_2[j_9] & 4294901760));
                            float2 _mul_f32x2_7;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&_mul_f32x2_6), "l"(*(const unsigned long long*)&_f2_8));
                            __nv_bfloat162 _bf16x2_7 = __float22bfloat162_rn(make_float2(_mul_f32x2_7.x, _mul_f32x2_7.y));
                            normalized[24 + j_9] = __as_u32(_bf16x2_7);
                        }
                        int _min_0 = ((token) < (M - 1) ? (token) : (M - 1));
                        int rope_token = _min_0;
                        unsigned int cos_words[8];
                        unsigned int sin_words[8];
                        #pragma unroll
                        for (int h_4 = 0; h_4 < 2; h_4++) {
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(rope_cos_sin + rope_token * 96 + h_4 * 8);
                                uint4* _vdst_4 = reinterpret_cast<uint4*>(&cos_words[4 * h_4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_4[_blk] = _vptr_4[_blk];
                                }
                            }
                            {
                                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(rope_cos_sin + rope_token * 96 + 48 + h_4 * 8);
                                uint4* _vdst_5 = reinterpret_cast<uint4*>(&sin_words[4 * h_4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_5[_blk] = _vptr_5[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_10 = 0; j_10 < 8; j_10++) {
                            unsigned int n_lo_word = normalized[j_10];
                            unsigned int n_hi_word = normalized[16 + j_10];
                            float2 _f2_9 = make_float2(__uint_as_float(cos_words[j_10] << 16), __uint_as_float(cos_words[j_10] & 4294901760));
                            float2 _f2_10 = make_float2(__uint_as_float(sin_words[j_10] << 16 ^ 2147483648), __uint_as_float(sin_words[j_10] & 4294901760 ^ 2147483648));
                            float2 _f2_11 = make_float2(__uint_as_float(sin_words[j_10] << 16), __uint_as_float(sin_words[j_10] & 4294901760));
                            float2 _f2_12 = make_float2(__uint_as_float(n_hi_word << 16), __uint_as_float(n_hi_word & 4294901760));
                            float2 _mul_f32x2_8;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&_f2_12), "l"(*(const unsigned long long*)&_f2_10));
                            float2 _f2_13 = make_float2(__uint_as_float(n_lo_word << 16), __uint_as_float(n_lo_word & 4294901760));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_fma_f32x2_0;
                            {
                                float2 _packed_f32x2_6_0 = _f2_9;
                                float2 _packed_f32x2_6_1 = _f2_13;
                                float2 _packed_f32x2_6_2 = _mul_f32x2_8;
                                asm volatile("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(reinterpret_cast<uint64_t&>(_packed_fma_f32x2_0)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_6_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_6_1)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_6_2)));
                            }
                            float2 _f2_14 = make_float2(__uint_as_float(n_lo_word << 16), __uint_as_float(n_lo_word & 4294901760));
                            float2 _mul_f32x2_9;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_f2_14), "l"(*(const unsigned long long*)&_f2_11));
                            float2 _f2_15 = make_float2(__uint_as_float(n_hi_word << 16), __uint_as_float(n_hi_word & 4294901760));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_fma_f32x2_1;
                            {
                                float2 _packed_f32x2_7_0 = _f2_9;
                                float2 _packed_f32x2_7_1 = _f2_15;
                                float2 _packed_f32x2_7_2 = _mul_f32x2_9;
                                asm volatile("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(reinterpret_cast<uint64_t&>(_packed_fma_f32x2_1)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_7_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_7_1)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_7_2)));
                            }
                            __nv_bfloat162 _bf16x2_8 = __float22bfloat162_rn(make_float2(_packed_fma_f32x2_0.x, _packed_fma_f32x2_0.y));
                            normalized[j_10] = __as_u32(_bf16x2_8);
                            __nv_bfloat162 _bf16x2_9 = __float22bfloat162_rn(make_float2(_packed_fma_f32x2_1.x, _packed_fma_f32x2_1.y));
                            normalized[16 + j_10] = __as_u32(_bf16x2_9);
                        }
                        unsigned int cos_words_3[8];
                        unsigned int sin_words_4[8];
                        #pragma unroll
                        for (int h_5 = 0; h_5 < 2; h_5++) {
                            {
                                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(rope_cos_sin + rope_token * 96 + 16 + h_5 * 8);
                                uint4* _vdst_8 = reinterpret_cast<uint4*>(&cos_words_3[4 * h_5]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_8[_blk] = _vptr_8[_blk];
                                }
                            }
                            {
                                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(rope_cos_sin + rope_token * 96 + 48 + 16 + h_5 * 8);
                                uint4* _vdst_9 = reinterpret_cast<uint4*>(&sin_words_4[4 * h_5]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_9[_blk] = _vptr_9[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_11 = 0; j_11 < 8; j_11++) {
                            unsigned int n_lo_word_1 = normalized[8 + j_11];
                            unsigned int n_hi_word_1 = normalized[24 + j_11];
                            float2 _f2_16 = make_float2(__uint_as_float(cos_words_3[j_11] << 16), __uint_as_float(cos_words_3[j_11] & 4294901760));
                            float2 _f2_17 = make_float2(__uint_as_float(sin_words_4[j_11] << 16 ^ 2147483648), __uint_as_float(sin_words_4[j_11] & 4294901760 ^ 2147483648));
                            float2 _f2_18 = make_float2(__uint_as_float(sin_words_4[j_11] << 16), __uint_as_float(sin_words_4[j_11] & 4294901760));
                            float2 _f2_19 = make_float2(__uint_as_float(n_hi_word_1 << 16), __uint_as_float(n_hi_word_1 & 4294901760));
                            float2 _mul_f32x2_10;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&_f2_19), "l"(*(const unsigned long long*)&_f2_17));
                            float2 _f2_20 = make_float2(__uint_as_float(n_lo_word_1 << 16), __uint_as_float(n_lo_word_1 & 4294901760));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_fma_f32x2_2;
                            {
                                float2 _packed_f32x2_10_0 = _f2_16;
                                float2 _packed_f32x2_10_1 = _f2_20;
                                float2 _packed_f32x2_10_2 = _mul_f32x2_10;
                                asm volatile("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(reinterpret_cast<uint64_t&>(_packed_fma_f32x2_2)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_10_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_10_1)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_10_2)));
                            }
                            float2 _f2_21 = make_float2(__uint_as_float(n_lo_word_1 << 16), __uint_as_float(n_lo_word_1 & 4294901760));
                            float2 _mul_f32x2_11;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&_f2_21), "l"(*(const unsigned long long*)&_f2_18));
                            float2 _f2_22 = make_float2(__uint_as_float(n_hi_word_1 << 16), __uint_as_float(n_hi_word_1 & 4294901760));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_fma_f32x2_3;
                            {
                                float2 _packed_f32x2_11_0 = _f2_16;
                                float2 _packed_f32x2_11_1 = _f2_22;
                                float2 _packed_f32x2_11_2 = _mul_f32x2_11;
                                asm volatile("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(reinterpret_cast<uint64_t&>(_packed_fma_f32x2_3)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_11_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_11_1)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_11_2)));
                            }
                            __nv_bfloat162 _bf16x2_10 = __float22bfloat162_rn(make_float2(_packed_fma_f32x2_2.x, _packed_fma_f32x2_2.y));
                            normalized[8 + j_11] = __as_u32(_bf16x2_10);
                            __nv_bfloat162 _bf16x2_11 = __float22bfloat162_rn(make_float2(_packed_fma_f32x2_3.x, _packed_fma_f32x2_3.y));
                            normalized[24 + j_11] = __as_u32(_bf16x2_11);
                        }
                        if (write_debug != 0) {
                            if (token < M) {
                                unsigned long long debug_row = ((unsigned long long)token * 56 + (unsigned long long)head) * 64;
                                #pragma unroll
                                for (int c = 0; c < 2; c++) {
                                    unsigned int debug_chunk[4];
                                    #pragma unroll
                                    for (int j_12 = 0; j_12 < 4; j_12++) {
                                        debug_chunk[j_12] = normalized[4 * c + j_12];
                                    }
                                    {
                                        int4 _iv4 = make_int4(debug_chunk[0 + 0], debug_chunk[0 + 1], debug_chunk[0 + 2], debug_chunk[0 + 3]);
                                        *reinterpret_cast<int4*>(((kind == 0) ? debug_q_words : debug_k_words) + (debug_row + (unsigned long long)(4 * c)) + 0) = _iv4;
                                    }
                                }
                                #pragma unroll
                                for (int c_1 = 0; c_1 < 2; c_1++) {
                                    unsigned int debug_chunk_1[4];
                                    #pragma unroll
                                    for (int j_13 = 0; j_13 < 4; j_13++) {
                                        debug_chunk_1[j_13] = normalized[8 + 4 * c_1 + j_13];
                                    }
                                    {
                                        int4 _iv4 = make_int4(debug_chunk_1[0 + 0], debug_chunk_1[0 + 1], debug_chunk_1[0 + 2], debug_chunk_1[0 + 3]);
                                        *reinterpret_cast<int4*>(((kind == 0) ? debug_q_words : debug_k_words) + (debug_row + 8 + 4 * c_1) + 0) = _iv4;
                                    }
                                }
                                #pragma unroll
                                for (int c_2 = 0; c_2 < 2; c_2++) {
                                    unsigned int debug_chunk_2[4];
                                    #pragma unroll
                                    for (int j_14 = 0; j_14 < 4; j_14++) {
                                        debug_chunk_2[j_14] = normalized[16 + 4 * c_2 + j_14];
                                    }
                                    {
                                        int4 _iv4 = make_int4(debug_chunk_2[0 + 0], debug_chunk_2[0 + 1], debug_chunk_2[0 + 2], debug_chunk_2[0 + 3]);
                                        *reinterpret_cast<int4*>(((kind == 0) ? debug_q_words : debug_k_words) + (debug_row + 24 + 4 * c_2) + 0) = _iv4;
                                    }
                                }
                                #pragma unroll
                                for (int c_3 = 0; c_3 < 2; c_3++) {
                                    unsigned int debug_chunk_3[4];
                                    #pragma unroll
                                    for (int j_15 = 0; j_15 < 4; j_15++) {
                                        debug_chunk_3[j_15] = normalized[24 + 4 * c_3 + j_15];
                                    }
                                    {
                                        int4 _iv4 = make_int4(debug_chunk_3[0 + 0], debug_chunk_3[0 + 1], debug_chunk_3[0 + 2], debug_chunk_3[0 + 3]);
                                        *reinterpret_cast<int4*>(((kind == 0) ? debug_q_words : debug_k_words) + (debug_row + 32 + 4 * c_3) + 0) = _iv4;
                                    }
                                }
                            }
                        }
                        unsigned int row_out[8];
                        float sf_values[4];
                        float values[16];
                        float absolute[16];
                        float quant_values[16];
                        unsigned int packed[2];
                        #pragma unroll
                        for (int j_16 = 0; j_16 < 8; j_16++) {
                            values[2 * j_16] = __uint_as_float(normalized[j_16] << 16);
                            values[2 * j_16 + 1] = __uint_as_float(normalized[j_16] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_17 = 0; j_17 < 16; j_17++) {
                            absolute[j_17] = values[j_17];
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
                        uint16_t _e4m3x2_12;
                        uint32_t _f16x2_12;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_12) : "h"(_e4m3x2_12));
                        uint16_t _fp8_h0_12 = (uint16_t)(_f16x2_12 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_12));
                        float sf_rounded = _fp8_rt_0;
                        float _rcp_2 = approx_rcp(sf_rounded * global_scale_rcp);
                        float _min_1 = fminf(_rcp_2, 3.4028234663852886e+38f);
                        float output_scale = _min_1;
                        float2 _f2_23 = make_float2(output_scale, output_scale);
                        #pragma unroll
                        for (int j_18 = 0; j_18 < 8; j_18++) {
                            float2 _f2_24 = make_float2(values[2 * j_18], values[2 * j_18 + 1]);
                            float2 _mul_f32x2_12;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_23));
                            quant_values[2 * j_18] = _mul_f32x2_12.x;
                            quant_values[2 * j_18 + 1] = _mul_f32x2_12.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(quant_values[8]), "f"(quant_values[9]), "f"(quant_values[10]), "f"(quant_values[11]), "f"(quant_values[12]), "f"(quant_values[13]), "f"(quant_values[14]), "f"(quant_values[15]));
                        row_out[0] = packed[0];
                        row_out[1] = packed[1];
                        sf_values[0] = sf_value;
                        float values_5[16];
                        float absolute_6[16];
                        float quant_values_7[16];
                        unsigned int packed_8[2];
                        #pragma unroll
                        for (int j_19 = 0; j_19 < 8; j_19++) {
                            values_5[2 * j_19] = __uint_as_float(normalized[8 + j_19] << 16);
                            values_5[2 * j_19 + 1] = __uint_as_float(normalized[8 + j_19] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_20 = 0; j_20 < 16; j_20++) {
                            absolute_6[j_20] = values_5[j_20];
                        }
                        float _fabs_16 = fabsf(absolute_6[0]);
                        absolute_6[0] = _fabs_16;
                        float _fabs_17 = fabsf(absolute_6[1]);
                        absolute_6[1] = _fabs_17;
                        float _fabs_18 = fabsf(absolute_6[2]);
                        absolute_6[2] = _fabs_18;
                        float _fabs_19 = fabsf(absolute_6[3]);
                        absolute_6[3] = _fabs_19;
                        float _fabs_20 = fabsf(absolute_6[4]);
                        absolute_6[4] = _fabs_20;
                        float _fabs_21 = fabsf(absolute_6[5]);
                        absolute_6[5] = _fabs_21;
                        float _fabs_22 = fabsf(absolute_6[6]);
                        absolute_6[6] = _fabs_22;
                        float _fabs_23 = fabsf(absolute_6[7]);
                        absolute_6[7] = _fabs_23;
                        float _fabs_24 = fabsf(absolute_6[8]);
                        absolute_6[8] = _fabs_24;
                        float _fabs_25 = fabsf(absolute_6[9]);
                        absolute_6[9] = _fabs_25;
                        float _fabs_26 = fabsf(absolute_6[10]);
                        absolute_6[10] = _fabs_26;
                        float _fabs_27 = fabsf(absolute_6[11]);
                        absolute_6[11] = _fabs_27;
                        float _fabs_28 = fabsf(absolute_6[12]);
                        absolute_6[12] = _fabs_28;
                        float _fabs_29 = fabsf(absolute_6[13]);
                        absolute_6[13] = _fabs_29;
                        float _fabs_30 = fabsf(absolute_6[14]);
                        absolute_6[14] = _fabs_30;
                        float _fabs_31 = fabsf(absolute_6[15]);
                        absolute_6[15] = _fabs_31;
                        float absolute_6_max = absolute_6[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_6_max = max_noftz(absolute_6_max, absolute_6[_lr]);
                        }
                        float amax_9 = absolute_6_max;
                        float _rcp_3 = approx_rcp(6.0f);
                        float sf_value_10 = global_scale * (amax_9 * _rcp_3);
                        float _fp8_rt_1;
                        uint16_t _e4m3x2_13;
                        uint32_t _f16x2_13;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_13) : "f"(0.0f), "f"(sf_value_10));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_13) : "h"(_e4m3x2_13));
                        uint16_t _fp8_h0_13 = (uint16_t)(_f16x2_13 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_13));
                        float sf_rounded_11 = _fp8_rt_1;
                        float _rcp_4 = approx_rcp(sf_rounded_11 * global_scale_rcp);
                        float _min_2 = fminf(_rcp_4, 3.4028234663852886e+38f);
                        float output_scale_12 = _min_2;
                        float2 _f2_25 = make_float2(output_scale_12, output_scale_12);
                        #pragma unroll
                        for (int j_21 = 0; j_21 < 8; j_21++) {
                            float2 _f2_26 = make_float2(values_5[2 * j_21], values_5[2 * j_21 + 1]);
                            float2 _mul_f32x2_13;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&_f2_26), "l"(*(const unsigned long long*)&_f2_25));
                            quant_values_7[2 * j_21] = _mul_f32x2_13.x;
                            quant_values_7[2 * j_21 + 1] = _mul_f32x2_13.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_8[0]) : "f"(quant_values_7[0]), "f"(quant_values_7[1]), "f"(quant_values_7[2]), "f"(quant_values_7[3]), "f"(quant_values_7[4]), "f"(quant_values_7[5]), "f"(quant_values_7[6]), "f"(quant_values_7[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_8[1]) : "f"(quant_values_7[8]), "f"(quant_values_7[9]), "f"(quant_values_7[10]), "f"(quant_values_7[11]), "f"(quant_values_7[12]), "f"(quant_values_7[13]), "f"(quant_values_7[14]), "f"(quant_values_7[15]));
                        row_out[2] = packed_8[0];
                        row_out[3] = packed_8[1];
                        sf_values[1] = sf_value_10;
                        float values_13[16];
                        float absolute_14[16];
                        float quant_values_15[16];
                        unsigned int packed_16[2];
                        #pragma unroll
                        for (int j_22 = 0; j_22 < 8; j_22++) {
                            values_13[2 * j_22] = __uint_as_float(normalized[16 + j_22] << 16);
                            values_13[2 * j_22 + 1] = __uint_as_float(normalized[16 + j_22] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_23 = 0; j_23 < 16; j_23++) {
                            absolute_14[j_23] = values_13[j_23];
                        }
                        float _fabs_32 = fabsf(absolute_14[0]);
                        absolute_14[0] = _fabs_32;
                        float _fabs_33 = fabsf(absolute_14[1]);
                        absolute_14[1] = _fabs_33;
                        float _fabs_34 = fabsf(absolute_14[2]);
                        absolute_14[2] = _fabs_34;
                        float _fabs_35 = fabsf(absolute_14[3]);
                        absolute_14[3] = _fabs_35;
                        float _fabs_36 = fabsf(absolute_14[4]);
                        absolute_14[4] = _fabs_36;
                        float _fabs_37 = fabsf(absolute_14[5]);
                        absolute_14[5] = _fabs_37;
                        float _fabs_38 = fabsf(absolute_14[6]);
                        absolute_14[6] = _fabs_38;
                        float _fabs_39 = fabsf(absolute_14[7]);
                        absolute_14[7] = _fabs_39;
                        float _fabs_40 = fabsf(absolute_14[8]);
                        absolute_14[8] = _fabs_40;
                        float _fabs_41 = fabsf(absolute_14[9]);
                        absolute_14[9] = _fabs_41;
                        float _fabs_42 = fabsf(absolute_14[10]);
                        absolute_14[10] = _fabs_42;
                        float _fabs_43 = fabsf(absolute_14[11]);
                        absolute_14[11] = _fabs_43;
                        float _fabs_44 = fabsf(absolute_14[12]);
                        absolute_14[12] = _fabs_44;
                        float _fabs_45 = fabsf(absolute_14[13]);
                        absolute_14[13] = _fabs_45;
                        float _fabs_46 = fabsf(absolute_14[14]);
                        absolute_14[14] = _fabs_46;
                        float _fabs_47 = fabsf(absolute_14[15]);
                        absolute_14[15] = _fabs_47;
                        float absolute_14_max = absolute_14[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_14_max = max_noftz(absolute_14_max, absolute_14[_lr]);
                        }
                        float amax_17 = absolute_14_max;
                        float _rcp_5 = approx_rcp(6.0f);
                        float sf_value_18 = global_scale * (amax_17 * _rcp_5);
                        float _fp8_rt_2;
                        uint16_t _e4m3x2_14;
                        uint32_t _f16x2_14;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_14) : "f"(0.0f), "f"(sf_value_18));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_14) : "h"(_e4m3x2_14));
                        uint16_t _fp8_h0_14 = (uint16_t)(_f16x2_14 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_2) : "h"(_fp8_h0_14));
                        float sf_rounded_19 = _fp8_rt_2;
                        float _rcp_6 = approx_rcp(sf_rounded_19 * global_scale_rcp);
                        float _min_3 = fminf(_rcp_6, 3.4028234663852886e+38f);
                        float output_scale_20 = _min_3;
                        float2 _f2_27 = make_float2(output_scale_20, output_scale_20);
                        #pragma unroll
                        for (int j_24 = 0; j_24 < 8; j_24++) {
                            float2 _f2_28 = make_float2(values_13[2 * j_24], values_13[2 * j_24 + 1]);
                            float2 _mul_f32x2_14;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_28), "l"(*(const unsigned long long*)&_f2_27));
                            quant_values_15[2 * j_24] = _mul_f32x2_14.x;
                            quant_values_15[2 * j_24 + 1] = _mul_f32x2_14.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_16[0]) : "f"(quant_values_15[0]), "f"(quant_values_15[1]), "f"(quant_values_15[2]), "f"(quant_values_15[3]), "f"(quant_values_15[4]), "f"(quant_values_15[5]), "f"(quant_values_15[6]), "f"(quant_values_15[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_16[1]) : "f"(quant_values_15[8]), "f"(quant_values_15[9]), "f"(quant_values_15[10]), "f"(quant_values_15[11]), "f"(quant_values_15[12]), "f"(quant_values_15[13]), "f"(quant_values_15[14]), "f"(quant_values_15[15]));
                        row_out[4] = packed_16[0];
                        row_out[5] = packed_16[1];
                        sf_values[2] = sf_value_18;
                        float values_21[16];
                        float absolute_22[16];
                        float quant_values_23[16];
                        unsigned int packed_24[2];
                        #pragma unroll
                        for (int j_25 = 0; j_25 < 8; j_25++) {
                            values_21[2 * j_25] = __uint_as_float(normalized[24 + j_25] << 16);
                            values_21[2 * j_25 + 1] = __uint_as_float(normalized[24 + j_25] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_26 = 0; j_26 < 16; j_26++) {
                            absolute_22[j_26] = values_21[j_26];
                        }
                        float _fabs_48 = fabsf(absolute_22[0]);
                        absolute_22[0] = _fabs_48;
                        float _fabs_49 = fabsf(absolute_22[1]);
                        absolute_22[1] = _fabs_49;
                        float _fabs_50 = fabsf(absolute_22[2]);
                        absolute_22[2] = _fabs_50;
                        float _fabs_51 = fabsf(absolute_22[3]);
                        absolute_22[3] = _fabs_51;
                        float _fabs_52 = fabsf(absolute_22[4]);
                        absolute_22[4] = _fabs_52;
                        float _fabs_53 = fabsf(absolute_22[5]);
                        absolute_22[5] = _fabs_53;
                        float _fabs_54 = fabsf(absolute_22[6]);
                        absolute_22[6] = _fabs_54;
                        float _fabs_55 = fabsf(absolute_22[7]);
                        absolute_22[7] = _fabs_55;
                        float _fabs_56 = fabsf(absolute_22[8]);
                        absolute_22[8] = _fabs_56;
                        float _fabs_57 = fabsf(absolute_22[9]);
                        absolute_22[9] = _fabs_57;
                        float _fabs_58 = fabsf(absolute_22[10]);
                        absolute_22[10] = _fabs_58;
                        float _fabs_59 = fabsf(absolute_22[11]);
                        absolute_22[11] = _fabs_59;
                        float _fabs_60 = fabsf(absolute_22[12]);
                        absolute_22[12] = _fabs_60;
                        float _fabs_61 = fabsf(absolute_22[13]);
                        absolute_22[13] = _fabs_61;
                        float _fabs_62 = fabsf(absolute_22[14]);
                        absolute_22[14] = _fabs_62;
                        float _fabs_63 = fabsf(absolute_22[15]);
                        absolute_22[15] = _fabs_63;
                        float absolute_22_max = absolute_22[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_22_max = max_noftz(absolute_22_max, absolute_22[_lr]);
                        }
                        float amax_25 = absolute_22_max;
                        float _rcp_7 = approx_rcp(6.0f);
                        float sf_value_26 = global_scale * (amax_25 * _rcp_7);
                        float _fp8_rt_3;
                        uint16_t _e4m3x2_15;
                        uint32_t _f16x2_15;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_15) : "f"(0.0f), "f"(sf_value_26));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_15) : "h"(_e4m3x2_15));
                        uint16_t _fp8_h0_15 = (uint16_t)(_f16x2_15 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_3) : "h"(_fp8_h0_15));
                        float sf_rounded_27 = _fp8_rt_3;
                        float _rcp_8 = approx_rcp(sf_rounded_27 * global_scale_rcp);
                        float _min_4 = fminf(_rcp_8, 3.4028234663852886e+38f);
                        float output_scale_28 = _min_4;
                        float2 _f2_29 = make_float2(output_scale_28, output_scale_28);
                        #pragma unroll
                        for (int j_27 = 0; j_27 < 8; j_27++) {
                            float2 _f2_30 = make_float2(values_21[2 * j_27], values_21[2 * j_27 + 1]);
                            float2 _mul_f32x2_15;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&_f2_30), "l"(*(const unsigned long long*)&_f2_29));
                            quant_values_23[2 * j_27] = _mul_f32x2_15.x;
                            quant_values_23[2 * j_27 + 1] = _mul_f32x2_15.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_24[0]) : "f"(quant_values_23[0]), "f"(quant_values_23[1]), "f"(quant_values_23[2]), "f"(quant_values_23[3]), "f"(quant_values_23[4]), "f"(quant_values_23[5]), "f"(quant_values_23[6]), "f"(quant_values_23[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_24[1]) : "f"(quant_values_23[8]), "f"(quant_values_23[9]), "f"(quant_values_23[10]), "f"(quant_values_23[11]), "f"(quant_values_23[12]), "f"(quant_values_23[13]), "f"(quant_values_23[14]), "f"(quant_values_23[15]));
                        row_out[6] = packed_24[0];
                        row_out[7] = packed_24[1];
                        sf_values[3] = sf_value_26;
                        exch[exch_slot] = sf_values[3];
                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                        float recv0 = exch[exch_slot];
                        float sf_group_values[4];
                        sf_group_values[0] = sf_values[0];
                        sf_group_values[1] = sf_values[1];
                        sf_group_values[2] = recv0;
                        sf_group_values[3] = sf_values[2];
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
                                : "=r"(_packed) : "f"(sf_group_values[0]), "f"(sf_group_values[1]),
                                                   "f"(sf_group_values[2]), "f"(sf_group_values[3]));
                            sf_packed[0] = _packed;
                        }
                        if (token < M) {
                            *(reinterpret_cast<int*>(out_sf + scale_row_base) + (0)) = sf_packed[0];
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 ^ (staging_row * 64 >> 7 & 3) << 4))), "r"(row_out[0]), "r"(row_out[1]), "r"(row_out[2]), "r"(row_out[3]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 24 ^ (staging_row * 64 + 24 >> 7 & 3) << 4))), "r"(row_out[4]), "r"(row_out[5]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 32 ^ (staging_row * 64 + 32 >> 7 & 3) << 4))), "r"(row_out[6]), "r"(row_out[7]) : "memory");
                    } else {
                        unsigned int row_out_1[8];
                        float sf_values_1[4];
                        float values_1[16];
                        float absolute_1[16];
                        float quant_values_1[16];
                        unsigned int packed_1[2];
                        #pragma unroll
                        for (int j_28 = 0; j_28 < 8; j_28++) {
                            values_1[2 * j_28] = __uint_as_float(words[j_28] << 16);
                            values_1[2 * j_28 + 1] = __uint_as_float(words[j_28] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_29 = 0; j_29 < 16; j_29++) {
                            absolute_1[j_29] = values_1[j_29];
                        }
                        float _fabs_64 = fabsf(absolute_1[0]);
                        absolute_1[0] = _fabs_64;
                        float _fabs_65 = fabsf(absolute_1[1]);
                        absolute_1[1] = _fabs_65;
                        float _fabs_66 = fabsf(absolute_1[2]);
                        absolute_1[2] = _fabs_66;
                        float _fabs_67 = fabsf(absolute_1[3]);
                        absolute_1[3] = _fabs_67;
                        float _fabs_68 = fabsf(absolute_1[4]);
                        absolute_1[4] = _fabs_68;
                        float _fabs_69 = fabsf(absolute_1[5]);
                        absolute_1[5] = _fabs_69;
                        float _fabs_70 = fabsf(absolute_1[6]);
                        absolute_1[6] = _fabs_70;
                        float _fabs_71 = fabsf(absolute_1[7]);
                        absolute_1[7] = _fabs_71;
                        float _fabs_72 = fabsf(absolute_1[8]);
                        absolute_1[8] = _fabs_72;
                        float _fabs_73 = fabsf(absolute_1[9]);
                        absolute_1[9] = _fabs_73;
                        float _fabs_74 = fabsf(absolute_1[10]);
                        absolute_1[10] = _fabs_74;
                        float _fabs_75 = fabsf(absolute_1[11]);
                        absolute_1[11] = _fabs_75;
                        float _fabs_76 = fabsf(absolute_1[12]);
                        absolute_1[12] = _fabs_76;
                        float _fabs_77 = fabsf(absolute_1[13]);
                        absolute_1[13] = _fabs_77;
                        float _fabs_78 = fabsf(absolute_1[14]);
                        absolute_1[14] = _fabs_78;
                        float _fabs_79 = fabsf(absolute_1[15]);
                        absolute_1[15] = _fabs_79;
                        float absolute_max_1 = absolute_1[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_max_1 = max_noftz(absolute_max_1, absolute_1[_lr]);
                        }
                        float amax_1 = absolute_max_1;
                        float _rcp_9 = approx_rcp(6.0f);
                        float sf_value_1 = global_scale * (amax_1 * _rcp_9);
                        float _fp8_rt_4;
                        uint16_t _e4m3x2_16;
                        uint32_t _f16x2_16;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_16) : "f"(0.0f), "f"(sf_value_1));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_16) : "h"(_e4m3x2_16));
                        uint16_t _fp8_h0_16 = (uint16_t)(_f16x2_16 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_4) : "h"(_fp8_h0_16));
                        float sf_rounded_1 = _fp8_rt_4;
                        float _rcp_10 = approx_rcp(sf_rounded_1 * global_scale_rcp);
                        float _min_5 = fminf(_rcp_10, 3.4028234663852886e+38f);
                        float output_scale_1 = _min_5;
                        float2 _f2_31 = make_float2(output_scale_1, output_scale_1);
                        #pragma unroll
                        for (int j_30 = 0; j_30 < 8; j_30++) {
                            float2 _f2_32 = make_float2(values_1[2 * j_30], values_1[2 * j_30 + 1]);
                            float2 _mul_f32x2_16;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&_f2_32), "l"(*(const unsigned long long*)&_f2_31));
                            quant_values_1[2 * j_30] = _mul_f32x2_16.x;
                            quant_values_1[2 * j_30 + 1] = _mul_f32x2_16.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_1[0]) : "f"(quant_values_1[0]), "f"(quant_values_1[1]), "f"(quant_values_1[2]), "f"(quant_values_1[3]), "f"(quant_values_1[4]), "f"(quant_values_1[5]), "f"(quant_values_1[6]), "f"(quant_values_1[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_1[1]) : "f"(quant_values_1[8]), "f"(quant_values_1[9]), "f"(quant_values_1[10]), "f"(quant_values_1[11]), "f"(quant_values_1[12]), "f"(quant_values_1[13]), "f"(quant_values_1[14]), "f"(quant_values_1[15]));
                        row_out_1[0] = packed_1[0];
                        row_out_1[1] = packed_1[1];
                        sf_values_1[0] = sf_value_1;
                        float values_0[16];
                        float absolute_1_1[16];
                        float quant_values_2[16];
                        unsigned int packed_3[2];
                        #pragma unroll
                        for (int j_31 = 0; j_31 < 8; j_31++) {
                            values_0[2 * j_31] = __uint_as_float(words[8 + j_31] << 16);
                            values_0[2 * j_31 + 1] = __uint_as_float(words[8 + j_31] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_32 = 0; j_32 < 16; j_32++) {
                            absolute_1_1[j_32] = values_0[j_32];
                        }
                        float _fabs_80 = fabsf(absolute_1_1[0]);
                        absolute_1_1[0] = _fabs_80;
                        float _fabs_81 = fabsf(absolute_1_1[1]);
                        absolute_1_1[1] = _fabs_81;
                        float _fabs_82 = fabsf(absolute_1_1[2]);
                        absolute_1_1[2] = _fabs_82;
                        float _fabs_83 = fabsf(absolute_1_1[3]);
                        absolute_1_1[3] = _fabs_83;
                        float _fabs_84 = fabsf(absolute_1_1[4]);
                        absolute_1_1[4] = _fabs_84;
                        float _fabs_85 = fabsf(absolute_1_1[5]);
                        absolute_1_1[5] = _fabs_85;
                        float _fabs_86 = fabsf(absolute_1_1[6]);
                        absolute_1_1[6] = _fabs_86;
                        float _fabs_87 = fabsf(absolute_1_1[7]);
                        absolute_1_1[7] = _fabs_87;
                        float _fabs_88 = fabsf(absolute_1_1[8]);
                        absolute_1_1[8] = _fabs_88;
                        float _fabs_89 = fabsf(absolute_1_1[9]);
                        absolute_1_1[9] = _fabs_89;
                        float _fabs_90 = fabsf(absolute_1_1[10]);
                        absolute_1_1[10] = _fabs_90;
                        float _fabs_91 = fabsf(absolute_1_1[11]);
                        absolute_1_1[11] = _fabs_91;
                        float _fabs_92 = fabsf(absolute_1_1[12]);
                        absolute_1_1[12] = _fabs_92;
                        float _fabs_93 = fabsf(absolute_1_1[13]);
                        absolute_1_1[13] = _fabs_93;
                        float _fabs_94 = fabsf(absolute_1_1[14]);
                        absolute_1_1[14] = _fabs_94;
                        float _fabs_95 = fabsf(absolute_1_1[15]);
                        absolute_1_1[15] = _fabs_95;
                        float absolute_1_max = absolute_1_1[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_1_max = max_noftz(absolute_1_max, absolute_1_1[_lr]);
                        }
                        float amax_4 = absolute_1_max;
                        float _rcp_11 = approx_rcp(6.0f);
                        float sf_value_5 = global_scale * (amax_4 * _rcp_11);
                        float _fp8_rt_5;
                        uint16_t _e4m3x2_17;
                        uint32_t _f16x2_17;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_17) : "f"(0.0f), "f"(sf_value_5));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_17) : "h"(_e4m3x2_17));
                        uint16_t _fp8_h0_17 = (uint16_t)(_f16x2_17 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_5) : "h"(_fp8_h0_17));
                        float sf_rounded_6 = _fp8_rt_5;
                        float _rcp_12 = approx_rcp(sf_rounded_6 * global_scale_rcp);
                        float _min_6 = fminf(_rcp_12, 3.4028234663852886e+38f);
                        float output_scale_7 = _min_6;
                        float2 _f2_33 = make_float2(output_scale_7, output_scale_7);
                        #pragma unroll
                        for (int j_33 = 0; j_33 < 8; j_33++) {
                            float2 _f2_34 = make_float2(values_0[2 * j_33], values_0[2 * j_33 + 1]);
                            float2 _mul_f32x2_17;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&_f2_34), "l"(*(const unsigned long long*)&_f2_33));
                            quant_values_2[2 * j_33] = _mul_f32x2_17.x;
                            quant_values_2[2 * j_33 + 1] = _mul_f32x2_17.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_3[0]) : "f"(quant_values_2[0]), "f"(quant_values_2[1]), "f"(quant_values_2[2]), "f"(quant_values_2[3]), "f"(quant_values_2[4]), "f"(quant_values_2[5]), "f"(quant_values_2[6]), "f"(quant_values_2[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_3[1]) : "f"(quant_values_2[8]), "f"(quant_values_2[9]), "f"(quant_values_2[10]), "f"(quant_values_2[11]), "f"(quant_values_2[12]), "f"(quant_values_2[13]), "f"(quant_values_2[14]), "f"(quant_values_2[15]));
                        row_out_1[2] = packed_3[0];
                        row_out_1[3] = packed_3[1];
                        sf_values_1[1] = sf_value_5;
                        float values_8[16];
                        float absolute_9[16];
                        float quant_values_10[16];
                        unsigned int packed_11[2];
                        #pragma unroll
                        for (int j_34 = 0; j_34 < 8; j_34++) {
                            values_8[2 * j_34] = __uint_as_float(words[16 + j_34] << 16);
                            values_8[2 * j_34 + 1] = __uint_as_float(words[16 + j_34] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_35 = 0; j_35 < 16; j_35++) {
                            absolute_9[j_35] = values_8[j_35];
                        }
                        float _fabs_96 = fabsf(absolute_9[0]);
                        absolute_9[0] = _fabs_96;
                        float _fabs_97 = fabsf(absolute_9[1]);
                        absolute_9[1] = _fabs_97;
                        float _fabs_98 = fabsf(absolute_9[2]);
                        absolute_9[2] = _fabs_98;
                        float _fabs_99 = fabsf(absolute_9[3]);
                        absolute_9[3] = _fabs_99;
                        float _fabs_100 = fabsf(absolute_9[4]);
                        absolute_9[4] = _fabs_100;
                        float _fabs_101 = fabsf(absolute_9[5]);
                        absolute_9[5] = _fabs_101;
                        float _fabs_102 = fabsf(absolute_9[6]);
                        absolute_9[6] = _fabs_102;
                        float _fabs_103 = fabsf(absolute_9[7]);
                        absolute_9[7] = _fabs_103;
                        float _fabs_104 = fabsf(absolute_9[8]);
                        absolute_9[8] = _fabs_104;
                        float _fabs_105 = fabsf(absolute_9[9]);
                        absolute_9[9] = _fabs_105;
                        float _fabs_106 = fabsf(absolute_9[10]);
                        absolute_9[10] = _fabs_106;
                        float _fabs_107 = fabsf(absolute_9[11]);
                        absolute_9[11] = _fabs_107;
                        float _fabs_108 = fabsf(absolute_9[12]);
                        absolute_9[12] = _fabs_108;
                        float _fabs_109 = fabsf(absolute_9[13]);
                        absolute_9[13] = _fabs_109;
                        float _fabs_110 = fabsf(absolute_9[14]);
                        absolute_9[14] = _fabs_110;
                        float _fabs_111 = fabsf(absolute_9[15]);
                        absolute_9[15] = _fabs_111;
                        float absolute_9_max = absolute_9[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_9_max = max_noftz(absolute_9_max, absolute_9[_lr]);
                        }
                        float amax_12 = absolute_9_max;
                        float _rcp_13 = approx_rcp(6.0f);
                        float sf_value_13 = global_scale * (amax_12 * _rcp_13);
                        float _fp8_rt_6;
                        uint16_t _e4m3x2_18;
                        uint32_t _f16x2_18;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_18) : "f"(0.0f), "f"(sf_value_13));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_18) : "h"(_e4m3x2_18));
                        uint16_t _fp8_h0_18 = (uint16_t)(_f16x2_18 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_6) : "h"(_fp8_h0_18));
                        float sf_rounded_14 = _fp8_rt_6;
                        float _rcp_14 = approx_rcp(sf_rounded_14 * global_scale_rcp);
                        float _min_7 = fminf(_rcp_14, 3.4028234663852886e+38f);
                        float output_scale_15 = _min_7;
                        float2 _f2_35 = make_float2(output_scale_15, output_scale_15);
                        #pragma unroll
                        for (int j_36 = 0; j_36 < 8; j_36++) {
                            float2 _f2_36 = make_float2(values_8[2 * j_36], values_8[2 * j_36 + 1]);
                            float2 _mul_f32x2_18;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&_f2_36), "l"(*(const unsigned long long*)&_f2_35));
                            quant_values_10[2 * j_36] = _mul_f32x2_18.x;
                            quant_values_10[2 * j_36 + 1] = _mul_f32x2_18.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_11[0]) : "f"(quant_values_10[0]), "f"(quant_values_10[1]), "f"(quant_values_10[2]), "f"(quant_values_10[3]), "f"(quant_values_10[4]), "f"(quant_values_10[5]), "f"(quant_values_10[6]), "f"(quant_values_10[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_11[1]) : "f"(quant_values_10[8]), "f"(quant_values_10[9]), "f"(quant_values_10[10]), "f"(quant_values_10[11]), "f"(quant_values_10[12]), "f"(quant_values_10[13]), "f"(quant_values_10[14]), "f"(quant_values_10[15]));
                        row_out_1[4] = packed_11[0];
                        row_out_1[5] = packed_11[1];
                        sf_values_1[2] = sf_value_13;
                        float values_16[16];
                        float absolute_17[16];
                        float quant_values_18[16];
                        unsigned int packed_19[2];
                        #pragma unroll
                        for (int j_37 = 0; j_37 < 8; j_37++) {
                            values_16[2 * j_37] = __uint_as_float(words[24 + j_37] << 16);
                            values_16[2 * j_37 + 1] = __uint_as_float(words[24 + j_37] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_38 = 0; j_38 < 16; j_38++) {
                            absolute_17[j_38] = values_16[j_38];
                        }
                        float _fabs_112 = fabsf(absolute_17[0]);
                        absolute_17[0] = _fabs_112;
                        float _fabs_113 = fabsf(absolute_17[1]);
                        absolute_17[1] = _fabs_113;
                        float _fabs_114 = fabsf(absolute_17[2]);
                        absolute_17[2] = _fabs_114;
                        float _fabs_115 = fabsf(absolute_17[3]);
                        absolute_17[3] = _fabs_115;
                        float _fabs_116 = fabsf(absolute_17[4]);
                        absolute_17[4] = _fabs_116;
                        float _fabs_117 = fabsf(absolute_17[5]);
                        absolute_17[5] = _fabs_117;
                        float _fabs_118 = fabsf(absolute_17[6]);
                        absolute_17[6] = _fabs_118;
                        float _fabs_119 = fabsf(absolute_17[7]);
                        absolute_17[7] = _fabs_119;
                        float _fabs_120 = fabsf(absolute_17[8]);
                        absolute_17[8] = _fabs_120;
                        float _fabs_121 = fabsf(absolute_17[9]);
                        absolute_17[9] = _fabs_121;
                        float _fabs_122 = fabsf(absolute_17[10]);
                        absolute_17[10] = _fabs_122;
                        float _fabs_123 = fabsf(absolute_17[11]);
                        absolute_17[11] = _fabs_123;
                        float _fabs_124 = fabsf(absolute_17[12]);
                        absolute_17[12] = _fabs_124;
                        float _fabs_125 = fabsf(absolute_17[13]);
                        absolute_17[13] = _fabs_125;
                        float _fabs_126 = fabsf(absolute_17[14]);
                        absolute_17[14] = _fabs_126;
                        float _fabs_127 = fabsf(absolute_17[15]);
                        absolute_17[15] = _fabs_127;
                        float absolute_17_max = absolute_17[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_17_max = max_noftz(absolute_17_max, absolute_17[_lr]);
                        }
                        float amax_20 = absolute_17_max;
                        float _rcp_15 = approx_rcp(6.0f);
                        float sf_value_21 = global_scale * (amax_20 * _rcp_15);
                        float _fp8_rt_7;
                        uint16_t _e4m3x2_19;
                        uint32_t _f16x2_19;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_19) : "f"(0.0f), "f"(sf_value_21));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_19) : "h"(_e4m3x2_19));
                        uint16_t _fp8_h0_19 = (uint16_t)(_f16x2_19 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_7) : "h"(_fp8_h0_19));
                        float sf_rounded_22 = _fp8_rt_7;
                        float _rcp_16 = approx_rcp(sf_rounded_22 * global_scale_rcp);
                        float _min_8 = fminf(_rcp_16, 3.4028234663852886e+38f);
                        float output_scale_23 = _min_8;
                        float2 _f2_37 = make_float2(output_scale_23, output_scale_23);
                        #pragma unroll
                        for (int j_39 = 0; j_39 < 8; j_39++) {
                            float2 _f2_38 = make_float2(values_16[2 * j_39], values_16[2 * j_39 + 1]);
                            float2 _mul_f32x2_19;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&_f2_38), "l"(*(const unsigned long long*)&_f2_37));
                            quant_values_18[2 * j_39] = _mul_f32x2_19.x;
                            quant_values_18[2 * j_39 + 1] = _mul_f32x2_19.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_19[0]) : "f"(quant_values_18[0]), "f"(quant_values_18[1]), "f"(quant_values_18[2]), "f"(quant_values_18[3]), "f"(quant_values_18[4]), "f"(quant_values_18[5]), "f"(quant_values_18[6]), "f"(quant_values_18[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_19[1]) : "f"(quant_values_18[8]), "f"(quant_values_18[9]), "f"(quant_values_18[10]), "f"(quant_values_18[11]), "f"(quant_values_18[12]), "f"(quant_values_18[13]), "f"(quant_values_18[14]), "f"(quant_values_18[15]));
                        row_out_1[6] = packed_19[0];
                        row_out_1[7] = packed_19[1];
                        sf_values_1[3] = sf_value_21;
                        exch[exch_slot] = sf_values_1[3];
                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                        float recv0_1 = exch[exch_slot];
                        float sf_group_values_1[4];
                        sf_group_values_1[0] = sf_values_1[0];
                        sf_group_values_1[1] = sf_values_1[1];
                        sf_group_values_1[2] = recv0_1;
                        sf_group_values_1[3] = sf_values_1[2];
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
                                : "=r"(_packed) : "f"(sf_group_values_1[0]), "f"(sf_group_values_1[1]),
                                                   "f"(sf_group_values_1[2]), "f"(sf_group_values_1[3]));
                            sf_packed_1[0] = _packed;
                        }
                        if (token < M) {
                            *(reinterpret_cast<int*>(out_sf + scale_row_base) + (0)) = sf_packed_1[0];
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 ^ (staging_row * 64 >> 7 & 3) << 4))), "r"(row_out_1[0]), "r"(row_out_1[1]), "r"(row_out_1[2]), "r"(row_out_1[3]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 24 ^ (staging_row * 64 + 24 >> 7 & 3) << 4))), "r"(row_out_1[4]), "r"(row_out_1[5]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 32 ^ (staging_row * 64 + 32 >> 7 & 3) << 4))), "r"(row_out_1[6]), "r"(row_out_1[7]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync %0, 256;" :: "r"(store_bar) : "memory");
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            tma_store_4d(OUTQ, 0, head_kind_local, off_m_1, destination, epi_staging_addr + staging_buf * 8192);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    tile_parity = tile_parity ^ 1;
                }
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
    // ---- Role: epilogue_b ----
    if (warp >= 12 && warp <= 19) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // epilogue_b_main
            unsigned int acc_stage_2 = 0;
            unsigned int work_stage_3 = 0;
            const int epi_index_1 = warp - 4 - 8;
            const int epi_warp_1 = epi_index_1 % 4;
            const int head_half_1 = epi_index_1 / 4;
            const int local_row_1 = epi_warp_1 * 32 + lane;
            int exch_slot_1 = (head_half_1 * 128 + local_row_1) * 3;
            int pair_bar_1 = 1 + head_half_1 * 4 + epi_warp_1;
            float alpha_v_1 = alpha[0];
            float global_scale_1 = out_global_scale[0];
            float _rcp_17 = approx_rcp(global_scale_1);
            float global_scale_rcp_1 = _rcp_17;
            int rows_per_token_1 = HEADS_PER_DESTINATION * 3;
            unsigned int this_bid_2 = bid;
            unsigned int tile_parity_1 = 0;
            int store_bar_1 = 14 + head_half_1;
            unsigned int _phase_mainloop_done_1 = 0;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < num_cluster_tiles; _tile_iter_3++) {
                mbarrier_wait(mainloop_done_addr + (acc_stage_2) * 8, _phase_mainloop_done_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int group_2 = this_bid_2 / (unsigned int)tiles_per_group;
                int first_m_2 = group_2 * GROUP_M;
                int remaining_2 = m_tiles - first_m_2;
                int group_size_2 = ((remaining_2 >= GROUP_M) ? GROUP_M : remaining_2);
                int local_2 = this_bid_2 % (unsigned int)tiles_per_group;
                int bid_m_2 = first_m_2 + local_2 % group_size_2;
                int bid_n_2 = local_2 / group_size_2;
                int off_m_2 = bid_m_2 * BLOCK_M;
                int off_n_2 = bid_n_2 * BLOCK_N;
                int token_1 = off_m_2 + local_row_1;
                int head_kind_1 = bid_n_2 * 2 + head_half_1;
                int head_1 = head_kind_1 / 3;
                int kind_1 = head_kind_1 % 3;
                int lane_addr_1 = taddr + (unsigned int)(epi_warp_1 * 32 << 16) + (unsigned int)(head_half_1 * B_HALF_N);
                unsigned int words_1[32];
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], lane_addr_1 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_40 = 0; j_40 < 8; j_40++) {
                    __nv_bfloat162 _bf16x2_12 = __float22bfloat162_rn(make_float2(_tmem_load_4[2 * j_40] * alpha_v_1, _tmem_load_4[2 * j_40 + 1] * alpha_v_1));
                    words_1[j_40] = __as_u32(_bf16x2_12);
                }
                float _tmem_load_5[16];
                tmem_ld_x16(&_tmem_load_5[0], lane_addr_1 + 80);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_41 = 0; j_41 < 8; j_41++) {
                    __nv_bfloat162 _bf16x2_13 = __float22bfloat162_rn(make_float2(_tmem_load_5[2 * j_41] * alpha_v_1, _tmem_load_5[2 * j_41 + 1] * alpha_v_1));
                    words_1[8 + j_41] = __as_u32(_bf16x2_13);
                }
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], lane_addr_1 + 96);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_42 = 0; j_42 < 8; j_42++) {
                    __nv_bfloat162 _bf16x2_14 = __float22bfloat162_rn(make_float2(_tmem_load_6[2 * j_42] * alpha_v_1, _tmem_load_6[2 * j_42 + 1] * alpha_v_1));
                    words_1[16 + j_42] = __as_u32(_bf16x2_14);
                }
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], lane_addr_1 + 112);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_43 = 0; j_43 < 8; j_43++) {
                    __nv_bfloat162 _bf16x2_15 = __float22bfloat162_rn(make_float2(_tmem_load_7[2 * j_43] * alpha_v_1, _tmem_load_7[2 * j_43 + 1] * alpha_v_1));
                    words_1[24 + j_43] = __as_u32(_bf16x2_15);
                }
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (acc_stage_2) * 8) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done_1 ^= 1;
                {
                    int destination_1 = head_1 / HEADS_PER_DESTINATION;
                    int local_head_1 = head_1 % HEADS_PER_DESTINATION;
                    int head_kind_local_1 = local_head_1 * 3 + kind_1;
                    int row_in_destination_1 = token_1 * rows_per_token_1 + head_kind_local_1;
                    unsigned int sr_1 = (unsigned int)row_in_destination_1;
                    unsigned int scale_swizzle_1 = sr_1 >> 7 << 10 | (sr_1 & 31) << 4 | (sr_1 >> 5 & 3) << 2;
                    unsigned long long scale_row_base_1 = (unsigned long long)destination_1 * (unsigned long long)SCALE_STRIDE + (unsigned long long)scale_swizzle_1;
                    unsigned int staging_buf_1 = tile_parity_1 << 1 | (unsigned int)head_half_1;
                    int staging_row_1 = (int)staging_buf_1 * 128 + local_row_1;
                    asm volatile("barrier.sync %0, 256;" :: "r"(store_bar_1) : "memory");
                    if (kind_1 < 2) {
                        float partial_1[4];
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 4; i_1++) {
                            float sum_lo_1 = 0.0f;
                            #pragma unroll
                            for (int j_44 = 0; j_44 < 4; j_44++) {
                                float v_lo_1 = __uint_as_float(words_1[i_1 * 8 + j_44] << 16);
                                float v_hi_1 = __uint_as_float(words_1[i_1 * 8 + j_44] & 4294901760);
                                float _fma_4 = __fmaf_rn(v_lo_1, v_lo_1, sum_lo_1);
                                sum_lo_1 = _fma_4;
                                float _fma_5 = __fmaf_rn(v_hi_1, v_hi_1, sum_lo_1);
                                sum_lo_1 = _fma_5;
                            }
                            float sum_hi_1 = 0.0f;
                            #pragma unroll
                            for (int j_45 = 4; j_45 < 8; j_45++) {
                                float v_lo2_1 = __uint_as_float(words_1[i_1 * 8 + j_45] << 16);
                                float v_hi2_1 = __uint_as_float(words_1[i_1 * 8 + j_45] & 4294901760);
                                float _fma_6 = __fmaf_rn(v_lo2_1, v_lo2_1, sum_hi_1);
                                sum_hi_1 = _fma_6;
                                float _fma_7 = __fmaf_rn(v_hi2_1, v_hi2_1, sum_hi_1);
                                sum_hi_1 = _fma_7;
                            }
                            partial_1[i_1] = sum_lo_1 + sum_hi_1;
                        }
                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar_1) : "memory");
                        float s01_1 = exch[exch_slot_1];
                        float p3 = exch[exch_slot_1 + 1];
                        float p4 = exch[exch_slot_1 + 2];
                        float s67_1 = partial_1[2] + partial_1[3];
                        exch[exch_slot_1] = partial_1[0];
                        exch[exch_slot_1 + 1] = partial_1[1];
                        exch[exch_slot_1 + 2] = s67_1;
                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar_1) : "memory");
                        float s23_1 = partial_1[0] + p3;
                        float s45_1 = p4 + partial_1[1];
                        float sum_sq_1 = s01_1 + s23_1 + (s45_1 + s67_1);
                        float _fdiv_rn_1 = __fdiv_rn(sum_sq_1, 128.0f);
                        float mean_sq_1 = _fdiv_rn_1;
                        float _rsqrt_1 = rsqrtf(mean_sq_1 + eps);
                        float rstd_1 = _rsqrt_1;
                        float2 _f2_39 = make_float2(rstd_1, rstd_1);
                        unsigned int normalized_1[32];
                        unsigned int weight_words_3[8];
                        #pragma unroll
                        for (int h_6 = 0; h_6 < 2; h_6++) {
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(((kind_1 == 0) ? q_norm_weight : k_norm_weight) + 32 + h_6 * 8);
                                uint4* _vdst_0 = reinterpret_cast<uint4*>(&weight_words_3[4 * h_6]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_0[_blk] = _vptr_0[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_46 = 0; j_46 < 8; j_46++) {
                            float2 _f2_40 = make_float2(__uint_as_float(words_1[j_46] << 16), __uint_as_float(words_1[j_46] & 4294901760));
                            float2 _mul_f32x2_20;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_20) : "l"(*(const unsigned long long*)&_f2_40), "l"(*(const unsigned long long*)&_f2_39));
                            float2 _f2_41 = make_float2(__uint_as_float(weight_words_3[j_46] << 16), __uint_as_float(weight_words_3[j_46] & 4294901760));
                            float2 _mul_f32x2_21;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_21) : "l"(*(const unsigned long long*)&_mul_f32x2_20), "l"(*(const unsigned long long*)&_f2_41));
                            __nv_bfloat162 _bf16x2_16 = __float22bfloat162_rn(make_float2(_mul_f32x2_21.x, _mul_f32x2_21.y));
                            normalized_1[j_46] = __as_u32(_bf16x2_16);
                        }
                        unsigned int weight_words_0_1[8];
                        #pragma unroll
                        for (int h_7 = 0; h_7 < 2; h_7++) {
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(((kind_1 == 0) ? q_norm_weight : k_norm_weight) + 80 + h_7 * 8);
                                uint4* _vdst_1 = reinterpret_cast<uint4*>(&weight_words_0_1[4 * h_7]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_1[_blk] = _vptr_1[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_47 = 0; j_47 < 8; j_47++) {
                            float2 _f2_42 = make_float2(__uint_as_float(words_1[8 + j_47] << 16), __uint_as_float(words_1[8 + j_47] & 4294901760));
                            float2 _mul_f32x2_22;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_22) : "l"(*(const unsigned long long*)&_f2_42), "l"(*(const unsigned long long*)&_f2_39));
                            float2 _f2_43 = make_float2(__uint_as_float(weight_words_0_1[j_47] << 16), __uint_as_float(weight_words_0_1[j_47] & 4294901760));
                            float2 _mul_f32x2_23;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_23) : "l"(*(const unsigned long long*)&_mul_f32x2_22), "l"(*(const unsigned long long*)&_f2_43));
                            __nv_bfloat162 _bf16x2_17 = __float22bfloat162_rn(make_float2(_mul_f32x2_23.x, _mul_f32x2_23.y));
                            normalized_1[8 + j_47] = __as_u32(_bf16x2_17);
                        }
                        unsigned int weight_words_1_1[8];
                        #pragma unroll
                        for (int h_8 = 0; h_8 < 2; h_8++) {
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(((kind_1 == 0) ? q_norm_weight : k_norm_weight) + 96 + h_8 * 8);
                                uint4* _vdst_2 = reinterpret_cast<uint4*>(&weight_words_1_1[4 * h_8]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_2[_blk] = _vptr_2[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_48 = 0; j_48 < 8; j_48++) {
                            float2 _f2_44 = make_float2(__uint_as_float(words_1[16 + j_48] << 16), __uint_as_float(words_1[16 + j_48] & 4294901760));
                            float2 _mul_f32x2_24;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_24) : "l"(*(const unsigned long long*)&_f2_44), "l"(*(const unsigned long long*)&_f2_39));
                            float2 _f2_45 = make_float2(__uint_as_float(weight_words_1_1[j_48] << 16), __uint_as_float(weight_words_1_1[j_48] & 4294901760));
                            float2 _mul_f32x2_25;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_25) : "l"(*(const unsigned long long*)&_mul_f32x2_24), "l"(*(const unsigned long long*)&_f2_45));
                            __nv_bfloat162 _bf16x2_18 = __float22bfloat162_rn(make_float2(_mul_f32x2_25.x, _mul_f32x2_25.y));
                            normalized_1[16 + j_48] = __as_u32(_bf16x2_18);
                        }
                        unsigned int weight_words_2_1[8];
                        #pragma unroll
                        for (int h_9 = 0; h_9 < 2; h_9++) {
                            {
                                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(((kind_1 == 0) ? q_norm_weight : k_norm_weight) + 112 + h_9 * 8);
                                uint4* _vdst_3 = reinterpret_cast<uint4*>(&weight_words_2_1[4 * h_9]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_3[_blk] = _vptr_3[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_49 = 0; j_49 < 8; j_49++) {
                            float2 _f2_46 = make_float2(__uint_as_float(words_1[24 + j_49] << 16), __uint_as_float(words_1[24 + j_49] & 4294901760));
                            float2 _mul_f32x2_26;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_26) : "l"(*(const unsigned long long*)&_f2_46), "l"(*(const unsigned long long*)&_f2_39));
                            float2 _f2_47 = make_float2(__uint_as_float(weight_words_2_1[j_49] << 16), __uint_as_float(weight_words_2_1[j_49] & 4294901760));
                            float2 _mul_f32x2_27;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_27) : "l"(*(const unsigned long long*)&_mul_f32x2_26), "l"(*(const unsigned long long*)&_f2_47));
                            __nv_bfloat162 _bf16x2_19 = __float22bfloat162_rn(make_float2(_mul_f32x2_27.x, _mul_f32x2_27.y));
                            normalized_1[24 + j_49] = __as_u32(_bf16x2_19);
                        }
                        int _min_9 = ((token_1) < (M - 1) ? (token_1) : (M - 1));
                        int rope_token_1 = _min_9;
                        unsigned int cos_words_1[8];
                        unsigned int sin_words_1[8];
                        #pragma unroll
                        for (int h_10 = 0; h_10 < 2; h_10++) {
                            {
                                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(rope_cos_sin + rope_token_1 * 96 + 32 + h_10 * 8);
                                uint4* _vdst_4 = reinterpret_cast<uint4*>(&cos_words_1[4 * h_10]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_4[_blk] = _vptr_4[_blk];
                                }
                            }
                            {
                                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(rope_cos_sin + rope_token_1 * 96 + 48 + 32 + h_10 * 8);
                                uint4* _vdst_5 = reinterpret_cast<uint4*>(&sin_words_1[4 * h_10]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_5[_blk] = _vptr_5[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_50 = 0; j_50 < 8; j_50++) {
                            unsigned int n_lo_word_2 = normalized_1[j_50];
                            unsigned int n_hi_word_2 = normalized_1[8 + j_50];
                            float2 _f2_48 = make_float2(__uint_as_float(cos_words_1[j_50] << 16), __uint_as_float(cos_words_1[j_50] & 4294901760));
                            float2 _f2_49 = make_float2(__uint_as_float(sin_words_1[j_50] << 16 ^ 2147483648), __uint_as_float(sin_words_1[j_50] & 4294901760 ^ 2147483648));
                            float2 _f2_50 = make_float2(__uint_as_float(sin_words_1[j_50] << 16), __uint_as_float(sin_words_1[j_50] & 4294901760));
                            float2 _f2_51 = make_float2(__uint_as_float(n_hi_word_2 << 16), __uint_as_float(n_hi_word_2 & 4294901760));
                            float2 _mul_f32x2_28;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_28) : "l"(*(const unsigned long long*)&_f2_51), "l"(*(const unsigned long long*)&_f2_49));
                            float2 _f2_52 = make_float2(__uint_as_float(n_lo_word_2 << 16), __uint_as_float(n_lo_word_2 & 4294901760));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_fma_f32x2_4;
                            {
                                float2 _packed_f32x2_6_0 = _f2_48;
                                float2 _packed_f32x2_6_1 = _f2_52;
                                float2 _packed_f32x2_6_2 = _mul_f32x2_28;
                                asm volatile("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(reinterpret_cast<uint64_t&>(_packed_fma_f32x2_4)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_6_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_6_1)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_6_2)));
                            }
                            float2 _f2_53 = make_float2(__uint_as_float(n_lo_word_2 << 16), __uint_as_float(n_lo_word_2 & 4294901760));
                            float2 _mul_f32x2_29;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_29) : "l"(*(const unsigned long long*)&_f2_53), "l"(*(const unsigned long long*)&_f2_50));
                            float2 _f2_54 = make_float2(__uint_as_float(n_hi_word_2 << 16), __uint_as_float(n_hi_word_2 & 4294901760));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_fma_f32x2_5;
                            {
                                float2 _packed_f32x2_7_0 = _f2_48;
                                float2 _packed_f32x2_7_1 = _f2_54;
                                float2 _packed_f32x2_7_2 = _mul_f32x2_29;
                                asm volatile("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(reinterpret_cast<uint64_t&>(_packed_fma_f32x2_5)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_7_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_7_1)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_7_2)));
                            }
                            __nv_bfloat162 _bf16x2_20 = __float22bfloat162_rn(make_float2(_packed_fma_f32x2_4.x, _packed_fma_f32x2_4.y));
                            normalized_1[j_50] = __as_u32(_bf16x2_20);
                            __nv_bfloat162 _bf16x2_21 = __float22bfloat162_rn(make_float2(_packed_fma_f32x2_5.x, _packed_fma_f32x2_5.y));
                            normalized_1[8 + j_50] = __as_u32(_bf16x2_21);
                        }
                        if (write_debug != 0) {
                            if (token_1 < M) {
                                unsigned long long debug_row_1 = ((unsigned long long)token_1 * 56 + (unsigned long long)head_1) * 64;
                                #pragma unroll
                                for (int c_4 = 0; c_4 < 2; c_4++) {
                                    unsigned int debug_chunk_4[4];
                                    #pragma unroll
                                    for (int j_51 = 0; j_51 < 4; j_51++) {
                                        debug_chunk_4[j_51] = normalized_1[4 * c_4 + j_51];
                                    }
                                    {
                                        int4 _iv4 = make_int4(debug_chunk_4[0 + 0], debug_chunk_4[0 + 1], debug_chunk_4[0 + 2], debug_chunk_4[0 + 3]);
                                        *reinterpret_cast<int4*>(((kind_1 == 0) ? debug_q_words : debug_k_words) + (debug_row_1 + 16 + 4 * c_4) + 0) = _iv4;
                                    }
                                }
                                #pragma unroll
                                for (int c_5 = 0; c_5 < 2; c_5++) {
                                    unsigned int debug_chunk_5[4];
                                    #pragma unroll
                                    for (int j_52 = 0; j_52 < 4; j_52++) {
                                        debug_chunk_5[j_52] = normalized_1[8 + 4 * c_5 + j_52];
                                    }
                                    {
                                        int4 _iv4 = make_int4(debug_chunk_5[0 + 0], debug_chunk_5[0 + 1], debug_chunk_5[0 + 2], debug_chunk_5[0 + 3]);
                                        *reinterpret_cast<int4*>(((kind_1 == 0) ? debug_q_words : debug_k_words) + (debug_row_1 + 40 + 4 * c_5) + 0) = _iv4;
                                    }
                                }
                                #pragma unroll
                                for (int c_6 = 0; c_6 < 2; c_6++) {
                                    unsigned int debug_chunk_6[4];
                                    #pragma unroll
                                    for (int j_53 = 0; j_53 < 4; j_53++) {
                                        debug_chunk_6[j_53] = normalized_1[16 + 4 * c_6 + j_53];
                                    }
                                    {
                                        int4 _iv4 = make_int4(debug_chunk_6[0 + 0], debug_chunk_6[0 + 1], debug_chunk_6[0 + 2], debug_chunk_6[0 + 3]);
                                        *reinterpret_cast<int4*>(((kind_1 == 0) ? debug_q_words : debug_k_words) + (debug_row_1 + 48 + 4 * c_6) + 0) = _iv4;
                                    }
                                }
                                #pragma unroll
                                for (int c_7 = 0; c_7 < 2; c_7++) {
                                    unsigned int debug_chunk_7[4];
                                    #pragma unroll
                                    for (int j_54 = 0; j_54 < 4; j_54++) {
                                        debug_chunk_7[j_54] = normalized_1[24 + 4 * c_7 + j_54];
                                    }
                                    {
                                        int4 _iv4 = make_int4(debug_chunk_7[0 + 0], debug_chunk_7[0 + 1], debug_chunk_7[0 + 2], debug_chunk_7[0 + 3]);
                                        *reinterpret_cast<int4*>(((kind_1 == 0) ? debug_q_words : debug_k_words) + (debug_row_1 + 56 + 4 * c_7) + 0) = _iv4;
                                    }
                                }
                            }
                        }
                        unsigned int row_out_2[8];
                        float sf_values_2[4];
                        float values_2[16];
                        float absolute_2[16];
                        float quant_values_3[16];
                        unsigned int packed_2[2];
                        #pragma unroll
                        for (int j_55 = 0; j_55 < 8; j_55++) {
                            values_2[2 * j_55] = __uint_as_float(normalized_1[j_55] << 16);
                            values_2[2 * j_55 + 1] = __uint_as_float(normalized_1[j_55] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_56 = 0; j_56 < 16; j_56++) {
                            absolute_2[j_56] = values_2[j_56];
                        }
                        float _fabs_128 = fabsf(absolute_2[0]);
                        absolute_2[0] = _fabs_128;
                        float _fabs_129 = fabsf(absolute_2[1]);
                        absolute_2[1] = _fabs_129;
                        float _fabs_130 = fabsf(absolute_2[2]);
                        absolute_2[2] = _fabs_130;
                        float _fabs_131 = fabsf(absolute_2[3]);
                        absolute_2[3] = _fabs_131;
                        float _fabs_132 = fabsf(absolute_2[4]);
                        absolute_2[4] = _fabs_132;
                        float _fabs_133 = fabsf(absolute_2[5]);
                        absolute_2[5] = _fabs_133;
                        float _fabs_134 = fabsf(absolute_2[6]);
                        absolute_2[6] = _fabs_134;
                        float _fabs_135 = fabsf(absolute_2[7]);
                        absolute_2[7] = _fabs_135;
                        float _fabs_136 = fabsf(absolute_2[8]);
                        absolute_2[8] = _fabs_136;
                        float _fabs_137 = fabsf(absolute_2[9]);
                        absolute_2[9] = _fabs_137;
                        float _fabs_138 = fabsf(absolute_2[10]);
                        absolute_2[10] = _fabs_138;
                        float _fabs_139 = fabsf(absolute_2[11]);
                        absolute_2[11] = _fabs_139;
                        float _fabs_140 = fabsf(absolute_2[12]);
                        absolute_2[12] = _fabs_140;
                        float _fabs_141 = fabsf(absolute_2[13]);
                        absolute_2[13] = _fabs_141;
                        float _fabs_142 = fabsf(absolute_2[14]);
                        absolute_2[14] = _fabs_142;
                        float _fabs_143 = fabsf(absolute_2[15]);
                        absolute_2[15] = _fabs_143;
                        float absolute_max_2 = absolute_2[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_max_2 = max_noftz(absolute_max_2, absolute_2[_lr]);
                        }
                        float amax_2 = absolute_max_2;
                        float _rcp_18 = approx_rcp(6.0f);
                        float sf_value_2 = global_scale_1 * (amax_2 * _rcp_18);
                        float _fp8_rt_8;
                        uint16_t _e4m3x2_8;
                        uint32_t _f16x2_8;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_8) : "f"(0.0f), "f"(sf_value_2));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_8) : "h"(_e4m3x2_8));
                        uint16_t _fp8_h0_8 = (uint16_t)(_f16x2_8 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_8) : "h"(_fp8_h0_8));
                        float sf_rounded_2 = _fp8_rt_8;
                        float _rcp_19 = approx_rcp(sf_rounded_2 * global_scale_rcp_1);
                        float _min_10 = fminf(_rcp_19, 3.4028234663852886e+38f);
                        float output_scale_2 = _min_10;
                        float2 _f2_55 = make_float2(output_scale_2, output_scale_2);
                        #pragma unroll
                        for (int j_57 = 0; j_57 < 8; j_57++) {
                            float2 _f2_56 = make_float2(values_2[2 * j_57], values_2[2 * j_57 + 1]);
                            float2 _mul_f32x2_30;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_30) : "l"(*(const unsigned long long*)&_f2_56), "l"(*(const unsigned long long*)&_f2_55));
                            quant_values_3[2 * j_57] = _mul_f32x2_30.x;
                            quant_values_3[2 * j_57 + 1] = _mul_f32x2_30.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_2[0]) : "f"(quant_values_3[0]), "f"(quant_values_3[1]), "f"(quant_values_3[2]), "f"(quant_values_3[3]), "f"(quant_values_3[4]), "f"(quant_values_3[5]), "f"(quant_values_3[6]), "f"(quant_values_3[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_2[1]) : "f"(quant_values_3[8]), "f"(quant_values_3[9]), "f"(quant_values_3[10]), "f"(quant_values_3[11]), "f"(quant_values_3[12]), "f"(quant_values_3[13]), "f"(quant_values_3[14]), "f"(quant_values_3[15]));
                        row_out_2[0] = packed_2[0];
                        row_out_2[1] = packed_2[1];
                        sf_values_2[0] = sf_value_2;
                        float values_3[16];
                        float absolute_4[16];
                        float quant_values_5[16];
                        unsigned int packed_6[2];
                        #pragma unroll
                        for (int j_58 = 0; j_58 < 8; j_58++) {
                            values_3[2 * j_58] = __uint_as_float(normalized_1[8 + j_58] << 16);
                            values_3[2 * j_58 + 1] = __uint_as_float(normalized_1[8 + j_58] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_59 = 0; j_59 < 16; j_59++) {
                            absolute_4[j_59] = values_3[j_59];
                        }
                        float _fabs_144 = fabsf(absolute_4[0]);
                        absolute_4[0] = _fabs_144;
                        float _fabs_145 = fabsf(absolute_4[1]);
                        absolute_4[1] = _fabs_145;
                        float _fabs_146 = fabsf(absolute_4[2]);
                        absolute_4[2] = _fabs_146;
                        float _fabs_147 = fabsf(absolute_4[3]);
                        absolute_4[3] = _fabs_147;
                        float _fabs_148 = fabsf(absolute_4[4]);
                        absolute_4[4] = _fabs_148;
                        float _fabs_149 = fabsf(absolute_4[5]);
                        absolute_4[5] = _fabs_149;
                        float _fabs_150 = fabsf(absolute_4[6]);
                        absolute_4[6] = _fabs_150;
                        float _fabs_151 = fabsf(absolute_4[7]);
                        absolute_4[7] = _fabs_151;
                        float _fabs_152 = fabsf(absolute_4[8]);
                        absolute_4[8] = _fabs_152;
                        float _fabs_153 = fabsf(absolute_4[9]);
                        absolute_4[9] = _fabs_153;
                        float _fabs_154 = fabsf(absolute_4[10]);
                        absolute_4[10] = _fabs_154;
                        float _fabs_155 = fabsf(absolute_4[11]);
                        absolute_4[11] = _fabs_155;
                        float _fabs_156 = fabsf(absolute_4[12]);
                        absolute_4[12] = _fabs_156;
                        float _fabs_157 = fabsf(absolute_4[13]);
                        absolute_4[13] = _fabs_157;
                        float _fabs_158 = fabsf(absolute_4[14]);
                        absolute_4[14] = _fabs_158;
                        float _fabs_159 = fabsf(absolute_4[15]);
                        absolute_4[15] = _fabs_159;
                        float absolute_4_max = absolute_4[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_4_max = max_noftz(absolute_4_max, absolute_4[_lr]);
                        }
                        float amax_7 = absolute_4_max;
                        float _rcp_20 = approx_rcp(6.0f);
                        float sf_value_8 = global_scale_1 * (amax_7 * _rcp_20);
                        float _fp8_rt_9;
                        uint16_t _e4m3x2_9;
                        uint32_t _f16x2_9;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_9) : "f"(0.0f), "f"(sf_value_8));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_9) : "h"(_e4m3x2_9));
                        uint16_t _fp8_h0_9 = (uint16_t)(_f16x2_9 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_9) : "h"(_fp8_h0_9));
                        float sf_rounded_9 = _fp8_rt_9;
                        float _rcp_21 = approx_rcp(sf_rounded_9 * global_scale_rcp_1);
                        float _min_11 = fminf(_rcp_21, 3.4028234663852886e+38f);
                        float output_scale_10 = _min_11;
                        float2 _f2_57 = make_float2(output_scale_10, output_scale_10);
                        #pragma unroll
                        for (int j_60 = 0; j_60 < 8; j_60++) {
                            float2 _f2_58 = make_float2(values_3[2 * j_60], values_3[2 * j_60 + 1]);
                            float2 _mul_f32x2_31;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_31) : "l"(*(const unsigned long long*)&_f2_58), "l"(*(const unsigned long long*)&_f2_57));
                            quant_values_5[2 * j_60] = _mul_f32x2_31.x;
                            quant_values_5[2 * j_60 + 1] = _mul_f32x2_31.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_6[0]) : "f"(quant_values_5[0]), "f"(quant_values_5[1]), "f"(quant_values_5[2]), "f"(quant_values_5[3]), "f"(quant_values_5[4]), "f"(quant_values_5[5]), "f"(quant_values_5[6]), "f"(quant_values_5[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_6[1]) : "f"(quant_values_5[8]), "f"(quant_values_5[9]), "f"(quant_values_5[10]), "f"(quant_values_5[11]), "f"(quant_values_5[12]), "f"(quant_values_5[13]), "f"(quant_values_5[14]), "f"(quant_values_5[15]));
                        row_out_2[2] = packed_6[0];
                        row_out_2[3] = packed_6[1];
                        sf_values_2[1] = sf_value_8;
                        float values_11[16];
                        float absolute_12[16];
                        float quant_values_13[16];
                        unsigned int packed_14[2];
                        #pragma unroll
                        for (int j_61 = 0; j_61 < 8; j_61++) {
                            values_11[2 * j_61] = __uint_as_float(normalized_1[16 + j_61] << 16);
                            values_11[2 * j_61 + 1] = __uint_as_float(normalized_1[16 + j_61] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_62 = 0; j_62 < 16; j_62++) {
                            absolute_12[j_62] = values_11[j_62];
                        }
                        float _fabs_160 = fabsf(absolute_12[0]);
                        absolute_12[0] = _fabs_160;
                        float _fabs_161 = fabsf(absolute_12[1]);
                        absolute_12[1] = _fabs_161;
                        float _fabs_162 = fabsf(absolute_12[2]);
                        absolute_12[2] = _fabs_162;
                        float _fabs_163 = fabsf(absolute_12[3]);
                        absolute_12[3] = _fabs_163;
                        float _fabs_164 = fabsf(absolute_12[4]);
                        absolute_12[4] = _fabs_164;
                        float _fabs_165 = fabsf(absolute_12[5]);
                        absolute_12[5] = _fabs_165;
                        float _fabs_166 = fabsf(absolute_12[6]);
                        absolute_12[6] = _fabs_166;
                        float _fabs_167 = fabsf(absolute_12[7]);
                        absolute_12[7] = _fabs_167;
                        float _fabs_168 = fabsf(absolute_12[8]);
                        absolute_12[8] = _fabs_168;
                        float _fabs_169 = fabsf(absolute_12[9]);
                        absolute_12[9] = _fabs_169;
                        float _fabs_170 = fabsf(absolute_12[10]);
                        absolute_12[10] = _fabs_170;
                        float _fabs_171 = fabsf(absolute_12[11]);
                        absolute_12[11] = _fabs_171;
                        float _fabs_172 = fabsf(absolute_12[12]);
                        absolute_12[12] = _fabs_172;
                        float _fabs_173 = fabsf(absolute_12[13]);
                        absolute_12[13] = _fabs_173;
                        float _fabs_174 = fabsf(absolute_12[14]);
                        absolute_12[14] = _fabs_174;
                        float _fabs_175 = fabsf(absolute_12[15]);
                        absolute_12[15] = _fabs_175;
                        float absolute_12_max = absolute_12[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_12_max = max_noftz(absolute_12_max, absolute_12[_lr]);
                        }
                        float amax_15 = absolute_12_max;
                        float _rcp_22 = approx_rcp(6.0f);
                        float sf_value_16 = global_scale_1 * (amax_15 * _rcp_22);
                        float _fp8_rt_10;
                        uint16_t _e4m3x2_10;
                        uint32_t _f16x2_10;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_10) : "f"(0.0f), "f"(sf_value_16));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_10) : "h"(_e4m3x2_10));
                        uint16_t _fp8_h0_10 = (uint16_t)(_f16x2_10 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_10) : "h"(_fp8_h0_10));
                        float sf_rounded_17 = _fp8_rt_10;
                        float _rcp_23 = approx_rcp(sf_rounded_17 * global_scale_rcp_1);
                        float _min_12 = fminf(_rcp_23, 3.4028234663852886e+38f);
                        float output_scale_18 = _min_12;
                        float2 _f2_59 = make_float2(output_scale_18, output_scale_18);
                        #pragma unroll
                        for (int j_63 = 0; j_63 < 8; j_63++) {
                            float2 _f2_60 = make_float2(values_11[2 * j_63], values_11[2 * j_63 + 1]);
                            float2 _mul_f32x2_32;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_32) : "l"(*(const unsigned long long*)&_f2_60), "l"(*(const unsigned long long*)&_f2_59));
                            quant_values_13[2 * j_63] = _mul_f32x2_32.x;
                            quant_values_13[2 * j_63 + 1] = _mul_f32x2_32.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_14[0]) : "f"(quant_values_13[0]), "f"(quant_values_13[1]), "f"(quant_values_13[2]), "f"(quant_values_13[3]), "f"(quant_values_13[4]), "f"(quant_values_13[5]), "f"(quant_values_13[6]), "f"(quant_values_13[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_14[1]) : "f"(quant_values_13[8]), "f"(quant_values_13[9]), "f"(quant_values_13[10]), "f"(quant_values_13[11]), "f"(quant_values_13[12]), "f"(quant_values_13[13]), "f"(quant_values_13[14]), "f"(quant_values_13[15]));
                        row_out_2[4] = packed_14[0];
                        row_out_2[5] = packed_14[1];
                        sf_values_2[2] = sf_value_16;
                        float values_19[16];
                        float absolute_20[16];
                        float quant_values_21[16];
                        unsigned int packed_22[2];
                        #pragma unroll
                        for (int j_64 = 0; j_64 < 8; j_64++) {
                            values_19[2 * j_64] = __uint_as_float(normalized_1[24 + j_64] << 16);
                            values_19[2 * j_64 + 1] = __uint_as_float(normalized_1[24 + j_64] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_65 = 0; j_65 < 16; j_65++) {
                            absolute_20[j_65] = values_19[j_65];
                        }
                        float _fabs_176 = fabsf(absolute_20[0]);
                        absolute_20[0] = _fabs_176;
                        float _fabs_177 = fabsf(absolute_20[1]);
                        absolute_20[1] = _fabs_177;
                        float _fabs_178 = fabsf(absolute_20[2]);
                        absolute_20[2] = _fabs_178;
                        float _fabs_179 = fabsf(absolute_20[3]);
                        absolute_20[3] = _fabs_179;
                        float _fabs_180 = fabsf(absolute_20[4]);
                        absolute_20[4] = _fabs_180;
                        float _fabs_181 = fabsf(absolute_20[5]);
                        absolute_20[5] = _fabs_181;
                        float _fabs_182 = fabsf(absolute_20[6]);
                        absolute_20[6] = _fabs_182;
                        float _fabs_183 = fabsf(absolute_20[7]);
                        absolute_20[7] = _fabs_183;
                        float _fabs_184 = fabsf(absolute_20[8]);
                        absolute_20[8] = _fabs_184;
                        float _fabs_185 = fabsf(absolute_20[9]);
                        absolute_20[9] = _fabs_185;
                        float _fabs_186 = fabsf(absolute_20[10]);
                        absolute_20[10] = _fabs_186;
                        float _fabs_187 = fabsf(absolute_20[11]);
                        absolute_20[11] = _fabs_187;
                        float _fabs_188 = fabsf(absolute_20[12]);
                        absolute_20[12] = _fabs_188;
                        float _fabs_189 = fabsf(absolute_20[13]);
                        absolute_20[13] = _fabs_189;
                        float _fabs_190 = fabsf(absolute_20[14]);
                        absolute_20[14] = _fabs_190;
                        float _fabs_191 = fabsf(absolute_20[15]);
                        absolute_20[15] = _fabs_191;
                        float absolute_20_max = absolute_20[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_20_max = max_noftz(absolute_20_max, absolute_20[_lr]);
                        }
                        float amax_23 = absolute_20_max;
                        float _rcp_24 = approx_rcp(6.0f);
                        float sf_value_24 = global_scale_1 * (amax_23 * _rcp_24);
                        float _fp8_rt_11;
                        uint16_t _e4m3x2_11;
                        uint32_t _f16x2_11;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_11) : "f"(0.0f), "f"(sf_value_24));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_11) : "h"(_e4m3x2_11));
                        uint16_t _fp8_h0_11 = (uint16_t)(_f16x2_11 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_11) : "h"(_fp8_h0_11));
                        float sf_rounded_25 = _fp8_rt_11;
                        float _rcp_25 = approx_rcp(sf_rounded_25 * global_scale_rcp_1);
                        float _min_13 = fminf(_rcp_25, 3.4028234663852886e+38f);
                        float output_scale_26 = _min_13;
                        float2 _f2_61 = make_float2(output_scale_26, output_scale_26);
                        #pragma unroll
                        for (int j_66 = 0; j_66 < 8; j_66++) {
                            float2 _f2_62 = make_float2(values_19[2 * j_66], values_19[2 * j_66 + 1]);
                            float2 _mul_f32x2_33;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_33) : "l"(*(const unsigned long long*)&_f2_62), "l"(*(const unsigned long long*)&_f2_61));
                            quant_values_21[2 * j_66] = _mul_f32x2_33.x;
                            quant_values_21[2 * j_66 + 1] = _mul_f32x2_33.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_22[0]) : "f"(quant_values_21[0]), "f"(quant_values_21[1]), "f"(quant_values_21[2]), "f"(quant_values_21[3]), "f"(quant_values_21[4]), "f"(quant_values_21[5]), "f"(quant_values_21[6]), "f"(quant_values_21[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_22[1]) : "f"(quant_values_21[8]), "f"(quant_values_21[9]), "f"(quant_values_21[10]), "f"(quant_values_21[11]), "f"(quant_values_21[12]), "f"(quant_values_21[13]), "f"(quant_values_21[14]), "f"(quant_values_21[15]));
                        row_out_2[6] = packed_22[0];
                        row_out_2[7] = packed_22[1];
                        sf_values_2[3] = sf_value_24;
                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar_1) : "memory");
                        float recv1 = exch[exch_slot_1];
                        exch[exch_slot_1] = sf_values_2[0];
                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar_1) : "memory");
                        float sf_group_values_2[4];
                        sf_group_values_2[0] = recv1;
                        sf_group_values_2[1] = sf_values_2[1];
                        sf_group_values_2[2] = sf_values_2[2];
                        sf_group_values_2[3] = sf_values_2[3];
                        unsigned int sf_packed_2[1];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(sf_group_values_2[0]), "f"(sf_group_values_2[1]),
                                                   "f"(sf_group_values_2[2]), "f"(sf_group_values_2[3]));
                            sf_packed_2[0] = _packed;
                        }
                        if (token_1 < M) {
                            *(reinterpret_cast<int*>(out_sf + (scale_row_base_1 + 512)) + (0)) = sf_packed_2[0];
                        }
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row_1 * 64 + 16 ^ (staging_row_1 * 64 + 16 >> 7 & 3) << 4))), "r"(row_out_2[0]), "r"(row_out_2[1]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row_1 * 64 + 40 ^ (staging_row_1 * 64 + 40 >> 7 & 3) << 4))), "r"(row_out_2[2]), "r"(row_out_2[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + (unsigned int)(staging_row_1 * 64 + 48 ^ (staging_row_1 * 64 + 48 >> 7 & 3) << 4))), "r"(row_out_2[4]), "r"(row_out_2[5]), "r"(row_out_2[6]), "r"(row_out_2[7]) : "memory");
                    } else {
                        unsigned int row_out_3[8];
                        float sf_values_3[4];
                        float values_4[16];
                        float absolute_3[16];
                        float quant_values_4[16];
                        unsigned int packed_4[2];
                        #pragma unroll
                        for (int j_67 = 0; j_67 < 8; j_67++) {
                            values_4[2 * j_67] = __uint_as_float(words_1[j_67] << 16);
                            values_4[2 * j_67 + 1] = __uint_as_float(words_1[j_67] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_68 = 0; j_68 < 16; j_68++) {
                            absolute_3[j_68] = values_4[j_68];
                        }
                        float _fabs_192 = fabsf(absolute_3[0]);
                        absolute_3[0] = _fabs_192;
                        float _fabs_193 = fabsf(absolute_3[1]);
                        absolute_3[1] = _fabs_193;
                        float _fabs_194 = fabsf(absolute_3[2]);
                        absolute_3[2] = _fabs_194;
                        float _fabs_195 = fabsf(absolute_3[3]);
                        absolute_3[3] = _fabs_195;
                        float _fabs_196 = fabsf(absolute_3[4]);
                        absolute_3[4] = _fabs_196;
                        float _fabs_197 = fabsf(absolute_3[5]);
                        absolute_3[5] = _fabs_197;
                        float _fabs_198 = fabsf(absolute_3[6]);
                        absolute_3[6] = _fabs_198;
                        float _fabs_199 = fabsf(absolute_3[7]);
                        absolute_3[7] = _fabs_199;
                        float _fabs_200 = fabsf(absolute_3[8]);
                        absolute_3[8] = _fabs_200;
                        float _fabs_201 = fabsf(absolute_3[9]);
                        absolute_3[9] = _fabs_201;
                        float _fabs_202 = fabsf(absolute_3[10]);
                        absolute_3[10] = _fabs_202;
                        float _fabs_203 = fabsf(absolute_3[11]);
                        absolute_3[11] = _fabs_203;
                        float _fabs_204 = fabsf(absolute_3[12]);
                        absolute_3[12] = _fabs_204;
                        float _fabs_205 = fabsf(absolute_3[13]);
                        absolute_3[13] = _fabs_205;
                        float _fabs_206 = fabsf(absolute_3[14]);
                        absolute_3[14] = _fabs_206;
                        float _fabs_207 = fabsf(absolute_3[15]);
                        absolute_3[15] = _fabs_207;
                        float absolute_max_3 = absolute_3[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_max_3 = max_noftz(absolute_max_3, absolute_3[_lr]);
                        }
                        float amax_3 = absolute_max_3;
                        float _rcp_26 = approx_rcp(6.0f);
                        float sf_value_3 = global_scale_1 * (amax_3 * _rcp_26);
                        float _fp8_rt_12;
                        uint16_t _e4m3x2_12;
                        uint32_t _f16x2_12;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(sf_value_3));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_12) : "h"(_e4m3x2_12));
                        uint16_t _fp8_h0_12 = (uint16_t)(_f16x2_12 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_12) : "h"(_fp8_h0_12));
                        float sf_rounded_3 = _fp8_rt_12;
                        float _rcp_27 = approx_rcp(sf_rounded_3 * global_scale_rcp_1);
                        float _min_14 = fminf(_rcp_27, 3.4028234663852886e+38f);
                        float output_scale_3 = _min_14;
                        float2 _f2_63 = make_float2(output_scale_3, output_scale_3);
                        #pragma unroll
                        for (int j_69 = 0; j_69 < 8; j_69++) {
                            float2 _f2_64 = make_float2(values_4[2 * j_69], values_4[2 * j_69 + 1]);
                            float2 _mul_f32x2_34;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_34) : "l"(*(const unsigned long long*)&_f2_64), "l"(*(const unsigned long long*)&_f2_63));
                            quant_values_4[2 * j_69] = _mul_f32x2_34.x;
                            quant_values_4[2 * j_69 + 1] = _mul_f32x2_34.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_4[0]) : "f"(quant_values_4[0]), "f"(quant_values_4[1]), "f"(quant_values_4[2]), "f"(quant_values_4[3]), "f"(quant_values_4[4]), "f"(quant_values_4[5]), "f"(quant_values_4[6]), "f"(quant_values_4[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_4[1]) : "f"(quant_values_4[8]), "f"(quant_values_4[9]), "f"(quant_values_4[10]), "f"(quant_values_4[11]), "f"(quant_values_4[12]), "f"(quant_values_4[13]), "f"(quant_values_4[14]), "f"(quant_values_4[15]));
                        row_out_3[0] = packed_4[0];
                        row_out_3[1] = packed_4[1];
                        sf_values_3[0] = sf_value_3;
                        float values_0_1[16];
                        float absolute_1_2[16];
                        float quant_values_2_1[16];
                        unsigned int packed_3_1[2];
                        #pragma unroll
                        for (int j_70 = 0; j_70 < 8; j_70++) {
                            values_0_1[2 * j_70] = __uint_as_float(words_1[8 + j_70] << 16);
                            values_0_1[2 * j_70 + 1] = __uint_as_float(words_1[8 + j_70] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_71 = 0; j_71 < 16; j_71++) {
                            absolute_1_2[j_71] = values_0_1[j_71];
                        }
                        float _fabs_208 = fabsf(absolute_1_2[0]);
                        absolute_1_2[0] = _fabs_208;
                        float _fabs_209 = fabsf(absolute_1_2[1]);
                        absolute_1_2[1] = _fabs_209;
                        float _fabs_210 = fabsf(absolute_1_2[2]);
                        absolute_1_2[2] = _fabs_210;
                        float _fabs_211 = fabsf(absolute_1_2[3]);
                        absolute_1_2[3] = _fabs_211;
                        float _fabs_212 = fabsf(absolute_1_2[4]);
                        absolute_1_2[4] = _fabs_212;
                        float _fabs_213 = fabsf(absolute_1_2[5]);
                        absolute_1_2[5] = _fabs_213;
                        float _fabs_214 = fabsf(absolute_1_2[6]);
                        absolute_1_2[6] = _fabs_214;
                        float _fabs_215 = fabsf(absolute_1_2[7]);
                        absolute_1_2[7] = _fabs_215;
                        float _fabs_216 = fabsf(absolute_1_2[8]);
                        absolute_1_2[8] = _fabs_216;
                        float _fabs_217 = fabsf(absolute_1_2[9]);
                        absolute_1_2[9] = _fabs_217;
                        float _fabs_218 = fabsf(absolute_1_2[10]);
                        absolute_1_2[10] = _fabs_218;
                        float _fabs_219 = fabsf(absolute_1_2[11]);
                        absolute_1_2[11] = _fabs_219;
                        float _fabs_220 = fabsf(absolute_1_2[12]);
                        absolute_1_2[12] = _fabs_220;
                        float _fabs_221 = fabsf(absolute_1_2[13]);
                        absolute_1_2[13] = _fabs_221;
                        float _fabs_222 = fabsf(absolute_1_2[14]);
                        absolute_1_2[14] = _fabs_222;
                        float _fabs_223 = fabsf(absolute_1_2[15]);
                        absolute_1_2[15] = _fabs_223;
                        float absolute_1_max_1 = absolute_1_2[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_1_max_1 = max_noftz(absolute_1_max_1, absolute_1_2[_lr]);
                        }
                        float amax_4_1 = absolute_1_max_1;
                        float _rcp_28 = approx_rcp(6.0f);
                        float sf_value_5_1 = global_scale_1 * (amax_4_1 * _rcp_28);
                        float _fp8_rt_13;
                        uint16_t _e4m3x2_13;
                        uint32_t _f16x2_13;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_13) : "f"(0.0f), "f"(sf_value_5_1));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_13) : "h"(_e4m3x2_13));
                        uint16_t _fp8_h0_13 = (uint16_t)(_f16x2_13 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_13) : "h"(_fp8_h0_13));
                        float sf_rounded_6_1 = _fp8_rt_13;
                        float _rcp_29 = approx_rcp(sf_rounded_6_1 * global_scale_rcp_1);
                        float _min_15 = fminf(_rcp_29, 3.4028234663852886e+38f);
                        float output_scale_7_1 = _min_15;
                        float2 _f2_65 = make_float2(output_scale_7_1, output_scale_7_1);
                        #pragma unroll
                        for (int j_72 = 0; j_72 < 8; j_72++) {
                            float2 _f2_66 = make_float2(values_0_1[2 * j_72], values_0_1[2 * j_72 + 1]);
                            float2 _mul_f32x2_35;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_35) : "l"(*(const unsigned long long*)&_f2_66), "l"(*(const unsigned long long*)&_f2_65));
                            quant_values_2_1[2 * j_72] = _mul_f32x2_35.x;
                            quant_values_2_1[2 * j_72 + 1] = _mul_f32x2_35.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_3_1[0]) : "f"(quant_values_2_1[0]), "f"(quant_values_2_1[1]), "f"(quant_values_2_1[2]), "f"(quant_values_2_1[3]), "f"(quant_values_2_1[4]), "f"(quant_values_2_1[5]), "f"(quant_values_2_1[6]), "f"(quant_values_2_1[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_3_1[1]) : "f"(quant_values_2_1[8]), "f"(quant_values_2_1[9]), "f"(quant_values_2_1[10]), "f"(quant_values_2_1[11]), "f"(quant_values_2_1[12]), "f"(quant_values_2_1[13]), "f"(quant_values_2_1[14]), "f"(quant_values_2_1[15]));
                        row_out_3[2] = packed_3_1[0];
                        row_out_3[3] = packed_3_1[1];
                        sf_values_3[1] = sf_value_5_1;
                        float values_8_1[16];
                        float absolute_9_1[16];
                        float quant_values_10_1[16];
                        unsigned int packed_11_1[2];
                        #pragma unroll
                        for (int j_73 = 0; j_73 < 8; j_73++) {
                            values_8_1[2 * j_73] = __uint_as_float(words_1[16 + j_73] << 16);
                            values_8_1[2 * j_73 + 1] = __uint_as_float(words_1[16 + j_73] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_74 = 0; j_74 < 16; j_74++) {
                            absolute_9_1[j_74] = values_8_1[j_74];
                        }
                        float _fabs_224 = fabsf(absolute_9_1[0]);
                        absolute_9_1[0] = _fabs_224;
                        float _fabs_225 = fabsf(absolute_9_1[1]);
                        absolute_9_1[1] = _fabs_225;
                        float _fabs_226 = fabsf(absolute_9_1[2]);
                        absolute_9_1[2] = _fabs_226;
                        float _fabs_227 = fabsf(absolute_9_1[3]);
                        absolute_9_1[3] = _fabs_227;
                        float _fabs_228 = fabsf(absolute_9_1[4]);
                        absolute_9_1[4] = _fabs_228;
                        float _fabs_229 = fabsf(absolute_9_1[5]);
                        absolute_9_1[5] = _fabs_229;
                        float _fabs_230 = fabsf(absolute_9_1[6]);
                        absolute_9_1[6] = _fabs_230;
                        float _fabs_231 = fabsf(absolute_9_1[7]);
                        absolute_9_1[7] = _fabs_231;
                        float _fabs_232 = fabsf(absolute_9_1[8]);
                        absolute_9_1[8] = _fabs_232;
                        float _fabs_233 = fabsf(absolute_9_1[9]);
                        absolute_9_1[9] = _fabs_233;
                        float _fabs_234 = fabsf(absolute_9_1[10]);
                        absolute_9_1[10] = _fabs_234;
                        float _fabs_235 = fabsf(absolute_9_1[11]);
                        absolute_9_1[11] = _fabs_235;
                        float _fabs_236 = fabsf(absolute_9_1[12]);
                        absolute_9_1[12] = _fabs_236;
                        float _fabs_237 = fabsf(absolute_9_1[13]);
                        absolute_9_1[13] = _fabs_237;
                        float _fabs_238 = fabsf(absolute_9_1[14]);
                        absolute_9_1[14] = _fabs_238;
                        float _fabs_239 = fabsf(absolute_9_1[15]);
                        absolute_9_1[15] = _fabs_239;
                        float absolute_9_max_1 = absolute_9_1[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_9_max_1 = max_noftz(absolute_9_max_1, absolute_9_1[_lr]);
                        }
                        float amax_12_1 = absolute_9_max_1;
                        float _rcp_30 = approx_rcp(6.0f);
                        float sf_value_13_1 = global_scale_1 * (amax_12_1 * _rcp_30);
                        float _fp8_rt_14;
                        uint16_t _e4m3x2_14;
                        uint32_t _f16x2_14;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_14) : "f"(0.0f), "f"(sf_value_13_1));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_14) : "h"(_e4m3x2_14));
                        uint16_t _fp8_h0_14 = (uint16_t)(_f16x2_14 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_14) : "h"(_fp8_h0_14));
                        float sf_rounded_14_1 = _fp8_rt_14;
                        float _rcp_31 = approx_rcp(sf_rounded_14_1 * global_scale_rcp_1);
                        float _min_16 = fminf(_rcp_31, 3.4028234663852886e+38f);
                        float output_scale_15_1 = _min_16;
                        float2 _f2_67 = make_float2(output_scale_15_1, output_scale_15_1);
                        #pragma unroll
                        for (int j_75 = 0; j_75 < 8; j_75++) {
                            float2 _f2_68 = make_float2(values_8_1[2 * j_75], values_8_1[2 * j_75 + 1]);
                            float2 _mul_f32x2_36;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_36) : "l"(*(const unsigned long long*)&_f2_68), "l"(*(const unsigned long long*)&_f2_67));
                            quant_values_10_1[2 * j_75] = _mul_f32x2_36.x;
                            quant_values_10_1[2 * j_75 + 1] = _mul_f32x2_36.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_11_1[0]) : "f"(quant_values_10_1[0]), "f"(quant_values_10_1[1]), "f"(quant_values_10_1[2]), "f"(quant_values_10_1[3]), "f"(quant_values_10_1[4]), "f"(quant_values_10_1[5]), "f"(quant_values_10_1[6]), "f"(quant_values_10_1[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_11_1[1]) : "f"(quant_values_10_1[8]), "f"(quant_values_10_1[9]), "f"(quant_values_10_1[10]), "f"(quant_values_10_1[11]), "f"(quant_values_10_1[12]), "f"(quant_values_10_1[13]), "f"(quant_values_10_1[14]), "f"(quant_values_10_1[15]));
                        row_out_3[4] = packed_11_1[0];
                        row_out_3[5] = packed_11_1[1];
                        sf_values_3[2] = sf_value_13_1;
                        float values_16_1[16];
                        float absolute_17_1[16];
                        float quant_values_18_1[16];
                        unsigned int packed_19_1[2];
                        #pragma unroll
                        for (int j_76 = 0; j_76 < 8; j_76++) {
                            values_16_1[2 * j_76] = __uint_as_float(words_1[24 + j_76] << 16);
                            values_16_1[2 * j_76 + 1] = __uint_as_float(words_1[24 + j_76] & 4294901760);
                        }
                        #pragma unroll
                        for (int j_77 = 0; j_77 < 16; j_77++) {
                            absolute_17_1[j_77] = values_16_1[j_77];
                        }
                        float _fabs_240 = fabsf(absolute_17_1[0]);
                        absolute_17_1[0] = _fabs_240;
                        float _fabs_241 = fabsf(absolute_17_1[1]);
                        absolute_17_1[1] = _fabs_241;
                        float _fabs_242 = fabsf(absolute_17_1[2]);
                        absolute_17_1[2] = _fabs_242;
                        float _fabs_243 = fabsf(absolute_17_1[3]);
                        absolute_17_1[3] = _fabs_243;
                        float _fabs_244 = fabsf(absolute_17_1[4]);
                        absolute_17_1[4] = _fabs_244;
                        float _fabs_245 = fabsf(absolute_17_1[5]);
                        absolute_17_1[5] = _fabs_245;
                        float _fabs_246 = fabsf(absolute_17_1[6]);
                        absolute_17_1[6] = _fabs_246;
                        float _fabs_247 = fabsf(absolute_17_1[7]);
                        absolute_17_1[7] = _fabs_247;
                        float _fabs_248 = fabsf(absolute_17_1[8]);
                        absolute_17_1[8] = _fabs_248;
                        float _fabs_249 = fabsf(absolute_17_1[9]);
                        absolute_17_1[9] = _fabs_249;
                        float _fabs_250 = fabsf(absolute_17_1[10]);
                        absolute_17_1[10] = _fabs_250;
                        float _fabs_251 = fabsf(absolute_17_1[11]);
                        absolute_17_1[11] = _fabs_251;
                        float _fabs_252 = fabsf(absolute_17_1[12]);
                        absolute_17_1[12] = _fabs_252;
                        float _fabs_253 = fabsf(absolute_17_1[13]);
                        absolute_17_1[13] = _fabs_253;
                        float _fabs_254 = fabsf(absolute_17_1[14]);
                        absolute_17_1[14] = _fabs_254;
                        float _fabs_255 = fabsf(absolute_17_1[15]);
                        absolute_17_1[15] = _fabs_255;
                        float absolute_17_max_1 = absolute_17_1[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 16; _lr++) {
                            absolute_17_max_1 = max_noftz(absolute_17_max_1, absolute_17_1[_lr]);
                        }
                        float amax_20_1 = absolute_17_max_1;
                        float _rcp_32 = approx_rcp(6.0f);
                        float sf_value_21_1 = global_scale_1 * (amax_20_1 * _rcp_32);
                        float _fp8_rt_15;
                        uint16_t _e4m3x2_15;
                        uint32_t _f16x2_15;
                        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_15) : "f"(0.0f), "f"(sf_value_21_1));
                        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_15) : "h"(_e4m3x2_15));
                        uint16_t _fp8_h0_15 = (uint16_t)(_f16x2_15 & 0xFFFFu);
                        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_15) : "h"(_fp8_h0_15));
                        float sf_rounded_22_1 = _fp8_rt_15;
                        float _rcp_33 = approx_rcp(sf_rounded_22_1 * global_scale_rcp_1);
                        float _min_17 = fminf(_rcp_33, 3.4028234663852886e+38f);
                        float output_scale_23_1 = _min_17;
                        float2 _f2_69 = make_float2(output_scale_23_1, output_scale_23_1);
                        #pragma unroll
                        for (int j_78 = 0; j_78 < 8; j_78++) {
                            float2 _f2_70 = make_float2(values_16_1[2 * j_78], values_16_1[2 * j_78 + 1]);
                            float2 _mul_f32x2_37;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_37) : "l"(*(const unsigned long long*)&_f2_70), "l"(*(const unsigned long long*)&_f2_69));
                            quant_values_18_1[2 * j_78] = _mul_f32x2_37.x;
                            quant_values_18_1[2 * j_78 + 1] = _mul_f32x2_37.y;
                        }
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_19_1[0]) : "f"(quant_values_18_1[0]), "f"(quant_values_18_1[1]), "f"(quant_values_18_1[2]), "f"(quant_values_18_1[3]), "f"(quant_values_18_1[4]), "f"(quant_values_18_1[5]), "f"(quant_values_18_1[6]), "f"(quant_values_18_1[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_19_1[1]) : "f"(quant_values_18_1[8]), "f"(quant_values_18_1[9]), "f"(quant_values_18_1[10]), "f"(quant_values_18_1[11]), "f"(quant_values_18_1[12]), "f"(quant_values_18_1[13]), "f"(quant_values_18_1[14]), "f"(quant_values_18_1[15]));
                        row_out_3[6] = packed_19_1[0];
                        row_out_3[7] = packed_19_1[1];
                        sf_values_3[3] = sf_value_21_1;
                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar_1) : "memory");
                        float recv1_1 = exch[exch_slot_1];
                        exch[exch_slot_1] = sf_values_3[0];
                        asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar_1) : "memory");
                        float sf_group_values_3[4];
                        sf_group_values_3[0] = recv1_1;
                        sf_group_values_3[1] = sf_values_3[1];
                        sf_group_values_3[2] = sf_values_3[2];
                        sf_group_values_3[3] = sf_values_3[3];
                        unsigned int sf_packed_3[1];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(sf_group_values_3[0]), "f"(sf_group_values_3[1]),
                                                   "f"(sf_group_values_3[2]), "f"(sf_group_values_3[3]));
                            sf_packed_3[0] = _packed;
                        }
                        if (token_1 < M) {
                            *(reinterpret_cast<int*>(out_sf + (scale_row_base_1 + 512)) + (0)) = sf_packed_3[0];
                        }
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row_1 * 64 + 16 ^ (staging_row_1 * 64 + 16 >> 7 & 3) << 4))), "r"(row_out_3[0]), "r"(row_out_3[1]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row_1 * 64 + 40 ^ (staging_row_1 * 64 + 40 >> 7 & 3) << 4))), "r"(row_out_3[2]), "r"(row_out_3[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + (unsigned int)(staging_row_1 * 64 + 48 ^ (staging_row_1 * 64 + 48 >> 7 & 3) << 4))), "r"(row_out_3[4]), "r"(row_out_3[5]), "r"(row_out_3[6]), "r"(row_out_3[7]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync %0, 256;" :: "r"(store_bar_1) : "memory");
                    tile_parity_1 = tile_parity_1 ^ 1;
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_3 * 8), "r"(0) : "memory");
                work_stage_3 += 1;
                if (work_stage_3 == 4) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                this_bid_2 = _clc_ctaid_3 + (unsigned int)cta_rank;
            }
            {
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
