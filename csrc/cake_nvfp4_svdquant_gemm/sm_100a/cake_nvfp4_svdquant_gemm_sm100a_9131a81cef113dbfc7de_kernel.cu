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
#define TMEM_NCOLS 80
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 64
#define TMEM_TMEM_SFB_OFFSET 72
#define NUM_TMA_PIPE_STAGES 14
#define NUM_ACC_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 8192
#define SMEM_SMEM_A_STRIDE 14336
#define SMEM_SMEM_B_OFF 9216
#define SMEM_SMEM_B_STAGE_BYTES 4096
#define SMEM_SMEM_B_STRIDE 14336
#define SMEM_SMEM_SFA_OFF 13312
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 14336
#define SMEM_SMEM_SFA_LO_OFF 13312
#define SMEM_SMEM_SFA_LO_STAGE_BYTES 512
#define SMEM_SMEM_SFA_LO_STRIDE 14336
#define SMEM_SMEM_SFA_HI_OFF 13824
#define SMEM_SMEM_SFA_HI_STAGE_BYTES 512
#define SMEM_SMEM_SFA_HI_STRIDE 14336
#define SMEM_SMEM_SFB_OFF 14336
#define SMEM_SMEM_SFB_STAGE_BYTES 1024
#define SMEM_SMEM_SFB_STRIDE 14336
#define SMEM_SMEM_D_OFF 1024
#define SMEM_SMEM_D_STAGE_BYTES 8192
#define SMEM_SMEM_D_STRIDE 14336
#define SMEM_SMEM_L1_OFF 9216
#define SMEM_SMEM_L1_STAGE_BYTES 4096
#define SMEM_SMEM_L1_STRIDE 14336
#define SMEM_SMEM_OUT_OFF 201728
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_TOTAL 218112
#define THREADS 256

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


__device__ __forceinline__ void elect_commit_cg1_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
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


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo256(int addr) {
    const int SBO = 256;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo256(int lo) {
    const int SBO = 256;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ void tcgen05_commit_cg1_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256, 1) __cluster_dims__(1,2,1) void
kernel_cake_nvfp4_svdquant_gemm_sm100a_9131a81cef113dbfc7de(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap D, const __grid_constant__ CUtensorMap L1, float* __restrict__ alpha, const __grid_constant__ CUtensorMap out, int M, int N, int K_tiles, int rank_tiles, int grid_n)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define tma_empty_addr (mbar_base + 112)
    #define acc_full_addr (mbar_base + 224)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = ((blockIdx.z * (gridDim.y / 2) + (blockIdx.y / 2)) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * (gridDim.y / 2) * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 232);

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_b_addr = smem + 9216;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 13312);
    const int smem_sfa_addr = smem + 13312;
    uint8_t* smem_sfa_lo = reinterpret_cast<uint8_t*>(smem_raw + 13312);
    const int smem_sfa_lo_addr = smem + 13312;
    uint8_t* smem_sfa_hi = reinterpret_cast<uint8_t*>(smem_raw + 13824);
    const int smem_sfa_hi_addr = smem + 13824;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 14336);
    const int smem_sfb_addr = smem + 14336;
    __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_d_addr = smem + 1024;
    __nv_bfloat16* smem_l1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_l1_addr = smem + 9216;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 201728);
    const int smem_out_addr = smem + 201728;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 29 barriers)
    // Mbarriers at smem_raw[0..232)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 14 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // tma_empty: 14 barriers, init_count=2
            mbarrier_init(smem + 112, 2);
            mbarrier_init(smem + 120, 2);
            mbarrier_init(smem + 128, 2);
            mbarrier_init(smem + 136, 2);
            mbarrier_init(smem + 144, 2);
            mbarrier_init(smem + 152, 2);
            mbarrier_init(smem + 160, 2);
            mbarrier_init(smem + 168, 2);
            mbarrier_init(smem + 176, 2);
            mbarrier_init(smem + 184, 2);
            mbarrier_init(smem + 192, 2);
            mbarrier_init(smem + 200, 2);
            mbarrier_init(smem + 208, 2);
            mbarrier_init(smem + 216, 2);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 80 used)
    if (warp == 0) {
        int _tmem_hold = smem + 232;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 64;
    const int tmem_tmem_sfb = taddr + 72;

    // ---- Role: mma ----
    if (warp == 0) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int mma_phase = 0;
            uint32_t _mbar_token_3 = mbarrier_try_wait(tma_full_addr + (mma_stage) * 8, mma_phase);
            unsigned int peek_full = _mbar_token_3;
            int mma_bid_m = blockIdx.x;
            int mma_bid_n = blockIdx.y;
            int mma_sfb_half = mma_bid_n % 2;
            #pragma unroll 1
            for (unsigned int k_tile_mma = 0; k_tile_mma < K_tiles; k_tile_mma++) {
                mbarrier_wait_token(tma_full_addr + (mma_stage) * 8, mma_phase, peek_full);
                unsigned int read_stage = mma_stage;
                mma_stage += 1;
                if (mma_stage == 14) { mma_stage = 0; mma_phase ^= 1; }
                peek_full = 1;
                if (k_tile_mma + 1 < (unsigned int)K_tiles) {
                    uint32_t _mbar_token_4 = mbarrier_try_wait(tma_full_addr + (mma_stage) * 8, mma_phase);
                    peek_full = _mbar_token_4;
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo256((((smem_sfa_addr) >> 4) + (read_stage) * 896)));
                    tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo256((((smem_sfa_addr) >> 4) + (read_stage) * 896 + 8)));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo256((((smem_sfb_addr) >> 4) + (read_stage) * 896)));
                    tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo256((((smem_sfb_addr) >> 4) + (read_stage) * 896 + 8)));
                }
                int init_flag = ((k_tile_mma == 0) ? 1 : 0);
                int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (read_stage) * 896);
                int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (read_stage) * 896);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x80004020 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x80004020 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8100480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + mma_sfb_half * 2 + 0, ((init_flag) ? 0 : 1));
                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 2, b_desc + 2,
                            0x8100480U, tmem_tmem_sfa + 4, tmem_tmem_sfb + mma_sfb_half * 2 + 4, 1);
                    }
                }
                elect_commit_cg1_multicast(tma_empty_addr + (read_stage) * 8, (uint16_t)(3));
            }
            #pragma unroll 1
            for (unsigned int rank_tile_mma = 0; rank_tile_mma < rank_tiles; rank_tile_mma++) {
                uint32_t _mbar_token_5 = mbarrier_try_wait(tma_full_addr + (mma_stage) * 8, mma_phase);
                unsigned int rank_token = _mbar_token_5;
                mbarrier_wait_token(tma_full_addr + (mma_stage) * 8, mma_phase, rank_token);
                unsigned int read_stage_1 = mma_stage;
                mma_stage += 1;
                if (mma_stage == 14) { mma_stage = 0; mma_phase ^= 1; }
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_1 = make_warp_uniform((((smem_d_addr) >> 4) & 0x3FFF) + (read_stage_1) * 896);
                int _mma_b_lo_1 = make_warp_uniform((((smem_l1_addr) >> 4) & 0x3FFF) + (read_stage_1) * 896);
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_accum), "r"(1));
                elect_commit_cg1_multicast(tma_empty_addr + (read_stage_1) * 8, (uint16_t)(3));
            }
            elect_commit(acc_full_addr);
        }
    }
    // ---- Role: scheduler ----
    if (warp == 1) {
        // idle — no tasks assigned
    }
    // ---- Role: load ----
    if (warp == 2) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int load_phase = 1;
            uint32_t _mbar_token_0 = mbarrier_try_wait(tma_empty_addr + (load_stage) * 8, load_phase);
            unsigned int peek_empty = _mbar_token_0;
            int bid_m = blockIdx.x;
            int bid_n = blockIdx.y;
            int off_m = bid_m * 128;
            int off_n = bid_n * 64;
            int sfb_atom_n = bid_n / 2;
            int sfb_half = bid_n % 2;
            #pragma unroll 1
            for (unsigned int k_tile = 0; k_tile < K_tiles; k_tile++) {
                mbarrier_wait_token(tma_empty_addr + (load_stage) * 8, load_phase, peek_empty);
                unsigned int write_stage = load_stage;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(tma_full_addr + (write_stage) * 8, 14336);
                }
                load_stage += 1;
                if (load_stage == 14) { load_stage = 0; load_phase ^= 1; }
                uint32_t _mbar_token_1 = mbarrier_try_wait(tma_empty_addr + (load_stage) * 8, load_phase);
                peek_empty = _mbar_token_1;
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                        :: "r"(smem_a_addr + write_stage * 14336 + (unsigned int)(cta_rank * 4096)), "l"((&A)), "r"(0), "r"(off_m + cta_rank * 64), "r"(k_tile),
                           "r"(tma_full_addr + (write_stage) * 8), "h"((uint16_t)(3)) : "memory");
                    tma_3d_gmem2smem(smem_b_addr + write_stage * 14336, (&B), 0, off_n, k_tile, tma_full_addr + (write_stage) * 8);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(smem_sfa_lo_addr + write_stage * 14336 + (unsigned int)(cta_rank * 512)), "l"((&SFA)), "r"(0), "r"(2 * k_tile), "r"(cta_rank * 2), "r"(bid_m),
                           "r"(tma_full_addr + (write_stage) * 8), "h"((uint16_t)(3)) : "memory");
                    tma_4d_gmem2smem(smem_sfb_addr + write_stage * 14336, (&SFB), 0, 2 * k_tile, 0, sfb_atom_n, tma_full_addr + (write_stage) * 8);
                }
            }
            #pragma unroll 1
            for (unsigned int rank_tile = 0; rank_tile < rank_tiles; rank_tile++) {
                int rank_offset = rank_tile * 32;
                mbarrier_wait_token(tma_empty_addr + (load_stage) * 8, load_phase, peek_empty);
                unsigned int write_stage_1 = load_stage;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(tma_full_addr + (write_stage_1) * 8, 14336);
                }
                load_stage += 1;
                if (load_stage == 14) { load_stage = 0; load_phase ^= 1; }
                peek_empty = 1;
                if (rank_tile + 1 < (unsigned int)rank_tiles) {
                    uint32_t _mbar_token_2 = mbarrier_try_wait(tma_empty_addr + (load_stage) * 8, load_phase);
                    peek_empty = _mbar_token_2;
                }
                if (elect_sync()) {
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(smem_d_addr + write_stage_1 * 14336 + (unsigned int)(cta_rank * 4096)), "l"((&D)), "r"(rank_offset), "r"(off_m + cta_rank * 64),
                           "r"(tma_full_addr + (write_stage_1) * 8), "h"((uint16_t)(3)) : "memory");
                    tma_2d_gmem2smem(smem_l1_addr + write_stage_1 * 14336, (&L1), rank_offset, off_n, tma_full_addr + (write_stage_1) * 8);
                    if (cta_rank == 0) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_sfa_lo_addr + write_stage_1 * 14336), "l"((&SFA)), "r"(0), "r"(0), "r"(0), "r"(bid_m),
                               "r"(tma_full_addr + (write_stage_1) * 8), "h"((uint16_t)(3)) : "memory");
                    } else {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_sfa_hi_addr + write_stage_1 * 14336), "l"((&SFA)), "r"(0), "r"(0), "r"(2), "r"(bid_m),
                               "r"(tma_full_addr + (write_stage_1) * 8), "h"((uint16_t)(3)) : "memory");
                    }
                    tma_4d_gmem2smem(smem_sfb_addr + write_stage_1 * 14336, (&SFB), 0, 0, 0, sfb_atom_n, tma_full_addr + (write_stage_1) * 8);
                }
            }
            #pragma unroll
            for (int _tail = 0; _tail < 14; _tail++) {
                mbarrier_wait(tma_empty_addr + (load_stage) * 8, load_phase);
                load_stage += 1;
                if (load_stage == 14) { load_stage = 0; load_phase ^= 1; }
            }
        }
    }
    // ---- Role: epilogue_load ----
    if (warp == 3) {
        { // epilogue_load_main
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&D))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&L1))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory");
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp - 4;
            int epi_bid_m = blockIdx.x;
            int epi_bid_n = blockIdx.y;
            int off_m_1 = epi_bid_m * 128;
            int off_n_1 = epi_bid_n * 64;
            unsigned int store_stage = 0;
            uint32_t _mbar_token_6 = mbarrier_try_wait(acc_full_addr, 0);
            unsigned int acc_wait_token = _mbar_token_6;
            float alpha_value = alpha[0];
            mbarrier_wait_token(acc_full_addr, 0, acc_wait_token);
            asm volatile("tcgen05.fence::after_thread_sync;");
            #pragma unroll
            for (int subtile = 0; subtile < 2; subtile++) {
                int tmem_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(subtile * 32);
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(tmem_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float result[32];
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    result[j] = _tmem_load_0[j] * alpha_value;
                }
                uint32_t result_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(result[_lp*2 + 0], result[_lp*2+1 + 0]));
                    result_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (subtile > 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            tma_store_2d((&out), off_n_1 + (subtile - 1) * 32, off_m_1, smem_out_addr + store_stage * 8192);
                        }
                    }
                    if (warp == 4) {
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    store_stage = store_stage ^ 1;
                }
                int out_stage_row = store_stage * 128 + (unsigned int)(epi_warp * 32) + lane;
                unsigned int out_abs = smem_out_addr + (unsigned int)(out_stage_row * 64);
                unsigned int out_swz = out_abs / 8 & 48;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (0 ^ out_swz)))), "r"(result_bf16[0]), "r"(result_bf16[1]), "r"(result_bf16[2]), "r"(result_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (16 ^ out_swz)))), "r"(result_bf16[4]), "r"(result_bf16[5]), "r"(result_bf16[6]), "r"(result_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (32 ^ out_swz)))), "r"(result_bf16[8]), "r"(result_bf16[9]), "r"(result_bf16[10]), "r"(result_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (48 ^ out_swz)))), "r"(result_bf16[12]), "r"(result_bf16[13]), "r"(result_bf16[14]), "r"(result_bf16[15]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 4) {
                if (elect_sync()) {
                    tma_store_2d((&out), off_n_1 + 32, off_m_1, smem_out_addr + store_stage * 8192);
                }
            }
            if (warp == 4) {
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
