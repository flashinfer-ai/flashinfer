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
#define TMEM_NCOLS 512
#define TMEM_PARTIALS_OFFSET 0
#define NUM_AB_PIPE_STAGES 5
#define NUM_PARTIAL_PIPE_STAGES 2
#define NUM_SCALE_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 81920
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_EPI_STAGING_OFF 163840
#define SMEM_EPI_STAGING_STAGE_BYTES 65536
#define SMEM_EPI_STAGING_STRIDE 65536
#define SMEM_SMEM_ASCALE_OFF 229376
#define SMEM_SMEM_ASCALE_STAGE_BYTES 2048
#define SMEM_SMEM_ASCALE_STRIDE 2048
#define SMEM_SMEM_BSCALE_OFF 231424
#define SMEM_SMEM_BSCALE_STAGE_BYTES 32
#define SMEM_SMEM_BSCALE_STRIDE 32
#define SMEM_TOTAL 231680
#define THREADS 384

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


__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
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


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
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

extern "C" {

__global__ __launch_bounds__(384, 1) __cluster_dims__(2,1,1) void
kernel_cake_grouped_fp8_gemm_d33456ef73586060263f(CakeTensorMap const* A, CakeTensorMap const* B, CakeTensorMap const* C_tma, __nv_bfloat16* __restrict__ C, float* __restrict__ a_scale, float* __restrict__ b_scale, int* __restrict__ m_indices, int M, int N, int K, int G)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem + 231456;
    #define ab_full_addr (mbar_base + 0)
    #define ab_free_addr (mbar_base + 40)
    #define partial_full_addr (mbar_base + 80)
    #define partial_free_addr (mbar_base + 96)
    #define scale_full_addr (mbar_base + 112)
    #define scale_free_addr (mbar_base + 120)
    #define producers_done_addr (mbar_base + 128)
    #define pair_exit_addr (mbar_base + 136)

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
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(C_tma)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 81920);
    const int smem_b_addr = smem + 81920;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 163840);
    const int epi_staging_addr = smem + 163840;
    float* smem_ascale = reinterpret_cast<float*>(smem_raw + 229376);
    const int smem_ascale_addr = smem + 229376;
    float* smem_bscale = reinterpret_cast<float*>(smem_raw + 231424);
    const int smem_bscale_addr = smem + 231424;
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 231600);
    int taddr;
    int tmem_partials;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 18 barriers)
    // Mbarriers at smem_raw[231456..231600)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 5 barriers, init_count=2
            mbarrier_init(smem + 231456, 2);
            mbarrier_init(smem + 231464, 2);
            mbarrier_init(smem + 231472, 2);
            mbarrier_init(smem + 231480, 2);
            mbarrier_init(smem + 231488, 2);
            // ab_free: 5 barriers, init_count=1
            mbarrier_init(smem + 231496, 1);
            mbarrier_init(smem + 231504, 1);
            mbarrier_init(smem + 231512, 1);
            mbarrier_init(smem + 231520, 1);
            mbarrier_init(smem + 231528, 1);
            // --- pipeline 'partial_pipe' ---
            // partial_full: 2 barriers, init_count=1
            mbarrier_init(smem + 231536, 1);
            mbarrier_init(smem + 231544, 1);
            // partial_free: 2 barriers, init_count=16
            mbarrier_init(smem + 231552, 16);
            mbarrier_init(smem + 231560, 16);
            // --- pipeline 'scale_pipe' ---
            // scale_full: 1 barriers, init_count=64
            mbarrier_init(smem + 231568, 64);
            // scale_free: 1 barriers, init_count=256
            mbarrier_init(smem + 231576, 256);
            // producers_done: 1 barriers, init_count=3
            mbarrier_init(smem + 231584, 3);
            // pair_exit: 1 barriers, init_count=1
            mbarrier_init(smem + 231592, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: acc_epi ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // acc_epi_main
            if (warp == 0) {
                int _tmem_hold_0 = smem + 231600;
                asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold_0), "r"(512) : "memory");
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            taddr = tmem_addr_storage[0];
            tmem_partials = taddr;
            if (warp == 0) {
                asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
            }
            unsigned int partial_stage = 0;
            unsigned int partial_full_phase = 0;
            unsigned int epi_panel_cursor = 0;
            unsigned int scale_stage = 0;
            const int acc_wg = warp / 4;
            const int acc_warp_in_wg = warp % 4;
            int wg_col_base = acc_wg * 128;
            int row_base = acc_warp_in_wg * 32;
            int row = row_base + lane;
            int m_tiles = (M - 1) / 256 + 1;
            int n_tiles = N / 256;
            int total_tiles = m_tiles * n_tiles;
            int cluster_m_tile = cluster_id / (unsigned int)n_tiles;
            int n_tile = cluster_id % (unsigned int)n_tiles;
            int step_m = num_clusters / (unsigned int)n_tiles;
            int step_n = num_clusters % (unsigned int)n_tiles;
            unsigned int _phase_scale_full = 0;
            #pragma unroll 1
            for (int tile_id = cluster_id; tile_id < total_tiles; tile_id += num_clusters) {
                int m_tile = cluster_m_tile * 2 + cta_rank;
                int cluster_m_base = cluster_m_tile * 256;
                int tile_rows = 256;
                if (tile_rows > M - cluster_m_base) {
                    tile_rows = M - cluster_m_base;
                }
                int last_group = 0;
                if (lane == 0) {
                    last_group = m_indices[cluster_m_base + tile_rows - 1];
                }
                int run_begin = 0;
                #pragma unroll 1
                for (int segment_iter = 0; segment_iter < tile_rows; segment_iter++) {
                    if (run_begin >= tile_rows) {
                        break;
                    }
                    int run_end = tile_rows;
                    int group = 0;
                    if (lane == 0) {
                        group = m_indices[cluster_m_base + run_begin];
                        if (group != last_group) {
                            #pragma unroll 1
                            for (int probe = run_begin + 1; probe < tile_rows; probe++) {
                                int next_group = m_indices[cluster_m_base + probe];
                                if (next_group != group) {
                                    run_end = probe;
                                    break;
                                }
                            }
                        }
                    }
                    int _shfl_6 = __shfl_sync(0xFFFFFFFF, run_end, 0);
                    run_end = _shfl_6;
                    int _shfl_7 = __shfl_sync(0xFFFFFFFF, group, 0);
                    group = _shfl_7;
                    float acc0[32];
                    float acc1[32];
                    float acc2[32];
                    float acc3[32];
                    acc0[0] = 0.0f;
                    acc0[1] = 0.0f;
                    acc0[2] = 0.0f;
                    acc0[3] = 0.0f;
                    acc0[4] = 0.0f;
                    acc0[5] = 0.0f;
                    acc0[6] = 0.0f;
                    acc0[7] = 0.0f;
                    acc0[8] = 0.0f;
                    acc0[9] = 0.0f;
                    acc0[10] = 0.0f;
                    acc0[11] = 0.0f;
                    acc0[12] = 0.0f;
                    acc0[13] = 0.0f;
                    acc0[14] = 0.0f;
                    acc0[15] = 0.0f;
                    acc0[16] = 0.0f;
                    acc0[17] = 0.0f;
                    acc0[18] = 0.0f;
                    acc0[19] = 0.0f;
                    acc0[20] = 0.0f;
                    acc0[21] = 0.0f;
                    acc0[22] = 0.0f;
                    acc0[23] = 0.0f;
                    acc0[24] = 0.0f;
                    acc0[25] = 0.0f;
                    acc0[26] = 0.0f;
                    acc0[27] = 0.0f;
                    acc0[28] = 0.0f;
                    acc0[29] = 0.0f;
                    acc0[30] = 0.0f;
                    acc0[31] = 0.0f;
                    acc1[0] = 0.0f;
                    acc1[1] = 0.0f;
                    acc1[2] = 0.0f;
                    acc1[3] = 0.0f;
                    acc1[4] = 0.0f;
                    acc1[5] = 0.0f;
                    acc1[6] = 0.0f;
                    acc1[7] = 0.0f;
                    acc1[8] = 0.0f;
                    acc1[9] = 0.0f;
                    acc1[10] = 0.0f;
                    acc1[11] = 0.0f;
                    acc1[12] = 0.0f;
                    acc1[13] = 0.0f;
                    acc1[14] = 0.0f;
                    acc1[15] = 0.0f;
                    acc1[16] = 0.0f;
                    acc1[17] = 0.0f;
                    acc1[18] = 0.0f;
                    acc1[19] = 0.0f;
                    acc1[20] = 0.0f;
                    acc1[21] = 0.0f;
                    acc1[22] = 0.0f;
                    acc1[23] = 0.0f;
                    acc1[24] = 0.0f;
                    acc1[25] = 0.0f;
                    acc1[26] = 0.0f;
                    acc1[27] = 0.0f;
                    acc1[28] = 0.0f;
                    acc1[29] = 0.0f;
                    acc1[30] = 0.0f;
                    acc1[31] = 0.0f;
                    acc2[0] = 0.0f;
                    acc2[1] = 0.0f;
                    acc2[2] = 0.0f;
                    acc2[3] = 0.0f;
                    acc2[4] = 0.0f;
                    acc2[5] = 0.0f;
                    acc2[6] = 0.0f;
                    acc2[7] = 0.0f;
                    acc2[8] = 0.0f;
                    acc2[9] = 0.0f;
                    acc2[10] = 0.0f;
                    acc2[11] = 0.0f;
                    acc2[12] = 0.0f;
                    acc2[13] = 0.0f;
                    acc2[14] = 0.0f;
                    acc2[15] = 0.0f;
                    acc2[16] = 0.0f;
                    acc2[17] = 0.0f;
                    acc2[18] = 0.0f;
                    acc2[19] = 0.0f;
                    acc2[20] = 0.0f;
                    acc2[21] = 0.0f;
                    acc2[22] = 0.0f;
                    acc2[23] = 0.0f;
                    acc2[24] = 0.0f;
                    acc2[25] = 0.0f;
                    acc2[26] = 0.0f;
                    acc2[27] = 0.0f;
                    acc2[28] = 0.0f;
                    acc2[29] = 0.0f;
                    acc2[30] = 0.0f;
                    acc2[31] = 0.0f;
                    acc3[0] = 0.0f;
                    acc3[1] = 0.0f;
                    acc3[2] = 0.0f;
                    acc3[3] = 0.0f;
                    acc3[4] = 0.0f;
                    acc3[5] = 0.0f;
                    acc3[6] = 0.0f;
                    acc3[7] = 0.0f;
                    acc3[8] = 0.0f;
                    acc3[9] = 0.0f;
                    acc3[10] = 0.0f;
                    acc3[11] = 0.0f;
                    acc3[12] = 0.0f;
                    acc3[13] = 0.0f;
                    acc3[14] = 0.0f;
                    acc3[15] = 0.0f;
                    acc3[16] = 0.0f;
                    acc3[17] = 0.0f;
                    acc3[18] = 0.0f;
                    acc3[19] = 0.0f;
                    acc3[20] = 0.0f;
                    acc3[21] = 0.0f;
                    acc3[22] = 0.0f;
                    acc3[23] = 0.0f;
                    acc3[24] = 0.0f;
                    acc3[25] = 0.0f;
                    acc3[26] = 0.0f;
                    acc3[27] = 0.0f;
                    acc3[28] = 0.0f;
                    acc3[29] = 0.0f;
                    acc3[30] = 0.0f;
                    acc3[31] = 0.0f;
                    float partial_fragment[16];
                    int k_blocks = 8;
                    int scale_groups = k_blocks / 4;
                    #pragma unroll 2
                    for (int scale_group = 0; scale_group < scale_groups; scale_group++) {
                        mbarrier_wait(scale_full_addr + (scale_stage) * 8, _phase_scale_full);
                        int scale_base = scale_stage * 128 * 4 + (unsigned int)(row * 2);
                        float a_values[2];
                        uint32_t _mbar_token_2 = mbarrier_test_wait(partial_full_addr + (partial_stage) * 8, partial_full_phase);
                        unsigned int partial_full_token = _mbar_token_2;
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&a_values[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_values[(0) + 1]))
                            : "r"(smem_ascale_addr + (unsigned int)(scale_base * 4)));
                        float b_s = smem_bscale[(scale_stage * 2 + (unsigned int)acc_wg) * 4];
                        #pragma unroll
                        for (int k_inner = 0; k_inner < 4; k_inner++) {
                            if (k_inner < 2) {
                                mbarrier_wait_token(partial_full_addr + (partial_stage) * 8, partial_full_phase, partial_full_token);
                            } else {
                                mbarrier_wait(partial_full_addr + (partial_stage) * 8, partial_full_phase);
                            }
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (k_inner == 2) {
                                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&a_values[0])), "=r"(*reinterpret_cast<uint32_t*>(&a_values[(0) + 1]))
                                    : "r"(smem_ascale_addr + (unsigned int)((scale_base + k_inner / 2 * 128 * 2) * 4)));
                            }
                            float a_s = a_values[k_inner % 2];
                            if (k_inner != 0) {
                                b_s = smem_bscale[(scale_stage * 2 + (unsigned int)acc_wg) * 4 + (unsigned int)k_inner];
                            }
                            float combined = a_s * b_s;
                            if (k_inner == 3) {
                                mbarrier_arrive(scale_free_addr + (scale_stage) * 8);
                            }
                            int _trl_addr_1 = tmem_partials + (partial_stage * 256 + (unsigned int)wg_col_base) + (row_base << 16);
                            tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_1);
                            {
                                unsigned long long _fma_acc_scale2_2;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_2) : "f"(combined));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[0]), "+f"(acc0[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_2));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[2]), "+f"(acc0[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_2));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[4]), "+f"(acc0[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_2));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[6]), "+f"(acc0[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_2));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[8]), "+f"(acc0[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_2));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[10]), "+f"(acc0[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_2));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[12]), "+f"(acc0[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_2));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[14]), "+f"(acc0[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_2));
                            }
                            int _trl_addr_3 = tmem_partials + (partial_stage * 256 + (unsigned int)wg_col_base + 16) + (row_base << 16);
                            tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_3);
                            {
                                unsigned long long _fma_acc_scale2_4;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_4) : "f"(combined));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[16]), "+f"(acc0[17]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_4));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[18]), "+f"(acc0[19]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_4));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[20]), "+f"(acc0[21]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_4));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[22]), "+f"(acc0[23]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_4));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[24]), "+f"(acc0[25]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_4));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[26]), "+f"(acc0[27]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_4));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[28]), "+f"(acc0[29]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_4));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[30]), "+f"(acc0[31]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_4));
                            }
                            int _trl_addr_5 = tmem_partials + (partial_stage * 256 + (unsigned int)wg_col_base + 32) + (row_base << 16);
                            tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_5);
                            {
                                unsigned long long _fma_acc_scale2_6;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_6) : "f"(combined));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[0]), "+f"(acc1[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_6));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[2]), "+f"(acc1[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_6));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[4]), "+f"(acc1[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_6));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[6]), "+f"(acc1[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_6));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[8]), "+f"(acc1[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_6));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[10]), "+f"(acc1[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_6));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[12]), "+f"(acc1[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_6));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[14]), "+f"(acc1[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_6));
                            }
                            int _trl_addr_7 = tmem_partials + (partial_stage * 256 + (unsigned int)wg_col_base + 32 + 16) + (row_base << 16);
                            tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_7);
                            {
                                unsigned long long _fma_acc_scale2_8;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_8) : "f"(combined));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[16]), "+f"(acc1[17]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_8));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[18]), "+f"(acc1[19]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_8));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[20]), "+f"(acc1[21]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_8));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[22]), "+f"(acc1[23]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_8));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[24]), "+f"(acc1[25]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_8));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[26]), "+f"(acc1[27]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_8));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[28]), "+f"(acc1[29]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_8));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[30]), "+f"(acc1[31]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_8));
                            }
                            if (k_inner == 0) {
                                uint32_t _mbar_token_3 = mbarrier_test_wait(partial_full_addr + (partial_stage ^ 1) * 8, partial_full_phase);
                                partial_full_token = _mbar_token_3;
                            }
                            int _trl_addr_9 = tmem_partials + (partial_stage * 256 + (unsigned int)wg_col_base + 64) + (row_base << 16);
                            tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_9);
                            {
                                unsigned long long _fma_acc_scale2_10;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_10) : "f"(combined));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[0]), "+f"(acc2[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[2]), "+f"(acc2[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[4]), "+f"(acc2[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[6]), "+f"(acc2[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[8]), "+f"(acc2[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[10]), "+f"(acc2[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[12]), "+f"(acc2[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[14]), "+f"(acc2[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_10));
                            }
                            int _trl_addr_11 = tmem_partials + (partial_stage * 256 + (unsigned int)wg_col_base + 64 + 16) + (row_base << 16);
                            tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_11);
                            {
                                unsigned long long _fma_acc_scale2_12;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_12) : "f"(combined));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[16]), "+f"(acc2[17]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[18]), "+f"(acc2[19]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[20]), "+f"(acc2[21]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[22]), "+f"(acc2[23]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[24]), "+f"(acc2[25]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[26]), "+f"(acc2[27]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[28]), "+f"(acc2[29]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[30]), "+f"(acc2[31]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_12));
                            }
                            int _trl_addr_13 = tmem_partials + (partial_stage * 256 + (unsigned int)wg_col_base + 96) + (row_base << 16);
                            tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_13);
                            {
                                unsigned long long _fma_acc_scale2_14;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_14) : "f"(combined));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[0]), "+f"(acc3[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[2]), "+f"(acc3[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[4]), "+f"(acc3[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[6]), "+f"(acc3[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[8]), "+f"(acc3[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[10]), "+f"(acc3[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[12]), "+f"(acc3[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[14]), "+f"(acc3[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_14));
                            }
                            int _trl_addr_15 = tmem_partials + (partial_stage * 256 + (unsigned int)wg_col_base + 96 + 16) + (row_base << 16);
                            tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_15);
                            {
                                unsigned long long _fma_acc_scale2_16;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_16) : "f"(combined));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[16]), "+f"(acc3[17]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[18]), "+f"(acc3[19]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[20]), "+f"(acc3[21]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[22]), "+f"(acc3[23]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[24]), "+f"(acc3[25]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[26]), "+f"(acc3[27]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[28]), "+f"(acc3[29]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[30]), "+f"(acc3[31]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_16));
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((partial_free_addr + (partial_stage) * 8) & 0xFEFFFFFF) : "memory");
                            }
                            partial_stage += 1;
                            if (partial_stage == 2) { partial_stage = 0; partial_full_phase ^= 1; }
                        }
                        _phase_scale_full ^= 1;
                    }
                    int homogeneous = 0;
                    if (run_begin == 0) {
                        if (run_end == tile_rows) {
                            homogeneous = 1;
                        }
                    }
                    if (homogeneous != 0) {
                        if (warp == 0) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        unsigned int epi_other_0 = epi_panel_cursor + 1;
                        if (epi_other_0 == 4) {
                            epi_other_0 = 0;
                        }
                        unsigned int epi_a_addr_0 = epi_staging_addr + epi_panel_cursor * 16384;
                        unsigned int epi_b_addr_0 = epi_staging_addr + epi_other_0 * 16384;
                        uint32_t acc0_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc0[_lp*2 + 0], acc0[_lp*2+1 + 0]));
                            acc0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        if (acc_wg == 0) {
                            #pragma unroll
                            for (int chunk = 0; chunk < 4; chunk++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_a_addr_0 + (unsigned int)(row * 128 + chunk * 16 ^ (row * 128 + chunk * 16 >> 7 & 7) << 4))), "r"(acc0_bf16[chunk * 4]), "r"(acc0_bf16[chunk * 4 + 1]), "r"(acc0_bf16[chunk * 4 + 2]), "r"(acc0_bf16[chunk * 4 + 3]) : "memory");
                            }
                        }
                        if (acc_wg == 1) {
                            #pragma unroll
                            for (int chunk_1 = 0; chunk_1 < 4; chunk_1++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_b_addr_0 + (unsigned int)(row * 128 + chunk_1 * 16 ^ (row * 128 + chunk_1 * 16 >> 7 & 7) << 4))), "r"(acc0_bf16[chunk_1 * 4]), "r"(acc0_bf16[chunk_1 * 4 + 1]), "r"(acc0_bf16[chunk_1 * 4 + 2]), "r"(acc0_bf16[chunk_1 * 4 + 3]) : "memory");
                            }
                        }
                        uint32_t acc1_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc1[_lp*2 + 0], acc1[_lp*2+1 + 0]));
                            acc1_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        if (acc_wg == 0) {
                            #pragma unroll
                            for (int chunk_2 = 0; chunk_2 < 4; chunk_2++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_a_addr_0 + (unsigned int)(row * 128 + (64 + chunk_2 * 16) ^ (row * 128 + (64 + chunk_2 * 16) >> 7 & 7) << 4))), "r"(acc1_bf16[chunk_2 * 4]), "r"(acc1_bf16[chunk_2 * 4 + 1]), "r"(acc1_bf16[chunk_2 * 4 + 2]), "r"(acc1_bf16[chunk_2 * 4 + 3]) : "memory");
                            }
                        }
                        if (acc_wg == 1) {
                            #pragma unroll
                            for (int chunk_3 = 0; chunk_3 < 4; chunk_3++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_b_addr_0 + (unsigned int)(row * 128 + (64 + chunk_3 * 16) ^ (row * 128 + (64 + chunk_3 * 16) >> 7 & 7) << 4))), "r"(acc1_bf16[chunk_3 * 4]), "r"(acc1_bf16[chunk_3 * 4 + 1]), "r"(acc1_bf16[chunk_3 * 4 + 2]), "r"(acc1_bf16[chunk_3 * 4 + 3]) : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                tma_store_2d(C_tma, n_tile * 256, m_tile * 128, epi_a_addr_0);
                                asm volatile("cp.async.bulk.commit_group;");
                                tma_store_2d(C_tma, n_tile * 256 + 128, m_tile * 128, epi_b_addr_0);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        epi_panel_cursor += 2;
                        if (epi_panel_cursor >= 4) {
                            epi_panel_cursor -= 4;
                        }
                        if (warp == 0) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        unsigned int epi_other_1 = epi_panel_cursor + 1;
                        if (epi_other_1 == 4) {
                            epi_other_1 = 0;
                        }
                        unsigned int epi_a_addr_1 = epi_staging_addr + epi_panel_cursor * 16384;
                        unsigned int epi_b_addr_1 = epi_staging_addr + epi_other_1 * 16384;
                        uint32_t acc2_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc2[_lp*2 + 0], acc2[_lp*2+1 + 0]));
                            acc2_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        if (acc_wg == 0) {
                            #pragma unroll
                            for (int chunk_4 = 0; chunk_4 < 4; chunk_4++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_a_addr_1 + (unsigned int)(row * 128 + chunk_4 * 16 ^ (row * 128 + chunk_4 * 16 >> 7 & 7) << 4))), "r"(acc2_bf16[chunk_4 * 4]), "r"(acc2_bf16[chunk_4 * 4 + 1]), "r"(acc2_bf16[chunk_4 * 4 + 2]), "r"(acc2_bf16[chunk_4 * 4 + 3]) : "memory");
                            }
                        }
                        if (acc_wg == 1) {
                            #pragma unroll
                            for (int chunk_5 = 0; chunk_5 < 4; chunk_5++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_b_addr_1 + (unsigned int)(row * 128 + chunk_5 * 16 ^ (row * 128 + chunk_5 * 16 >> 7 & 7) << 4))), "r"(acc2_bf16[chunk_5 * 4]), "r"(acc2_bf16[chunk_5 * 4 + 1]), "r"(acc2_bf16[chunk_5 * 4 + 2]), "r"(acc2_bf16[chunk_5 * 4 + 3]) : "memory");
                            }
                        }
                        uint32_t acc3_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc3[_lp*2 + 0], acc3[_lp*2+1 + 0]));
                            acc3_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        if (acc_wg == 0) {
                            #pragma unroll
                            for (int chunk_6 = 0; chunk_6 < 4; chunk_6++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_a_addr_1 + (unsigned int)(row * 128 + (64 + chunk_6 * 16) ^ (row * 128 + (64 + chunk_6 * 16) >> 7 & 7) << 4))), "r"(acc3_bf16[chunk_6 * 4]), "r"(acc3_bf16[chunk_6 * 4 + 1]), "r"(acc3_bf16[chunk_6 * 4 + 2]), "r"(acc3_bf16[chunk_6 * 4 + 3]) : "memory");
                            }
                        }
                        if (acc_wg == 1) {
                            #pragma unroll
                            for (int chunk_7 = 0; chunk_7 < 4; chunk_7++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_b_addr_1 + (unsigned int)(row * 128 + (64 + chunk_7 * 16) ^ (row * 128 + (64 + chunk_7 * 16) >> 7 & 7) << 4))), "r"(acc3_bf16[chunk_7 * 4]), "r"(acc3_bf16[chunk_7 * 4 + 1]), "r"(acc3_bf16[chunk_7 * 4 + 2]), "r"(acc3_bf16[chunk_7 * 4 + 3]) : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                tma_store_2d(C_tma, n_tile * 256 + 64, m_tile * 128, epi_a_addr_1);
                                asm volatile("cp.async.bulk.commit_group;");
                                tma_store_2d(C_tma, n_tile * 256 + 192, m_tile * 128, epi_b_addr_1);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        epi_panel_cursor += 2;
                        if (epi_panel_cursor >= 4) {
                            epi_panel_cursor -= 4;
                        }
                    } else {
                        int cluster_row = cta_rank * 128 + (tid & 127);
                        if (cluster_row >= run_begin) {
                            if (cluster_row < run_end) {
                                long long output_row = cluster_m_base;
                                output_row += cluster_row;
                                #pragma unroll
                                for (int vec = 0; vec < 32; vec += 8) {
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(acc0[vec + 0], acc0[vec + 1]);
                                        _pk[1] = __floats2bfloat162_rn(acc0[vec + 2], acc0[vec + 3]);
                                        _pk[2] = __floats2bfloat162_rn(acc0[vec + 4], acc0[vec + 5]);
                                        _pk[3] = __floats2bfloat162_rn(acc0[vec + 6], acc0[vec + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(C + (output_row * (long long)N + (long long)(n_tile * 256) + (long long)wg_col_base + (long long)vec)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(acc1[vec + 0], acc1[vec + 1]);
                                        _pk[1] = __floats2bfloat162_rn(acc1[vec + 2], acc1[vec + 3]);
                                        _pk[2] = __floats2bfloat162_rn(acc1[vec + 4], acc1[vec + 5]);
                                        _pk[3] = __floats2bfloat162_rn(acc1[vec + 6], acc1[vec + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(C + (output_row * (long long)N + (long long)(n_tile * 256) + (long long)wg_col_base + 32 + vec)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(acc2[vec + 0], acc2[vec + 1]);
                                        _pk[1] = __floats2bfloat162_rn(acc2[vec + 2], acc2[vec + 3]);
                                        _pk[2] = __floats2bfloat162_rn(acc2[vec + 4], acc2[vec + 5]);
                                        _pk[3] = __floats2bfloat162_rn(acc2[vec + 6], acc2[vec + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(C + (output_row * (long long)N + (long long)(n_tile * 256) + (long long)wg_col_base + 64 + vec)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(acc3[vec + 0], acc3[vec + 1]);
                                        _pk[1] = __floats2bfloat162_rn(acc3[vec + 2], acc3[vec + 3]);
                                        _pk[2] = __floats2bfloat162_rn(acc3[vec + 4], acc3[vec + 5]);
                                        _pk[3] = __floats2bfloat162_rn(acc3[vec + 6], acc3[vec + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(C + (output_row * (long long)N + (long long)(n_tile * 256) + (long long)wg_col_base + 96 + vec)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                }
                            }
                        }
                    }
                    run_begin = run_end;
                }
                cluster_m_tile += step_m;
                n_tile += step_n;
                if (n_tile >= n_tiles) {
                    n_tile -= n_tiles;
                    cluster_m_tile += 1;
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
            }
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int _phase_producers_done_0 = 0;
            unsigned int _phase_pair_exit_0 = 0;
            if (warp == 0) {
                mbarrier_wait(producers_done_addr, _phase_producers_done_0);
                _phase_producers_done_0 ^= 1;
                if (elect_sync()) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(pair_exit_addr), "r"(1 - cta_rank) : "memory");
                }
                mbarrier_wait(pair_exit_addr, _phase_pair_exit_0);
                _phase_pair_exit_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: mma_role ----
    if (warp == 8) {
        { // mma_role_main
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            taddr = tmem_addr_storage[0];
            tmem_partials = taddr;
            unsigned int ab_stage = 0;
            unsigned int ab_full_phase = 0;
            unsigned int partial_stage_1 = 0;
            int m_tiles_1 = (M - 1) / 256 + 1;
            int n_tiles_1 = N / 256;
            int total_tiles_1 = m_tiles_1 * n_tiles_1;
            unsigned int _phase_partial_free = 1;
            if (cta_rank == 0) {
                int cluster_m_tile_1 = cluster_id / (unsigned int)n_tiles_1;
                int n_tile_1 = cluster_id % (unsigned int)n_tiles_1;
                int step_m_1 = num_clusters / (unsigned int)n_tiles_1;
                int step_n_1 = num_clusters % (unsigned int)n_tiles_1;
                #pragma unroll 1
                for (int _tile_id = cluster_id; _tile_id < total_tiles_1; _tile_id += num_clusters) {
                    int cluster_m_base_1 = cluster_m_tile_1 * 256;
                    int tile_rows_1 = 256;
                    if (tile_rows_1 > M - cluster_m_base_1) {
                        tile_rows_1 = M - cluster_m_base_1;
                    }
                    int last_group_1 = 0;
                    if (lane == 0) {
                        last_group_1 = m_indices[cluster_m_base_1 + tile_rows_1 - 1];
                    }
                    int run_begin_1 = 0;
                    #pragma unroll 1
                    for (int segment_iter_1 = 0; segment_iter_1 < tile_rows_1; segment_iter_1++) {
                        if (run_begin_1 >= tile_rows_1) {
                            break;
                        }
                        int run_end_1 = tile_rows_1;
                        int group_1 = 0;
                        if (lane == 0) {
                            group_1 = m_indices[cluster_m_base_1 + run_begin_1];
                            if (group_1 != last_group_1) {
                                #pragma unroll 1
                                for (int probe_1 = run_begin_1 + 1; probe_1 < tile_rows_1; probe_1++) {
                                    int next_group_1 = m_indices[cluster_m_base_1 + probe_1];
                                    if (next_group_1 != group_1) {
                                        run_end_1 = probe_1;
                                        break;
                                    }
                                }
                            }
                        }
                        int _shfl_4 = __shfl_sync(0xFFFFFFFF, run_end_1, 0);
                        run_end_1 = _shfl_4;
                        int _shfl_5 = __shfl_sync(0xFFFFFFFF, group_1, 0);
                        group_1 = _shfl_5;
                        int k_blocks_1 = 8;
                        uint32_t _mbar_token_0 = mbarrier_try_wait(ab_full_addr + (ab_stage) * 8, ab_full_phase);
                        unsigned int ab_full_token = _mbar_token_0;
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < k_blocks_1; iter_k++) {
                            mbarrier_wait(partial_free_addr + (partial_stage_1) * 8, _phase_partial_free);
                            mbarrier_wait_token(ab_full_addr + (ab_stage) * 8, ab_full_phase, ab_full_token);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024;
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
                    "mov.b32 id, 272629776;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_partials + (partial_stage_1 * 256))), "r"(0));
                            elect_commit_cg2_multicast(ab_free_addr + (ab_stage) * 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(partial_full_addr + (partial_stage_1) * 8, (uint16_t)(3));
                            ab_stage += 1;
                            if (ab_stage == 5) { ab_stage = 0; ab_full_phase ^= 1; }
                            ab_full_token = 1;
                            if (k_blocks_1 > iter_k + 1) {
                                uint32_t _mbar_token_1 = mbarrier_try_wait(ab_full_addr + (ab_stage) * 8, ab_full_phase);
                                ab_full_token = _mbar_token_1;
                            }
                            partial_stage_1 += 1;
                            if (partial_stage_1 == 2) { partial_stage_1 = 0; _phase_partial_free ^= 1; }
                        }
                        run_begin_1 = run_end_1;
                    }
                    cluster_m_tile_1 += step_m_1;
                    n_tile_1 += step_n_1;
                    if (n_tile_1 >= n_tiles_1) {
                        n_tile_1 -= n_tiles_1;
                        cluster_m_tile_1 += 1;
                    }
                }
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
        }
    }
    // ---- Role: tma_role ----
    if (warp == 9) {
        { // tma_role_main
            unsigned int ab_stage_1 = 0;
            int m_tiles_2 = (M - 1) / 256 + 1;
            int n_tiles_2 = N / 256;
            int total_tiles_2 = m_tiles_2 * n_tiles_2;
            int cluster_m_tile_2 = cluster_id / (unsigned int)n_tiles_2;
            int n_tile_2 = cluster_id % (unsigned int)n_tiles_2;
            int step_m_2 = num_clusters / (unsigned int)n_tiles_2;
            int step_n_2 = num_clusters % (unsigned int)n_tiles_2;
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (int tile_id_1 = cluster_id; tile_id_1 < total_tiles_2; tile_id_1 += num_clusters) {
                int m_tile_1 = cluster_m_tile_2 * 2 + cta_rank;
                int cluster_m_base_2 = cluster_m_tile_2 * 256;
                int tile_rows_2 = 256;
                if (tile_rows_2 > M - cluster_m_base_2) {
                    tile_rows_2 = M - cluster_m_base_2;
                }
                int last_group_2 = 0;
                if (lane == 0) {
                    last_group_2 = m_indices[cluster_m_base_2 + tile_rows_2 - 1];
                }
                int run_begin_2 = 0;
                #pragma unroll 1
                for (int segment_iter_2 = 0; segment_iter_2 < tile_rows_2; segment_iter_2++) {
                    if (run_begin_2 >= tile_rows_2) {
                        break;
                    }
                    int run_end_2 = tile_rows_2;
                    int group_2 = 0;
                    if (lane == 0) {
                        group_2 = m_indices[cluster_m_base_2 + run_begin_2];
                        if (group_2 != last_group_2) {
                            #pragma unroll 1
                            for (int probe_2 = run_begin_2 + 1; probe_2 < tile_rows_2; probe_2++) {
                                int next_group_2 = m_indices[cluster_m_base_2 + probe_2];
                                if (next_group_2 != group_2) {
                                    run_end_2 = probe_2;
                                    break;
                                }
                            }
                        }
                    }
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, run_end_2, 0);
                    run_end_2 = _shfl_0;
                    int _shfl_1 = __shfl_sync(0xFFFFFFFF, group_2, 0);
                    group_2 = _shfl_1;
                    int k_blocks_2 = 8;
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < k_blocks_2; iter_k_1++) {
                        mbarrier_wait(ab_free_addr + (ab_stage_1) * 8, _phase_ab_free);
                        if (elect_sync()) {
                            tma_3d_gmem2smem_cta2(smem_a_addr + ab_stage_1 * 16384, A, 0, m_tile_1 * 128, iter_k_1, ((ab_full_addr + (ab_stage_1) * 8) & 0xFEFFFFFF));
                            tma_4d_gmem2smem_cta2(smem_b_addr + ab_stage_1 * 16384, B, 0, n_tile_2 * 256 + cta_rank * 128, iter_k_1, group_2, ((ab_full_addr + (ab_stage_1) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((ab_full_addr + (ab_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        }
                        ab_stage_1 += 1;
                        if (ab_stage_1 == 5) { ab_stage_1 = 0; _phase_ab_free ^= 1; }
                    }
                    run_begin_2 = run_end_2;
                }
                cluster_m_tile_2 += step_m_2;
                n_tile_2 += step_n_2;
                if (n_tile_2 >= n_tiles_2) {
                    n_tile_2 -= n_tiles_2;
                    cluster_m_tile_2 += 1;
                }
            }
            if (elect_sync()) {
                #pragma unroll
                for (int _ab_drain = 0; _ab_drain < 5; _ab_drain++) {
                    mbarrier_wait(ab_free_addr + (ab_stage_1) * 8, _phase_ab_free);
                    ab_stage_1 += 1;
                    if (ab_stage_1 == 5) { ab_stage_1 = 0; _phase_ab_free ^= 1; }
                }
                mbarrier_arrive(producers_done_addr);
            }
        }
    }
    // ---- Role: scale_role ----
    if (warp >= 10 && warp <= 11) {
        { // scale_role_main
            const int pair = warp - 10;
            unsigned int scale_stage_1 = 0;
            int m_tiles_3 = (M - 1) / 256 + 1;
            int n_tiles_3 = N / 256;
            int total_tiles_3 = m_tiles_3 * n_tiles_3;
            int cluster_m_tile_3 = cluster_id / (unsigned int)n_tiles_3;
            int n_tile_3 = cluster_id % (unsigned int)n_tiles_3;
            int step_m_3 = num_clusters / (unsigned int)n_tiles_3;
            int step_n_3 = num_clusters % (unsigned int)n_tiles_3;
            unsigned int _phase_scale_free = 1;
            #pragma unroll 1
            for (int tile_id_2 = cluster_id; tile_id_2 < total_tiles_3; tile_id_2 += num_clusters) {
                int m_tile_2 = cluster_m_tile_3 * 2 + cta_rank;
                int cluster_m_base_3 = cluster_m_tile_3 * 256;
                int tile_rows_3 = 256;
                if (tile_rows_3 > M - cluster_m_base_3) {
                    tile_rows_3 = M - cluster_m_base_3;
                }
                int last_group_3 = 0;
                if (lane == 0) {
                    last_group_3 = m_indices[cluster_m_base_3 + tile_rows_3 - 1];
                }
                int run_begin_3 = 0;
                #pragma unroll 1
                for (int segment_iter_3 = 0; segment_iter_3 < tile_rows_3; segment_iter_3++) {
                    if (run_begin_3 >= tile_rows_3) {
                        break;
                    }
                    int run_end_3 = tile_rows_3;
                    int group_3 = 0;
                    if (lane == 0) {
                        group_3 = m_indices[cluster_m_base_3 + run_begin_3];
                        if (group_3 != last_group_3) {
                            #pragma unroll 1
                            for (int probe_3 = run_begin_3 + 1; probe_3 < tile_rows_3; probe_3++) {
                                int next_group_3 = m_indices[cluster_m_base_3 + probe_3];
                                if (next_group_3 != group_3) {
                                    run_end_3 = probe_3;
                                    break;
                                }
                            }
                        }
                    }
                    int _shfl_2 = __shfl_sync(0xFFFFFFFF, run_end_3, 0);
                    run_end_3 = _shfl_2;
                    int _shfl_3 = __shfl_sync(0xFFFFFFFF, group_3, 0);
                    group_3 = _shfl_3;
                    int k_blocks_3 = 8;
                    int scale_groups_1 = k_blocks_3 / 4;
                    int n_blocks = N / 128;
                    #pragma unroll 1
                    for (int scale_group_1 = 0; scale_group_1 < scale_groups_1; scale_group_1++) {
                        mbarrier_wait(scale_free_addr + (scale_stage_1) * 8, _phase_scale_free);
                        int stage_base = scale_stage_1 * 128 * 4;
                        #pragma unroll
                        for (int chunk_8 = 0; chunk_8 < 4; chunk_8++) {
                            int row_1 = chunk_8 * 32 + lane;
                            int g_row = m_tile_2 * 128 + row_1;
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8, %2;"
                                :: "r"(smem_ascale_addr + (unsigned int)((stage_base + (pair * 128 + row_1) * 2) * 4)), "l"(a_scale + (g_row * k_blocks_3 + scale_group_1 * 4 + pair * 2)), "r"((g_row < M) ? 8 : 0));
                        }
                        if (warp == 10) {
                            #pragma unroll
                            for (int n_half = 0; n_half < 2; n_half++) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred p;\n\t"
                                    "setp.ne.b32 p, %0, 0;\n\t"
                                    "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                                    "}"
                                    :: "r"((lane == 0) ? 1 : 0), "r"(smem_bscale_addr + (scale_stage_1 * 2 + (unsigned int)n_half) * 4 * 4), "l"(b_scale + ((group_3 * n_blocks + n_tile_3 * 2 + n_half) * k_blocks_3 + scale_group_1 * 4)));
                            }
                        }
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(scale_full_addr + (scale_stage_1) * 8) : "memory");
                        _phase_scale_free ^= 1;
                    }
                    run_begin_3 = run_end_3;
                }
                cluster_m_tile_3 += step_m_3;
                n_tile_3 += step_n_3;
                if (n_tile_3 >= n_tiles_3) {
                    n_tile_3 -= n_tiles_3;
                    cluster_m_tile_3 += 1;
                }
            }
            #pragma unroll
            for (int _drain = 0; _drain < 1; _drain++) {
                mbarrier_wait(scale_free_addr + (scale_stage_1) * 8, _phase_scale_free);
                _phase_scale_free ^= 1;
            }
            if (elect_sync()) {
                mbarrier_arrive(producers_done_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
