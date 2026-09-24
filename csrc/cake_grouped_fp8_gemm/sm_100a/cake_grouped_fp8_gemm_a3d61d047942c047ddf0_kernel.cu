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
#define TMEM_NCOLS 512
#define TMEM_PARTIALS_OFFSET 0
#define NUM_AB_PIPE_STAGES 6
#define NUM_PARTIAL_PIPE_STAGES 4
#define NUM_PARTIAL_PAIR_PIPE_STAGES 2
#define NUM_SCALE_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 0
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 98304
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_EPI_STAGING_0_OFF 196608
#define SMEM_EPI_STAGING_0_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_0_STRIDE 8192
#define SMEM_EPI_STAGING_1_OFF 204800
#define SMEM_EPI_STAGING_1_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_1_STRIDE 8192
#define SMEM_EPI_STAGING_2_OFF 212992
#define SMEM_EPI_STAGING_2_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_2_STRIDE 8192
#define SMEM_EPI_STAGING_3_OFF 221184
#define SMEM_EPI_STAGING_3_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_3_STRIDE 8192
#define SMEM_SMEM_ASCALE_OFF 229376
#define SMEM_SMEM_ASCALE_STAGE_BYTES 2048
#define SMEM_SMEM_ASCALE_STRIDE 2048
#define SMEM_SMEM_BSCALE_OFF 231424
#define SMEM_SMEM_BSCALE_STAGE_BYTES 16
#define SMEM_SMEM_BSCALE_STRIDE 16
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


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
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

__global__ __launch_bounds__(384, 1) void
kernel_cake_grouped_fp8_gemm_a3d61d047942c047ddf0(CakeTensorMap const* A, CakeTensorMap const* B, CakeTensorMap const* C_tma, __nv_bfloat16* __restrict__ C, CakeTensorMap const* a_scale, CakeTensorMap const* b_scale, int* __restrict__ m_indices, int M, int N, int K, int G)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem + 231440;
    #define ab_full_addr (mbar_base + 0)
    #define ab_free_addr (mbar_base + 48)
    #define partial_full_addr (mbar_base + 96)
    #define partial_free_addr (mbar_base + 112)
    #define scale_full_addr (mbar_base + 144)
    #define scale_free_addr (mbar_base + 152)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(C_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(a_scale)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(b_scale)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int smem_a_addr = smem + 0;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 98304);
    const int smem_b_addr = smem + 98304;
    __nv_bfloat16* epi_staging_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 196608);
    const int epi_staging_0_addr = smem + 196608;
    __nv_bfloat16* epi_staging_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 204800);
    const int epi_staging_1_addr = smem + 204800;
    __nv_bfloat16* epi_staging_2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 212992);
    const int epi_staging_2_addr = smem + 212992;
    __nv_bfloat16* epi_staging_3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 221184);
    const int epi_staging_3_addr = smem + 221184;
    float* smem_ascale = reinterpret_cast<float*>(smem_raw + 229376);
    const int smem_ascale_addr = smem + 229376;
    float* smem_bscale = reinterpret_cast<float*>(smem_raw + 231424);
    const int smem_bscale_addr = smem + 231424;
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 231600);
    int taddr;
    int tmem_partials;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[231440..231600)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 6 barriers, init_count=1
            mbarrier_init(smem + 231440, 1);
            mbarrier_init(smem + 231448, 1);
            mbarrier_init(smem + 231456, 1);
            mbarrier_init(smem + 231464, 1);
            mbarrier_init(smem + 231472, 1);
            mbarrier_init(smem + 231480, 1);
            // ab_free: 6 barriers, init_count=1
            mbarrier_init(smem + 231488, 1);
            mbarrier_init(smem + 231496, 1);
            mbarrier_init(smem + 231504, 1);
            mbarrier_init(smem + 231512, 1);
            mbarrier_init(smem + 231520, 1);
            mbarrier_init(smem + 231528, 1);
            // --- pipeline 'partial_pair_pipe' ---
            // partial_full: 2 barriers, init_count=1
            mbarrier_init(smem + 231536, 1);
            mbarrier_init(smem + 231544, 1);
            // --- pipeline 'partial_pipe' ---
            // partial_free: 4 barriers, init_count=8
            mbarrier_init(smem + 231552, 8);
            mbarrier_init(smem + 231560, 8);
            mbarrier_init(smem + 231568, 8);
            mbarrier_init(smem + 231576, 8);
            // --- pipeline 'scale_pipe' ---
            // scale_full: 1 barriers, init_count=1
            mbarrier_init(smem + 231584, 1);
            // scale_free: 1 barriers, init_count=256
            mbarrier_init(smem + 231592, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

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
                asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold_0), "r"(512) : "memory");
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            taddr = tmem_addr_storage[0];
            tmem_partials = taddr;
            if (warp == 0) {
                asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
            }
            unsigned int partial_pair_stage = 0;
            unsigned int partial_full_phase = 0;
            unsigned int scale_stage = 0;
            const int acc_wg = warp / 4;
            const int acc_warp_in_wg = warp % 4;
            int wg_col_base = acc_wg * 64;
            int row_base = acc_warp_in_wg * 32;
            int row = row_base + lane;
            int m_tiles = (M + 128 - 1) / 128;
            int n_tiles = N / 128;
            int total_tiles = m_tiles * n_tiles;
            int m_tile = bid / n_tiles;
            int n_tile = bid % n_tiles;
            int dm_tile = num_bids / n_tiles;
            int dn_tile = num_bids % n_tiles;
            unsigned int _phase_scale_full = 0;
            #pragma unroll 1
            for (int tile_id = bid; tile_id < total_tiles; tile_id += num_bids) {
                int tile_rows = 128;
                if (m_tile * 128 + tile_rows > M) {
                    tile_rows = M - m_tile * 128;
                }
                int last_group = 0;
                if (lane == 0) {
                    last_group = m_indices[m_tile * 128 + tile_rows - 1];
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
                        group = m_indices[m_tile * 128 + run_begin];
                        if (group != last_group) {
                            #pragma unroll 1
                            for (int probe = run_begin + 1; probe < tile_rows; probe++) {
                                int next_group = m_indices[m_tile * 128 + probe];
                                if (next_group != group) {
                                    run_end = probe;
                                    break;
                                }
                            }
                        }
                    }
                    int _shfl_8 = __shfl_sync(0xFFFFFFFF, run_end, 0);
                    run_end = _shfl_8;
                    int _shfl_9 = __shfl_sync(0xFFFFFFFF, group, 0);
                    group = _shfl_9;
                    float acc0[32];
                    float acc1[32];
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
                    float partial_fragment_a[16];
                    float partial_fragment_b[16];
                    float partial_fragment_c[16];
                    float partial_fragment_d[16];
                    float next_fragment_a[16];
                    float next_fragment_b[16];
                    float next_fragment_c[16];
                    float next_fragment_d[16];
                    int k_blocks = K / 128;
                    int scale_groups = k_blocks / 4;
                    #pragma unroll 1
                    for (int scale_group = 0; scale_group < scale_groups; scale_group++) {
                        mbarrier_wait(scale_full_addr + (scale_stage) * 8, _phase_scale_full);
                        int scale_base = scale_stage * 128 * 4 + (unsigned int)(row * 4);
                        float scale_products[4];
                        #pragma unroll
                        for (int preload_k = 0; preload_k < 4; preload_k++) {
                            float a_s = smem_ascale[scale_base + preload_k];
                            float b_s = smem_bscale[scale_stage * 4 + (unsigned int)preload_k];
                            scale_products[preload_k] = a_s * b_s;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(scale_free_addr + (scale_stage) * 8);
                        _phase_scale_full ^= 1;
                        #pragma unroll
                        for (int pair_inner = 0; pair_inner < 2; pair_inner++) {
                            mbarrier_wait(partial_full_addr + (partial_pair_stage) * 8, partial_full_phase);
                            unsigned int partial_stage0 = partial_pair_stage * 2;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            float combined0 = scale_products[pair_inner * 2];
                            int _trl_addr_1 = tmem_partials + (partial_stage0 * 128 + (unsigned int)wg_col_base) + (row_base << 16);
                            tmem_ld_x16(&partial_fragment_a[0], _trl_addr_1);
                            int _trl_addr_2 = tmem_partials + (partial_stage0 * 128 + (unsigned int)wg_col_base + 16) + (row_base << 16);
                            tmem_ld_x16(&partial_fragment_b[0], _trl_addr_2);
                            int _trl_addr_3 = tmem_partials + (partial_stage0 * 128 + (unsigned int)wg_col_base + 32) + (row_base << 16);
                            tmem_ld_x16(&partial_fragment_c[0], _trl_addr_3);
                            int _trl_addr_4 = tmem_partials + (partial_stage0 * 128 + (unsigned int)wg_col_base + 32 + 16) + (row_base << 16);
                            tmem_ld_x16(&partial_fragment_d[0], _trl_addr_4);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            if (elect_sync()) {
                                mbarrier_arrive(partial_free_addr + (partial_pair_stage * 2) * 8);
                            }
                            unsigned int partial_stage1 = partial_pair_stage * 2 + 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            float combined1 = scale_products[pair_inner * 2 + 1];
                            int _trl_addr_5 = tmem_partials + (partial_stage1 * 128 + (unsigned int)wg_col_base) + (row_base << 16);
                            tmem_ld_x16(&next_fragment_a[0], _trl_addr_5);
                            int _trl_addr_6 = tmem_partials + (partial_stage1 * 128 + (unsigned int)wg_col_base + 16) + (row_base << 16);
                            tmem_ld_x16(&next_fragment_b[0], _trl_addr_6);
                            int _trl_addr_7 = tmem_partials + (partial_stage1 * 128 + (unsigned int)wg_col_base + 32) + (row_base << 16);
                            tmem_ld_x16(&next_fragment_c[0], _trl_addr_7);
                            int _trl_addr_8 = tmem_partials + (partial_stage1 * 128 + (unsigned int)wg_col_base + 32 + 16) + (row_base << 16);
                            tmem_ld_x16(&next_fragment_d[0], _trl_addr_8);
                            {
                                unsigned long long _fma_acc_scale2_9;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_9) : "f"(combined0));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[0]), "+f"(acc0[1]) : "f"(partial_fragment_a[0]), "f"(partial_fragment_a[1]), "l"(_fma_acc_scale2_9));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[2]), "+f"(acc0[3]) : "f"(partial_fragment_a[2]), "f"(partial_fragment_a[3]), "l"(_fma_acc_scale2_9));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[4]), "+f"(acc0[5]) : "f"(partial_fragment_a[4]), "f"(partial_fragment_a[5]), "l"(_fma_acc_scale2_9));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[6]), "+f"(acc0[7]) : "f"(partial_fragment_a[6]), "f"(partial_fragment_a[7]), "l"(_fma_acc_scale2_9));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[8]), "+f"(acc0[9]) : "f"(partial_fragment_a[8]), "f"(partial_fragment_a[9]), "l"(_fma_acc_scale2_9));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[10]), "+f"(acc0[11]) : "f"(partial_fragment_a[10]), "f"(partial_fragment_a[11]), "l"(_fma_acc_scale2_9));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[12]), "+f"(acc0[13]) : "f"(partial_fragment_a[12]), "f"(partial_fragment_a[13]), "l"(_fma_acc_scale2_9));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[14]), "+f"(acc0[15]) : "f"(partial_fragment_a[14]), "f"(partial_fragment_a[15]), "l"(_fma_acc_scale2_9));
                            }
                            {
                                unsigned long long _fma_acc_scale2_10;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_10) : "f"(combined0));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[16]), "+f"(acc0[17]) : "f"(partial_fragment_b[0]), "f"(partial_fragment_b[1]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[18]), "+f"(acc0[19]) : "f"(partial_fragment_b[2]), "f"(partial_fragment_b[3]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[20]), "+f"(acc0[21]) : "f"(partial_fragment_b[4]), "f"(partial_fragment_b[5]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[22]), "+f"(acc0[23]) : "f"(partial_fragment_b[6]), "f"(partial_fragment_b[7]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[24]), "+f"(acc0[25]) : "f"(partial_fragment_b[8]), "f"(partial_fragment_b[9]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[26]), "+f"(acc0[27]) : "f"(partial_fragment_b[10]), "f"(partial_fragment_b[11]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[28]), "+f"(acc0[29]) : "f"(partial_fragment_b[12]), "f"(partial_fragment_b[13]), "l"(_fma_acc_scale2_10));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[30]), "+f"(acc0[31]) : "f"(partial_fragment_b[14]), "f"(partial_fragment_b[15]), "l"(_fma_acc_scale2_10));
                            }
                            {
                                unsigned long long _fma_acc_scale2_11;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_11) : "f"(combined0));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[0]), "+f"(acc1[1]) : "f"(partial_fragment_c[0]), "f"(partial_fragment_c[1]), "l"(_fma_acc_scale2_11));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[2]), "+f"(acc1[3]) : "f"(partial_fragment_c[2]), "f"(partial_fragment_c[3]), "l"(_fma_acc_scale2_11));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[4]), "+f"(acc1[5]) : "f"(partial_fragment_c[4]), "f"(partial_fragment_c[5]), "l"(_fma_acc_scale2_11));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[6]), "+f"(acc1[7]) : "f"(partial_fragment_c[6]), "f"(partial_fragment_c[7]), "l"(_fma_acc_scale2_11));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[8]), "+f"(acc1[9]) : "f"(partial_fragment_c[8]), "f"(partial_fragment_c[9]), "l"(_fma_acc_scale2_11));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[10]), "+f"(acc1[11]) : "f"(partial_fragment_c[10]), "f"(partial_fragment_c[11]), "l"(_fma_acc_scale2_11));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[12]), "+f"(acc1[13]) : "f"(partial_fragment_c[12]), "f"(partial_fragment_c[13]), "l"(_fma_acc_scale2_11));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[14]), "+f"(acc1[15]) : "f"(partial_fragment_c[14]), "f"(partial_fragment_c[15]), "l"(_fma_acc_scale2_11));
                            }
                            {
                                unsigned long long _fma_acc_scale2_12;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_12) : "f"(combined0));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[16]), "+f"(acc1[17]) : "f"(partial_fragment_d[0]), "f"(partial_fragment_d[1]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[18]), "+f"(acc1[19]) : "f"(partial_fragment_d[2]), "f"(partial_fragment_d[3]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[20]), "+f"(acc1[21]) : "f"(partial_fragment_d[4]), "f"(partial_fragment_d[5]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[22]), "+f"(acc1[23]) : "f"(partial_fragment_d[6]), "f"(partial_fragment_d[7]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[24]), "+f"(acc1[25]) : "f"(partial_fragment_d[8]), "f"(partial_fragment_d[9]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[26]), "+f"(acc1[27]) : "f"(partial_fragment_d[10]), "f"(partial_fragment_d[11]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[28]), "+f"(acc1[29]) : "f"(partial_fragment_d[12]), "f"(partial_fragment_d[13]), "l"(_fma_acc_scale2_12));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[30]), "+f"(acc1[31]) : "f"(partial_fragment_d[14]), "f"(partial_fragment_d[15]), "l"(_fma_acc_scale2_12));
                            }
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            {
                                unsigned long long _fma_acc_scale2_13;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_13) : "f"(combined1));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[0]), "+f"(acc0[1]) : "f"(next_fragment_a[0]), "f"(next_fragment_a[1]), "l"(_fma_acc_scale2_13));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[2]), "+f"(acc0[3]) : "f"(next_fragment_a[2]), "f"(next_fragment_a[3]), "l"(_fma_acc_scale2_13));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[4]), "+f"(acc0[5]) : "f"(next_fragment_a[4]), "f"(next_fragment_a[5]), "l"(_fma_acc_scale2_13));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[6]), "+f"(acc0[7]) : "f"(next_fragment_a[6]), "f"(next_fragment_a[7]), "l"(_fma_acc_scale2_13));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[8]), "+f"(acc0[9]) : "f"(next_fragment_a[8]), "f"(next_fragment_a[9]), "l"(_fma_acc_scale2_13));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[10]), "+f"(acc0[11]) : "f"(next_fragment_a[10]), "f"(next_fragment_a[11]), "l"(_fma_acc_scale2_13));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[12]), "+f"(acc0[13]) : "f"(next_fragment_a[12]), "f"(next_fragment_a[13]), "l"(_fma_acc_scale2_13));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[14]), "+f"(acc0[15]) : "f"(next_fragment_a[14]), "f"(next_fragment_a[15]), "l"(_fma_acc_scale2_13));
                            }
                            {
                                unsigned long long _fma_acc_scale2_14;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_14) : "f"(combined1));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[16]), "+f"(acc0[17]) : "f"(next_fragment_b[0]), "f"(next_fragment_b[1]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[18]), "+f"(acc0[19]) : "f"(next_fragment_b[2]), "f"(next_fragment_b[3]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[20]), "+f"(acc0[21]) : "f"(next_fragment_b[4]), "f"(next_fragment_b[5]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[22]), "+f"(acc0[23]) : "f"(next_fragment_b[6]), "f"(next_fragment_b[7]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[24]), "+f"(acc0[25]) : "f"(next_fragment_b[8]), "f"(next_fragment_b[9]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[26]), "+f"(acc0[27]) : "f"(next_fragment_b[10]), "f"(next_fragment_b[11]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[28]), "+f"(acc0[29]) : "f"(next_fragment_b[12]), "f"(next_fragment_b[13]), "l"(_fma_acc_scale2_14));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[30]), "+f"(acc0[31]) : "f"(next_fragment_b[14]), "f"(next_fragment_b[15]), "l"(_fma_acc_scale2_14));
                            }
                            {
                                unsigned long long _fma_acc_scale2_15;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_15) : "f"(combined1));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[0]), "+f"(acc1[1]) : "f"(next_fragment_c[0]), "f"(next_fragment_c[1]), "l"(_fma_acc_scale2_15));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[2]), "+f"(acc1[3]) : "f"(next_fragment_c[2]), "f"(next_fragment_c[3]), "l"(_fma_acc_scale2_15));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[4]), "+f"(acc1[5]) : "f"(next_fragment_c[4]), "f"(next_fragment_c[5]), "l"(_fma_acc_scale2_15));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[6]), "+f"(acc1[7]) : "f"(next_fragment_c[6]), "f"(next_fragment_c[7]), "l"(_fma_acc_scale2_15));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[8]), "+f"(acc1[9]) : "f"(next_fragment_c[8]), "f"(next_fragment_c[9]), "l"(_fma_acc_scale2_15));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[10]), "+f"(acc1[11]) : "f"(next_fragment_c[10]), "f"(next_fragment_c[11]), "l"(_fma_acc_scale2_15));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[12]), "+f"(acc1[13]) : "f"(next_fragment_c[12]), "f"(next_fragment_c[13]), "l"(_fma_acc_scale2_15));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[14]), "+f"(acc1[15]) : "f"(next_fragment_c[14]), "f"(next_fragment_c[15]), "l"(_fma_acc_scale2_15));
                            }
                            {
                                unsigned long long _fma_acc_scale2_16;
                                asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_16) : "f"(combined1));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[16]), "+f"(acc1[17]) : "f"(next_fragment_d[0]), "f"(next_fragment_d[1]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[18]), "+f"(acc1[19]) : "f"(next_fragment_d[2]), "f"(next_fragment_d[3]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[20]), "+f"(acc1[21]) : "f"(next_fragment_d[4]), "f"(next_fragment_d[5]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[22]), "+f"(acc1[23]) : "f"(next_fragment_d[6]), "f"(next_fragment_d[7]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[24]), "+f"(acc1[25]) : "f"(next_fragment_d[8]), "f"(next_fragment_d[9]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[26]), "+f"(acc1[27]) : "f"(next_fragment_d[10]), "f"(next_fragment_d[11]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[28]), "+f"(acc1[29]) : "f"(next_fragment_d[12]), "f"(next_fragment_d[13]), "l"(_fma_acc_scale2_16));
                                asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[30]), "+f"(acc1[31]) : "f"(next_fragment_d[14]), "f"(next_fragment_d[15]), "l"(_fma_acc_scale2_16));
                            }
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            if (elect_sync()) {
                                mbarrier_arrive(partial_free_addr + (partial_pair_stage * 2 + 1) * 8);
                            }
                            partial_pair_stage += 1;
                            if (partial_pair_stage == 2) { partial_pair_stage = 0; partial_full_phase ^= 1; }
                        }
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
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        unsigned int epi_stage = 0;
                        int c_stage_row = epi_stage * 128 + (unsigned int)row;
                        uint32_t acc0_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc0[_lp*2 + 0], acc0[_lp*2+1 + 0]));
                            acc0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t acc1_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc1[_lp*2 + 0], acc1[_lp*2+1 + 0]));
                            acc1_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        if (acc_wg == 0) {
                            #pragma unroll
                            for (int chunk = 0; chunk < 4; chunk++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_0_addr + (unsigned int)(c_stage_row * 64 + chunk * 16 ^ (c_stage_row * 64 + chunk * 16 >> 7 & 3) << 4))), "r"(acc0_bf16[chunk * 4]), "r"(acc0_bf16[chunk * 4 + 1]), "r"(acc0_bf16[chunk * 4 + 2]), "r"(acc0_bf16[chunk * 4 + 3]) : "memory");
                            }
                            #pragma unroll
                            for (int chunk_1 = 0; chunk_1 < 4; chunk_1++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_1_addr + (unsigned int)(c_stage_row * 64 + chunk_1 * 16 ^ (c_stage_row * 64 + chunk_1 * 16 >> 7 & 3) << 4))), "r"(acc1_bf16[chunk_1 * 4]), "r"(acc1_bf16[chunk_1 * 4 + 1]), "r"(acc1_bf16[chunk_1 * 4 + 2]), "r"(acc1_bf16[chunk_1 * 4 + 3]) : "memory");
                            }
                        }
                        if (acc_wg == 1) {
                            #pragma unroll
                            for (int chunk_2 = 0; chunk_2 < 4; chunk_2++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_2_addr + (unsigned int)(c_stage_row * 64 + chunk_2 * 16 ^ (c_stage_row * 64 + chunk_2 * 16 >> 7 & 3) << 4))), "r"(acc0_bf16[chunk_2 * 4]), "r"(acc0_bf16[chunk_2 * 4 + 1]), "r"(acc0_bf16[chunk_2 * 4 + 2]), "r"(acc0_bf16[chunk_2 * 4 + 3]) : "memory");
                            }
                            #pragma unroll
                            for (int chunk_3 = 0; chunk_3 < 4; chunk_3++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_3_addr + (unsigned int)(c_stage_row * 64 + chunk_3 * 16 ^ (c_stage_row * 64 + chunk_3 * 16 >> 7 & 3) << 4))), "r"(acc1_bf16[chunk_3 * 4]), "r"(acc1_bf16[chunk_3 * 4 + 1]), "r"(acc1_bf16[chunk_3 * 4 + 2]), "r"(acc1_bf16[chunk_3 * 4 + 3]) : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int staging_offset = epi_stage * 8192;
                                tma_store_3d(C_tma, 0, m_tile * 128, n_tile * 4, epi_staging_0_addr + (unsigned int)staging_offset);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    } else if (row >= run_begin) {
                        if (row < run_end) {
                            long long output_row = m_tile * 128 + row;
                            #pragma unroll
                            for (int vec = 0; vec < 32; vec += 8) {
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(acc0[vec + 0], acc0[vec + 1]);
                                    _pk[1] = __floats2bfloat162_rn(acc0[vec + 2], acc0[vec + 3]);
                                    _pk[2] = __floats2bfloat162_rn(acc0[vec + 4], acc0[vec + 5]);
                                    _pk[3] = __floats2bfloat162_rn(acc0[vec + 6], acc0[vec + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(C + (output_row * (long long)N + (long long)(n_tile * 128) + (long long)wg_col_base + (long long)vec)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(acc1[vec + 0], acc1[vec + 1]);
                                    _pk[1] = __floats2bfloat162_rn(acc1[vec + 2], acc1[vec + 3]);
                                    _pk[2] = __floats2bfloat162_rn(acc1[vec + 4], acc1[vec + 5]);
                                    _pk[3] = __floats2bfloat162_rn(acc1[vec + 6], acc1[vec + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(C + (output_row * (long long)N + (long long)(n_tile * 128) + (long long)wg_col_base + 32 + vec)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    }
                    run_begin = run_end;
                }
                m_tile += dm_tile;
                n_tile += dn_tile;
                if (n_tile >= n_tiles) {
                    n_tile -= n_tiles;
                    m_tile += 1;
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
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
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
            unsigned int partial_pair_stage_1 = 0;
            unsigned int partial_free_phase = 1;
            int m_tiles_1 = (M + 128 - 1) / 128;
            int n_tiles_1 = N / 128;
            int total_tiles_1 = m_tiles_1 * n_tiles_1;
            #pragma unroll 1
            for (int _tile_id = bid; _tile_id < total_tiles_1; _tile_id += num_bids) {
                int m_tile_1 = _tile_id / n_tiles_1;
                int tile_rows_1 = 128;
                if (m_tile_1 * 128 + tile_rows_1 > M) {
                    tile_rows_1 = M - m_tile_1 * 128;
                }
                int last_group_1 = 0;
                if (lane == 0) {
                    last_group_1 = m_indices[m_tile_1 * 128 + tile_rows_1 - 1];
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
                        group_1 = m_indices[m_tile_1 * 128 + run_begin_1];
                        if (group_1 != last_group_1) {
                            #pragma unroll 1
                            for (int probe_1 = run_begin_1 + 1; probe_1 < tile_rows_1; probe_1++) {
                                int next_group_1 = m_indices[m_tile_1 * 128 + probe_1];
                                if (next_group_1 != group_1) {
                                    run_end_1 = probe_1;
                                    break;
                                }
                            }
                        }
                    }
                    int _shfl_6 = __shfl_sync(0xFFFFFFFF, run_end_1, 0);
                    run_end_1 = _shfl_6;
                    int _shfl_7 = __shfl_sync(0xFFFFFFFF, group_1, 0);
                    group_1 = _shfl_7;
                    int k_blocks_1 = K / 128;
                    uint32_t _mbar_token_0 = mbarrier_try_wait(ab_full_addr + (ab_stage) * 8, ab_full_phase);
                    unsigned int ab_full_token = _mbar_token_0;
                    int k_pairs = k_blocks_1 / 2;
                    #pragma unroll 1
                    for (int pair_k = 0; pair_k < k_pairs; pair_k++) {
                        #pragma unroll
                        for (int pair_half = 0; pair_half < 2; pair_half++) {
                            unsigned int partial_stage = partial_pair_stage_1 * 2 + (unsigned int)pair_half;
                            mbarrier_wait(partial_free_addr + (partial_stage) * 8, partial_free_phase);
                            mbarrier_wait_token(ab_full_addr + (ab_stage) * 8, ab_full_phase, ab_full_token);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024);
                            int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024);
                            {
                                uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                                uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_partials + (partial_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 0);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_partials + (partial_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_partials + (partial_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_partials + (partial_stage * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136314896, 1);
                                }
                            }
                            elect_commit(ab_free_addr + (ab_stage) * 8);
                            ab_stage += 1;
                            if (ab_stage == 6) { ab_stage = 0; ab_full_phase ^= 1; }
                            if (pair_half == 0) {
                                uint32_t _mbar_token_1 = mbarrier_try_wait(ab_full_addr + (ab_stage) * 8, ab_full_phase);
                                ab_full_token = _mbar_token_1;
                            } else {
                                ab_full_token = 1;
                                if (k_pairs > pair_k + 1) {
                                    uint32_t _mbar_token_2 = mbarrier_try_wait(ab_full_addr + (ab_stage) * 8, ab_full_phase);
                                    ab_full_token = _mbar_token_2;
                                }
                            }
                        }
                        elect_commit(partial_full_addr + (partial_pair_stage_1) * 8);
                        partial_pair_stage_1 += 1;
                        if (partial_pair_stage_1 == 2) { partial_pair_stage_1 = 0; partial_free_phase ^= 1; }
                    }
                    run_begin_1 = run_end_1;
                }
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
        }
    }
    // ---- Role: tma_role ----
    if (warp == 9) {
        { // tma_role_main
            unsigned int ab_stage_1 = 0;
            int m_tiles_2 = (M + 128 - 1) / 128;
            int n_tiles_2 = N / 128;
            int total_tiles_2 = m_tiles_2 * n_tiles_2;
            int m_tile_2 = bid / n_tiles_2;
            int n_tile_1 = bid % n_tiles_2;
            int dm_tile_1 = num_bids / n_tiles_2;
            int dn_tile_1 = num_bids % n_tiles_2;
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (int tile_id_1 = bid; tile_id_1 < total_tiles_2; tile_id_1 += num_bids) {
                int tile_rows_2 = 128;
                if (m_tile_2 * 128 + tile_rows_2 > M) {
                    tile_rows_2 = M - m_tile_2 * 128;
                }
                int last_group_2 = 0;
                if (lane == 0) {
                    last_group_2 = m_indices[m_tile_2 * 128 + tile_rows_2 - 1];
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
                        group_2 = m_indices[m_tile_2 * 128 + run_begin_2];
                        if (group_2 != last_group_2) {
                            #pragma unroll 1
                            for (int probe_2 = run_begin_2 + 1; probe_2 < tile_rows_2; probe_2++) {
                                int next_group_2 = m_indices[m_tile_2 * 128 + probe_2];
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
                    int k_blocks_2 = K / 128;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < k_blocks_2; iter_k++) {
                        mbarrier_wait(ab_free_addr + (ab_stage_1) * 8, _phase_ab_free);
                        if (elect_sync()) {
                            tma_3d_gmem2smem(smem_a_addr + ab_stage_1 * 16384, A, 0, m_tile_2 * 128, iter_k, ab_full_addr + (ab_stage_1) * 8);
                        }
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(ab_full_addr + (ab_stage_1) * 8, 32768);
                        }
                        ab_stage_1 += 1;
                        if (ab_stage_1 == 6) { ab_stage_1 = 0; _phase_ab_free ^= 1; }
                    }
                    run_begin_2 = run_end_2;
                }
                m_tile_2 += dm_tile_1;
                n_tile_1 += dn_tile_1;
                if (n_tile_1 >= n_tiles_2) {
                    n_tile_1 -= n_tiles_2;
                    m_tile_2 += 1;
                }
            }
        }
    }
    // ---- Role: scale_role ----
    if (warp == 10) {
        { // scale_role_main
            unsigned int scale_stage_1 = 0;
            int m_tiles_3 = (M + 128 - 1) / 128;
            int n_tiles_3 = N / 128;
            int total_tiles_3 = m_tiles_3 * n_tiles_3;
            int m_tile_3 = bid / n_tiles_3;
            int n_tile_2 = bid % n_tiles_3;
            int dm_tile_2 = num_bids / n_tiles_3;
            int dn_tile_2 = num_bids % n_tiles_3;
            unsigned int _phase_scale_free = 1;
            #pragma unroll 1
            for (int tile_id_2 = bid; tile_id_2 < total_tiles_3; tile_id_2 += num_bids) {
                int tile_rows_3 = 128;
                if (m_tile_3 * 128 + tile_rows_3 > M) {
                    tile_rows_3 = M - m_tile_3 * 128;
                }
                int last_group_3 = 0;
                if (lane == 0) {
                    last_group_3 = m_indices[m_tile_3 * 128 + tile_rows_3 - 1];
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
                        group_3 = m_indices[m_tile_3 * 128 + run_begin_3];
                        if (group_3 != last_group_3) {
                            #pragma unroll 1
                            for (int probe_3 = run_begin_3 + 1; probe_3 < tile_rows_3; probe_3++) {
                                int next_group_3 = m_indices[m_tile_3 * 128 + probe_3];
                                if (next_group_3 != group_3) {
                                    run_end_3 = probe_3;
                                    break;
                                }
                            }
                        }
                    }
                    int _shfl_4 = __shfl_sync(0xFFFFFFFF, run_end_3, 0);
                    run_end_3 = _shfl_4;
                    int _shfl_5 = __shfl_sync(0xFFFFFFFF, group_3, 0);
                    group_3 = _shfl_5;
                    int k_blocks_3 = K / 128;
                    int scale_groups_1 = k_blocks_3 / 4;
                    int n_blocks = N / 128;
                    #pragma unroll 1
                    for (int scale_group_1 = 0; scale_group_1 < scale_groups_1; scale_group_1++) {
                        mbarrier_wait(scale_free_addr + (scale_stage_1) * 8, _phase_scale_free);
                        int stage_base = scale_stage_1 * 128 * 4;
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(scale_full_addr + (scale_stage_1) * 8, 2064);
                            tma_2d_gmem2smem(smem_ascale_addr + (unsigned int)(stage_base * 4), a_scale, scale_group_1 * 4, m_tile_3 * 128, scale_full_addr + (scale_stage_1) * 8);
                            tma_3d_gmem2smem(smem_bscale_addr + scale_stage_1 * 4 * 4, b_scale, scale_group_1 * 4, n_tile_2, group_3, scale_full_addr + (scale_stage_1) * 8);
                        }
                        _phase_scale_free ^= 1;
                    }
                    run_begin_3 = run_end_3;
                }
                m_tile_3 += dm_tile_2;
                n_tile_2 += dn_tile_2;
                if (n_tile_2 >= n_tiles_3) {
                    n_tile_2 -= n_tiles_3;
                    m_tile_3 += 1;
                }
            }
            #pragma unroll
            for (int _drain = 0; _drain < 1; _drain++) {
                mbarrier_wait(scale_free_addr + (scale_stage_1) * 8, _phase_scale_free);
                _phase_scale_free ^= 1;
            }
        }
    }
    // ---- Role: tma_b_role ----
    if (warp == 11) {
        { // tma_b_role_main
            unsigned int ab_stage_b = 0;
            int m_tiles_b = (M + 128 - 1) / 128;
            int n_tiles_b = N / 128;
            int total_tiles_b = m_tiles_b * n_tiles_b;
            int m_tile_b = bid / n_tiles_b;
            int n_tile_b = bid % n_tiles_b;
            int dm_tile_b = num_bids / n_tiles_b;
            int dn_tile_b = num_bids % n_tiles_b;
            unsigned int _phase_ab_free_1 = 1;
            #pragma unroll 1
            for (int tile_id_b = bid; tile_id_b < total_tiles_b; tile_id_b += num_bids) {
                int tile_rows_4 = 128;
                if (m_tile_b * 128 + tile_rows_4 > M) {
                    tile_rows_4 = M - m_tile_b * 128;
                }
                int last_group_4 = 0;
                if (lane == 0) {
                    last_group_4 = m_indices[m_tile_b * 128 + tile_rows_4 - 1];
                }
                int run_begin_4 = 0;
                #pragma unroll 1
                for (int segment_iter_4 = 0; segment_iter_4 < tile_rows_4; segment_iter_4++) {
                    if (run_begin_4 >= tile_rows_4) {
                        break;
                    }
                    int run_end_4 = tile_rows_4;
                    int group_b = 0;
                    if (lane == 0) {
                        group_b = m_indices[m_tile_b * 128 + run_begin_4];
                        if (group_b != last_group_4) {
                            #pragma unroll 1
                            for (int probe_4 = run_begin_4 + 1; probe_4 < tile_rows_4; probe_4++) {
                                int next_group_4 = m_indices[m_tile_b * 128 + probe_4];
                                if (next_group_4 != group_b) {
                                    run_end_4 = probe_4;
                                    break;
                                }
                            }
                        }
                    }
                    int _shfl_2 = __shfl_sync(0xFFFFFFFF, run_end_4, 0);
                    run_end_4 = _shfl_2;
                    int _shfl_3 = __shfl_sync(0xFFFFFFFF, group_b, 0);
                    group_b = _shfl_3;
                    int k_blocks_b = K / 128;
                    #pragma unroll 1
                    for (int iter_k_b = 0; iter_k_b < k_blocks_b; iter_k_b++) {
                        mbarrier_wait(ab_free_addr + (ab_stage_b) * 8, _phase_ab_free_1);
                        if (elect_sync()) {
                            tma_4d_gmem2smem(smem_b_addr + ab_stage_b * 16384, B, 0, n_tile_b * 128, iter_k_b, group_b, ab_full_addr + (ab_stage_b) * 8);
                        }
                        ab_stage_b += 1;
                        if (ab_stage_b == 6) { ab_stage_b = 0; _phase_ab_free_1 ^= 1; }
                    }
                    run_begin_4 = run_end_4;
                }
                m_tile_b += dm_tile_b;
                n_tile_b += dn_tile_b;
                if (n_tile_b >= n_tiles_b) {
                    n_tile_b -= n_tiles_b;
                    m_tile_b += 1;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
