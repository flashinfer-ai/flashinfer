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
static_assert(alignof(CUtensorMap) >= 64, "CUtensorMap CUDA ABI must be at least 64-byte aligned");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 248
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 8
#define TMEM_SFB_OFFSET 168
#define NUM_K_PIPE_STAGES 5
#define NUM_OUT_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 164864
#define SMEM_SMEM_B_STAGE_BYTES 2048
#define SMEM_SMEM_B_STRIDE 2048
#define SMEM_EPI_STAGING_OFF 175104
#define SMEM_EPI_STAGING_STAGE_BYTES 512
#define SMEM_EPI_STAGING_STRIDE 512
#define SMEM_SMEM_SFA_OFF 177152
#define SMEM_SMEM_SFA_STAGE_BYTES 4096
#define SMEM_SMEM_SFA_STRIDE 4096
#define SMEM_SMEM_SFB_OFF 197632
#define SMEM_SMEM_SFB_STAGE_BYTES 256
#define SMEM_SMEM_SFB_STRIDE 256
#define SMEM_TOTAL 198912
#define THREADS 512
#define BLOCK_M 128
#define BLOCK_N 8
#define BLOCK_K 512
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x4_wait(float* dst, int addr) {
    tmem_ld_x4(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_cake_warp_decode_7a7e68276dee79d7d8a0(const __grid_constant__ CUtensorMap A, uint8_t* __restrict__ B, const __grid_constant__ CUtensorMap SFA, uint8_t* __restrict__ SFB, const __grid_constant__ CUtensorMap C, uint8_t* __restrict__ SFC, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    #define b_full_addr (mbar_base + 40)
    #define sfa_full_addr (mbar_base + 80)
    #define sfb_full_addr (mbar_base + 120)
    #define sfa_free_addr (mbar_base + 160)
    #define sfb_free_addr (mbar_base + 200)
    #define tmem_sfa_full_addr (mbar_base + 240)
    #define tmem_sfb_full_addr (mbar_base + 280)
    #define k_done_addr (mbar_base + 320)
    #define mma_full_addr (mbar_base + 360)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int smem_b_addr = smem + 164864;
    uint8_t* epi_staging = reinterpret_cast<uint8_t*>(smem_raw + 175104);
    const int epi_staging_addr = smem + 175104;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 177152);
    const int smem_sfa_addr = smem + 177152;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_sfb_addr = smem + 197632;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 46 barriers)
    // Mbarriers at smem_raw[0..368)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // b_full: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // sfa_full: 5 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // sfb_full: 5 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // sfa_free: 5 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // sfb_free: 5 barriers, init_count=4
            mbarrier_init(smem + 200, 4);
            mbarrier_init(smem + 208, 4);
            mbarrier_init(smem + 216, 4);
            mbarrier_init(smem + 224, 4);
            mbarrier_init(smem + 232, 4);
            // tmem_sfa_full: 5 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // tmem_sfb_full: 5 barriers, init_count=1
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            // k_done: 5 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            // --- pipeline 'out_pipe' ---
            // mma_full: 1 barriers, init_count=1
            mbarrier_init(smem + 360, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 248 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 368);
    if (warp == 0) {
        int _tmem_hold = smem + 368;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sfa = taddr + 8;
    const int tmem_sfb = taddr + 168;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            const int warp_0 = warp;
            const int lane_1 = lane;
            int n_tile = blockIdx.y;
            int m_tile = blockIdx.x;
            int expert = tile_expert[n_tile];
            int valid_rows = tile_mn_limit[n_tile] - n_tile * BLOCK_N;
            float sc = scale_c[expert];
            float sg = scale_gate[expert];
            float cl = clamp_limit[expert];
            float al = act_alpha[expert];
            float be = act_beta[expert];
            float neg_cl = -cl;
            float beta_sg = be * sg;
            float alpha_sg = al * sg;
            float quant_pair[8] = {0};
            unsigned int _phase_mma_full_0 = 0;
            mbarrier_wait(mma_full_addr, _phase_mma_full_0);
            _phase_mma_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            float _tmem_load_1[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                : "r"(taddr + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int base_row = warp_0 * 16 + lane_1 / 4 * 2;
            asm volatile("cp.async.bulk.wait_group.read 0;");
            asm volatile("barrier.sync 7, 128;" ::: "memory");
            for (int token_group = 0; token_group < 1; token_group++) {
                int token0 = lane_1 % 4 * 2 + token_group * 8;
                int token1 = token0 + 1;
                float raw00 = _tmem_load_0[token_group * 4] * sg;
                float raw01 = _tmem_load_0[token_group * 4 + 1] * sg;
                float raw10 = _tmem_load_1[token_group * 4] * sg;
                float raw11 = _tmem_load_1[token_group * 4 + 1] * sg;
                float raw20 = _tmem_load_0[token_group * 4 + 2] * sg;
                float raw21 = _tmem_load_0[token_group * 4 + 3] * sg;
                float raw30 = _tmem_load_1[token_group * 4 + 2] * sg;
                float raw31 = _tmem_load_1[token_group * 4 + 3] * sg;
                float _expf_0 = __expf(-raw00);
                float value00 = sc * raw00 / (1.0f + _expf_0);
                float _expf_1 = __expf(-raw01);
                float value01 = sc * raw01 / (1.0f + _expf_1);
                float _expf_2 = __expf(-raw10);
                float value10 = sc * raw10 / (1.0f + _expf_2);
                float _expf_3 = __expf(-raw11);
                float value11 = sc * raw11 / (1.0f + _expf_3);
                float _expf_4 = __expf(-raw20);
                float value20 = sc * raw20 / (1.0f + _expf_4);
                float _expf_5 = __expf(-raw21);
                float value21 = sc * raw21 / (1.0f + _expf_5);
                float _expf_6 = __expf(-raw30);
                float value30 = sc * raw30 / (1.0f + _expf_6);
                float _expf_7 = __expf(-raw31);
                float value31 = sc * raw31 / (1.0f + _expf_7);
                float _fabs_0 = fabsf(value00);
                float _fabs_1 = fabsf(value20);
                float _max_0 = max_noftz(_fabs_0, _fabs_1);
                float _fabs_2 = fabsf(value10);
                float _fabs_3 = fabsf(value30);
                float _max_1 = max_noftz(_fabs_2, _fabs_3);
                float _max_2 = max_noftz(_max_0, _max_1);
                float block_max0 = _max_2;
                float _fabs_4 = fabsf(value01);
                float _fabs_5 = fabsf(value21);
                float _max_3 = max_noftz(_fabs_4, _fabs_5);
                float _fabs_6 = fabsf(value11);
                float _fabs_7 = fabsf(value31);
                float _max_4 = max_noftz(_fabs_6, _fabs_7);
                float _max_5 = max_noftz(_max_3, _max_4);
                float block_max1 = _max_5;
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, block_max0, 4);
                float _max_6 = max_noftz(block_max0, _shfl_xor_0);
                block_max0 = _max_6;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, block_max1, 4);
                float _max_7 = max_noftz(block_max1, _shfl_xor_1);
                block_max1 = _max_7;
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, block_max0, 8);
                float _max_8 = max_noftz(block_max0, _shfl_xor_2);
                block_max0 = _max_8;
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, block_max1, 8);
                float _max_9 = max_noftz(block_max1, _shfl_xor_3);
                block_max1 = _max_9;
                float _fp8_rt_0;
                uint16_t _e4m3x2_0;
                uint32_t _f16x2_0;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_0) : "f"(0.0f), "f"(block_max0 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                uint16_t _fp8_h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_0));
                float scale0 = _fp8_rt_0;
                float _fp8_rt_1;
                uint16_t _e4m3x2_1;
                uint32_t _f16x2_1;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(block_max1 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_1));
                float scale1 = _fp8_rt_1;
                float inv_scale0 = 0.0f;
                float inv_scale1 = 0.0f;
                if (scale0 != 0.0f) {
                    inv_scale0 = 1.0f / scale0;
                }
                if (scale1 != 0.0f) {
                    inv_scale1 = 1.0f / scale1;
                }
                quant_pair[0] = value00 * inv_scale0;
                quant_pair[1] = value20 * inv_scale0;
                uint32_t _slice_lo_mask_0;
                {
                    int _lim_2 = 2;
                    if (_lim_2 <= 0) { _slice_lo_mask_0 = 0u; }
                    else if (_lim_2 >= 8) { _slice_lo_mask_0 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_2));
                    }
                }
                uint32_t _slice_hi_mask_0;
                {
                    int _lim_3 = 8;
                    if (_lim_3 <= 0) { _slice_hi_mask_0 = 0u; }
                    else if (_lim_3 >= 8) { _slice_hi_mask_0 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_0) : "r"(_lim_3));
                    }
                }
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_0[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01 * inv_scale1;
                quant_pair[1] = value21 * inv_scale1;
                uint32_t _fp4_1[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value10 * inv_scale0;
                quant_pair[1] = value30 * inv_scale0;
                uint32_t _slice_lo_mask_1;
                {
                    int _lim_4 = 2;
                    if (_lim_4 <= 0) { _slice_lo_mask_1 = 0u; }
                    else if (_lim_4 >= 8) { _slice_lo_mask_1 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_4));
                    }
                }
                uint32_t _slice_hi_mask_1;
                {
                    int _lim_5 = 8;
                    if (_lim_5 <= 0) { _slice_hi_mask_1 = 0u; }
                    else if (_lim_5 >= 8) { _slice_hi_mask_1 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_1) : "r"(_lim_5));
                    }
                }
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_2[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value11 * inv_scale1;
                quant_pair[1] = value31 * inv_scale1;
                uint32_t _fp4_3[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, scale0, 16);
                float scale0_upper = _shfl_xor_4;
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, scale1, 16);
                float scale1_upper = _shfl_xor_5;
                if (lane_1 < 4) {
                    int sf_feature = m_tile * 8 + warp_0 * 2;
                    int sf_token_group = token0 / 8;
                    int sf_tile_stride = M_out / 64 * 32;
                    int sf_base = n_tile * sf_tile_stride + sf_token_group * (M_out / 64) * 32 + sf_feature / 4 * 32 + token0 % 8 * 4 + sf_feature % 4;
                    int sf_base_upper = n_tile * sf_tile_stride + sf_token_group * (M_out / 64) * 32 + (sf_feature + 1) / 4 * 32 + token0 % 8 * 4 + (sf_feature + 1) % 4;
                    if (token0 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_upper));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_upper) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base + 4)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_upper));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_upper + 4)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_section = warp_0 / 2 * 8 * 32;
                int smem_feature = warp_0 % 2 * 32 + lane_1 / 4 * 4;
                int smem_flat0 = token0 * 64 + smem_feature;
                int smem_flat1 = token1 * 64 + smem_feature;
                int smem_index0 = smem_flat0 / 2 ^ smem_flat0 / 256 % 2 * 16;
                int smem_index1 = smem_flat1 / 2 ^ smem_flat1 / 256 % 2 * 16;
                epi_staging[smem_section + smem_index0] = _fp4_0[0];
                epi_staging[smem_section + smem_index0 + 1] = _fp4_2[0];
                epi_staging[smem_section + smem_index1] = _fp4_1[0];
                epi_staging[smem_section + smem_index1 + 1] = _fp4_3[0];
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 7, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    int padding_rows = (8 - valid_rows % 8) % 8;
                    tma_store_4d((&C), m_tile * ((1) ? 128 : 64), padding_rows, 1073741824, n_tile * 8 - padding_rows + 1073741824, epi_staging_addr);
                    tma_store_4d((&C), m_tile * 128 + 64, padding_rows, 1073741824, n_tile * 8 - padding_rows + 1073741824, epi_staging_addr + 256);
                }
            }
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("barrier.sync 7, 128;" ::: "memory");
        }
    }
    // ---- Role: copy_sfb ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // copy_sfb_main
            unsigned int stage = 0;
            const int lane_0 = lane;
            unsigned int word[1];
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_k_done = 1;
            #pragma unroll 1
            for (int _iter_k = 0; _iter_k < K_tiles; _iter_k++) {
                mbarrier_wait(sfb_full_addr + (stage) * 8, _phase_sfb_full);
                mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int q = 0; q < 8; q++) {
                    word[0] = 0;
                    if (lane_0 < 8) {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[0])) : "r"(smem_sfb_addr + stage * 256 + (unsigned int)(lane_0 / 8 * 256) + (unsigned int)(q * 32) + (unsigned int)(lane_0 % 8 * 4)));
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 8 + 160 + stage * 16 + (unsigned int)(q * 2)), "r"(word[0]));
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        mbarrier_arrive(tmem_sfb_full_addr + (stage) * 8);
                    }
                }
                if (elect_sync()) {
                    mbarrier_arrive(sfb_free_addr + (stage) * 8);
                }
                stage += 1;
                if (stage == 5) { stage = 0; _phase_sfb_full ^= 1; _phase_k_done ^= 1; }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp >= 8 && warp <= 9) {
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_1 = 0;
            int n_tile_1 = blockIdx.y;
            int local_thread = (warp - 8) * 32 + lane;
            int valid_rows_1 = tile_mn_limit[n_tile_1] - n_tile_1 * BLOCK_N;
            int row_stride_bytes = K / 2;
            unsigned int _phase_k_done_1 = 1;
            #pragma unroll 1
            for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
                int dst_base = smem_b_addr + stage_1 * 2048;
                for (int row_group = 0; row_group < 1; row_group++) {
                    int elt_offset = local_thread * 32 + row_group * 2048;
                    int row = elt_offset / 256;
                    int col = elt_offset % 256;
                    int routed = route_map[n_tile_1 * 8 + row];
                    int src_base = routed * row_stride_bytes + iter_k * (BLOCK_K / 2) + col / 2;
                    int dst_chunk = elt_offset / 2 ^ row % 8 * 16;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                        "}"
                        :: "r"((row < valid_rows_1) ? 1 : 0), "r"(dst_base + dst_chunk), "l"(B + src_base));
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                        "}"
                        :: "r"((row < valid_rows_1) ? 1 : 0), "r"(dst_base + 1024 + dst_chunk), "l"(B + (src_base + 128)));
                }
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(b_full_addr + (stage_1) * 8) : "memory");
                asm volatile("barrier.sync 8, 64;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(b_full_addr + (stage_1) * 8);
                    }
                }
                stage_1 += 1;
                if (stage_1 == 5) { stage_1 = 0; _phase_k_done_1 ^= 1; }
            }
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 10) {
        { // load_sfb_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_2 = 0;
            int n_tile_2 = blockIdx.y;
            const int lane_0_1 = lane;
            int block4 = lane_0_1 % 8;
            int row0 = lane_0_1 / 8;
            int valid_rows_2 = tile_mn_limit[n_tile_2] - n_tile_2 * BLOCK_N;
            int sf_stride = K / 16;
            unsigned int _phase_sfb_free = 1;
            #pragma unroll 1
            for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                mbarrier_wait(sfb_free_addr + (stage_2) * 8, _phase_sfb_free);
                int sf_col = iter_k_1 * (BLOCK_K / 16) + block4 * 4;
                for (int row_group_1 = 0; row_group_1 < 2; row_group_1++) {
                    int row_1 = row0 + row_group_1 * 4;
                    int routed_1 = route_map[n_tile_2 * 8 + row_1];
                    int dst_offset = row_1 / 8 * 256 + block4 * 32 + row_1 % 8 * 4;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                        "}"
                        :: "r"((row_1 < valid_rows_2) ? 1 : 0), "r"(smem_sfb_addr + stage_2 * 256 + (unsigned int)dst_offset), "l"(SFB + (routed_1 * sf_stride + sf_col)));
                }
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(sfb_full_addr + (stage_2) * 8) : "memory");
                asm volatile("barrier.sync 9, 32;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(sfb_full_addr + (stage_2) * 8);
                }
                stage_2 += 1;
                if (stage_2 == 5) { stage_2 = 0; _phase_sfb_free ^= 1; }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 11) {
        { // load_a_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_3 = 0;
            int m_tile_1 = blockIdx.x;
            int n_tile_3 = blockIdx.y;
            int expert_1 = tile_expert[n_tile_3];
            unsigned int _phase_k_done_2 = 1;
            #pragma unroll 1
            for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                mbarrier_wait(k_done_addr + (stage_3) * 8, _phase_k_done_2);
                int sf_unused = expert_1;
                if (elect_sync()) {
                    tma_4d_gmem2smem(smem_a_addr + stage_3 * 32768, (&A), 0, m_tile_1 * 128, iter_k_2 * 2, sf_unused, a_full_addr + (stage_3) * 8);
                    tma_4d_gmem2smem(smem_a_addr + stage_3 * 32768 + 16384, (&A), 0, m_tile_1 * 128, iter_k_2 * 2 + 1, sf_unused, a_full_addr + (stage_3) * 8);
                    mbarrier_arrive_expect_tx(a_full_addr + (stage_3) * 8, 32768);
                }
                stage_3 += 1;
                if (stage_3 == 5) { stage_3 = 0; _phase_k_done_2 ^= 1; }
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 12) {
        { // load_sfa_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_4 = 0;
            int m_tile_2 = blockIdx.x;
            int n_tile_4 = blockIdx.y;
            int expert_2 = tile_expert[n_tile_4];
            unsigned int _phase_sfa_free = 1;
            #pragma unroll 1
            for (int iter_k_3 = 0; iter_k_3 < K_tiles; iter_k_3++) {
                mbarrier_wait(sfa_free_addr + (stage_4) * 8, _phase_sfa_free);
                int sf_tile = (expert_2 * grid_m + m_tile_2) * K_tiles + iter_k_3;
                if (elect_sync()) {
                    tma_4d_gmem2smem(smem_sfa_addr + stage_4 * 4096, (&SFA), 0, 0, iter_k_3 * 8, expert_2 * grid_m + m_tile_2, sfa_full_addr + (stage_4) * 8);
                    mbarrier_arrive_expect_tx(sfa_full_addr + (stage_4) * 8, 4096);
                }
                stage_4 += 1;
                if (stage_4 == 5) { stage_4 = 0; _phase_sfa_free ^= 1; }
            }
        }
    }
    // ---- Role: copy_sfa ----
    if (warp == 13) {
        { // copy_sfa_main
            unsigned int stage_5 = 0;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_k_done_3 = 1;
            #pragma unroll 1
            for (int _iter_k_1 = 0; _iter_k_1 < K_tiles; _iter_k_1++) {
                mbarrier_wait(sfa_full_addr + (stage_5) * 8, _phase_sfa_full);
                mbarrier_wait(k_done_addr + (stage_5) * 8, _phase_k_done_3);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_5 * 32)), "l"(_tcgen05_cp_desc_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 4))), "l"(_tcgen05_cp_desc_1)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 8))), "l"(_tcgen05_cp_desc_2)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 12))), "l"(_tcgen05_cp_desc_3)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 16))), "l"(_tcgen05_cp_desc_4)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 2560)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 20))), "l"(_tcgen05_cp_desc_5)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 3072)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 24))), "l"(_tcgen05_cp_desc_6)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 3584)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 28))), "l"(_tcgen05_cp_desc_7)
                            : "memory");
                    }
                }
                elect_commit2(tmem_sfa_full_addr + (stage_5) * 8, sfa_free_addr + (stage_5) * 8);
                stage_5 += 1;
                if (stage_5 == 5) { stage_5 = 0; _phase_sfa_full ^= 1; _phase_k_done_3 ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 14) {
        { // mma_main
            unsigned int stage_6 = 0;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_tmem_sfa_full = 0;
            unsigned int _phase_tmem_sfb_full = 0;
            {
                #pragma unroll 1
                for (int iter_k_4 = 0; iter_k_4 < K_tiles; iter_k_4++) {
                    mbarrier_wait(a_full_addr + (stage_6) * 8, _phase_a_full);
                    mbarrier_wait(b_full_addr + (stage_6) * 8, _phase_b_full);
                    mbarrier_wait(tmem_sfa_full_addr + (stage_6) * 8, _phase_tmem_sfa_full);
                    mbarrier_wait(tmem_sfb_full_addr + (stage_6) * 8, _phase_tmem_sfb_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_8 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_8 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_8) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_8) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + stage_6 * 32 + 0, (unsigned int)tmem_sfb + stage_6 * 16 + 0, ((((1) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_9 = make_warp_uniform((((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_9 = make_warp_uniform((((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_9) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_9) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 4) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 2) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_10 = make_warp_uniform((((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_10 = make_warp_uniform((((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_10) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_10) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 8) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 4) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_11 = make_warp_uniform((((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_11 = make_warp_uniform((((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_11) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_11) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 12) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 6) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_12 = make_warp_uniform((((smem_a_addr + 16384) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_12 = make_warp_uniform((((smem_b_addr + 1024) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_12) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_12) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 16) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 8) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_13 = make_warp_uniform((((smem_a_addr + 16416) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_13 = make_warp_uniform((((smem_b_addr + 1056) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_13) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_13) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 20) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 10) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_14 = make_warp_uniform((((smem_a_addr + 16448) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_14 = make_warp_uniform((((smem_b_addr + 1088) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_14) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_14) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 24) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 12) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_15 = make_warp_uniform((((smem_a_addr + 16480) >> 4) & 0x3FFF) + (stage_6) * 2048);
                    int _mma_b_lo_15 = make_warp_uniform((((smem_b_addr + 1120) >> 4) & 0x3FFF) + (stage_6) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_15) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_15) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 28) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 14) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                        }
                    }
                    if (iter_k_4 + 1 == K_tiles) {
                        elect_commit2(k_done_addr + (stage_6) * 8, mma_full_addr);
                    } else {
                        elect_commit(k_done_addr + (stage_6) * 8);
                    }
                    stage_6 += 1;
                    if (stage_6 == 5) { stage_6 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; _phase_tmem_sfa_full ^= 1; _phase_tmem_sfb_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 15) {
        // idle — no tasks assigned
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
