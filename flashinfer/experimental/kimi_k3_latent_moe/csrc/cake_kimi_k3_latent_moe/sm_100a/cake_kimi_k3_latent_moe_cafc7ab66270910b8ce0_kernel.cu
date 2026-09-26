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
#define TMEM_NCOLS 256
#define TMEM_ACC_OFFSET 0
#define NUM_TMA_PIPE_STAGES 8
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_LAND_OFF 2048
#define SMEM_LAND_STAGE_BYTES 32768
#define SMEM_LAND_STRIDE 32768
#define SMEM_SMEM_A_OFF 34816
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 165888
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_SMEM_BN_OFF 231424
#define SMEM_SMEM_BN_STAGE_BYTES 8192
#define SMEM_SMEM_BN_STRIDE 8192
#define SMEM_SMEM_AG_OFF 34816
#define SMEM_SMEM_AG_STAGE_BYTES 8192
#define SMEM_SMEM_AG_STRIDE 16384
#define SMEM_SMEM_AU_OFF 43008
#define SMEM_SMEM_AU_STAGE_BYTES 8192
#define SMEM_SMEM_AU_STRIDE 16384
#define SMEM_TOTAL 231424

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


__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_latent_moe_cafc7ab66270910b8ce0(const __grid_constant__ CUtensorMap A_R, const __grid_constant__ CUtensorMap A_L, const __grid_constant__ CUtensorMap A_S, const __grid_constant__ CUtensorMap A_2, const __grid_constant__ CUtensorMap B_1, const __grid_constant__ CUtensorMap B_2, float* __restrict__ out_r, __nv_bfloat16* __restrict__ out_l, __nv_bfloat16* __restrict__ out_s, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out, unsigned long long* __restrict__ tl, int num_tokens, int k1_off, int num_partials, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define a_full_addr (mbar_base + 0)
    #define a_empty_addr (mbar_base + 64)
    #define acc_full_addr (mbar_base + 128)
    #define acc_empty_addr (mbar_base + 136)
    #define red_bar_addr (mbar_base + 144)
    #define issue_bar_addr (mbar_base + 152)
    #define rows_bar_addr (mbar_base + 160)
    #define bn_bar_addr (mbar_base + 168)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* land = reinterpret_cast<float*>(smem_raw + 2048);
    const int land_addr = smem + 2048;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 34816);
    const int smem_a_addr = smem + 34816;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 165888);
    const int smem_b_addr = smem + 165888;
    __nv_bfloat16* smem_bn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 231424);
    const int smem_bn_addr = smem + 231424;
    __nv_bfloat16* smem_ag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 34816);
    const int smem_ag_addr = smem + 34816;
    __nv_bfloat16* smem_au = reinterpret_cast<__nv_bfloat16*>(smem_raw + 43008);
    const int smem_au_addr = smem + 43008;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 22 barriers)
    // Mbarriers at smem_raw[0..176)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // a_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // a_empty: 8 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // --- pipeline 'epi_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            // acc_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            // red_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // issue_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            // rows_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // bn_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 144, 32768);
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 176);
    if (warp == 0) {
        int _tmem_hold = smem + 176;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int g_e = bid;
            int epi_warp = warp % 4;
            int tid_1 = epi_warp * 32 + lane;
            int tl_e = g_e * 16;
            int tile = g_e;
            int slot = 0;
            {
                tile = g_e / 2;
                slot = g_e % 2;
            }
            int cls = ((tile < 7) ? 0 : ((tile < 35) ? 1 : 2));
            int row_r = tile * 128;
            int row_l = (tile - 7) * 128;
            int row_s = (tile - 7 - 28) * 64;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            unsigned int epi_stage = 0;
            float red[64];
            int fin = 1;
            unsigned int _phase_acc_full = 0;
            mbarrier_wait(acc_full_addr + (epi_stage) * 8, _phase_acc_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (cls == 2) {
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                    : "r"(taddr + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                    : "r"(taddr + 128));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                red[0] = _tmem_load_0[0];
                red[32] = _tmem_load_1[0];
                red[1] = _tmem_load_0[1];
                red[33] = _tmem_load_1[1];
                red[2] = _tmem_load_0[2];
                red[34] = _tmem_load_1[2];
                red[3] = _tmem_load_0[3];
                red[35] = _tmem_load_1[3];
                red[4] = _tmem_load_0[4];
                red[36] = _tmem_load_1[4];
                red[5] = _tmem_load_0[5];
                red[37] = _tmem_load_1[5];
                red[6] = _tmem_load_0[6];
                red[38] = _tmem_load_1[6];
                red[7] = _tmem_load_0[7];
                red[39] = _tmem_load_1[7];
                red[8] = _tmem_load_0[8];
                red[40] = _tmem_load_1[8];
                red[9] = _tmem_load_0[9];
                red[41] = _tmem_load_1[9];
                red[10] = _tmem_load_0[10];
                red[42] = _tmem_load_1[10];
                red[11] = _tmem_load_0[11];
                red[43] = _tmem_load_1[11];
                red[12] = _tmem_load_0[12];
                red[44] = _tmem_load_1[12];
                red[13] = _tmem_load_0[13];
                red[45] = _tmem_load_1[13];
                red[14] = _tmem_load_0[14];
                red[46] = _tmem_load_1[14];
                red[15] = _tmem_load_0[15];
                red[47] = _tmem_load_1[15];
                red[16] = _tmem_load_0[16];
                red[48] = _tmem_load_1[16];
                red[17] = _tmem_load_0[17];
                red[49] = _tmem_load_1[17];
                red[18] = _tmem_load_0[18];
                red[50] = _tmem_load_1[18];
                red[19] = _tmem_load_0[19];
                red[51] = _tmem_load_1[19];
                red[20] = _tmem_load_0[20];
                red[52] = _tmem_load_1[20];
                red[21] = _tmem_load_0[21];
                red[53] = _tmem_load_1[21];
                red[22] = _tmem_load_0[22];
                red[54] = _tmem_load_1[22];
                red[23] = _tmem_load_0[23];
                red[55] = _tmem_load_1[23];
                red[24] = _tmem_load_0[24];
                red[56] = _tmem_load_1[24];
                red[25] = _tmem_load_0[25];
                red[57] = _tmem_load_1[25];
                red[26] = _tmem_load_0[26];
                red[58] = _tmem_load_1[26];
                red[27] = _tmem_load_0[27];
                red[59] = _tmem_load_1[27];
                red[28] = _tmem_load_0[28];
                red[60] = _tmem_load_1[28];
                red[29] = _tmem_load_0[29];
                red[61] = _tmem_load_1[29];
                red[30] = _tmem_load_0[30];
                red[62] = _tmem_load_1[30];
                red[31] = _tmem_load_0[31];
                red[63] = _tmem_load_1[31];
            } else {
                float _tmem_load_2[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(lane_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                    : "r"(lane_addr + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                red[0] = _tmem_load_2[0];
                red[1] = _tmem_load_2[1];
                red[2] = _tmem_load_2[2];
                red[3] = _tmem_load_2[3];
                red[4] = _tmem_load_2[4];
                red[5] = _tmem_load_2[5];
                red[6] = _tmem_load_2[6];
                red[7] = _tmem_load_2[7];
                red[8] = _tmem_load_2[8];
                red[9] = _tmem_load_2[9];
                red[10] = _tmem_load_2[10];
                red[11] = _tmem_load_2[11];
                red[12] = _tmem_load_2[12];
                red[13] = _tmem_load_2[13];
                red[14] = _tmem_load_2[14];
                red[15] = _tmem_load_2[15];
                red[16] = _tmem_load_2[16];
                red[17] = _tmem_load_2[17];
                red[18] = _tmem_load_2[18];
                red[19] = _tmem_load_2[19];
                red[20] = _tmem_load_2[20];
                red[21] = _tmem_load_2[21];
                red[22] = _tmem_load_2[22];
                red[23] = _tmem_load_2[23];
                red[24] = _tmem_load_2[24];
                red[25] = _tmem_load_2[25];
                red[26] = _tmem_load_2[26];
                red[27] = _tmem_load_2[27];
                red[28] = _tmem_load_2[28];
                red[29] = _tmem_load_2[29];
                red[30] = _tmem_load_2[30];
                red[31] = _tmem_load_2[31];
                red[32] = _tmem_load_2[32];
                red[33] = _tmem_load_2[33];
                red[34] = _tmem_load_2[34];
                red[35] = _tmem_load_2[35];
                red[36] = _tmem_load_2[36];
                red[37] = _tmem_load_2[37];
                red[38] = _tmem_load_2[38];
                red[39] = _tmem_load_2[39];
                red[40] = _tmem_load_2[40];
                red[41] = _tmem_load_2[41];
                red[42] = _tmem_load_2[42];
                red[43] = _tmem_load_2[43];
                red[44] = _tmem_load_2[44];
                red[45] = _tmem_load_2[45];
                red[46] = _tmem_load_2[46];
                red[47] = _tmem_load_2[47];
                red[48] = _tmem_load_2[48];
                red[49] = _tmem_load_2[49];
                red[50] = _tmem_load_2[50];
                red[51] = _tmem_load_2[51];
                red[52] = _tmem_load_2[52];
                red[53] = _tmem_load_2[53];
                red[54] = _tmem_load_2[54];
                red[55] = _tmem_load_2[55];
                red[56] = _tmem_load_2[56];
                red[57] = _tmem_load_2[57];
                red[58] = _tmem_load_2[58];
                red[59] = _tmem_load_2[59];
                red[60] = _tmem_load_2[60];
                red[61] = _tmem_load_2[61];
                red[62] = _tmem_load_2[62];
                red[63] = _tmem_load_2[63];
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(acc_empty_addr + (epi_stage) * 8);
                }
            }
            if (slot == 1) {
                uint32_t _mapa_0;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_0) : "r"(red_bar_addr), "r"(0));
                uint32_t _mapa_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_1) : "r"(land_addr + (unsigned int)(tid_1 * 16)), "r"(0));
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1), "r"(__float_as_uint(red[0])), "r"(__float_as_uint(red[1])), "r"(__float_as_uint(red[2])), "r"(__float_as_uint(red[3])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 2048), "r"(__float_as_uint(red[4])), "r"(__float_as_uint(red[5])), "r"(__float_as_uint(red[6])), "r"(__float_as_uint(red[7])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 4096), "r"(__float_as_uint(red[8])), "r"(__float_as_uint(red[9])), "r"(__float_as_uint(red[10])), "r"(__float_as_uint(red[11])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 6144), "r"(__float_as_uint(red[12])), "r"(__float_as_uint(red[13])), "r"(__float_as_uint(red[14])), "r"(__float_as_uint(red[15])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 8192), "r"(__float_as_uint(red[16])), "r"(__float_as_uint(red[17])), "r"(__float_as_uint(red[18])), "r"(__float_as_uint(red[19])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 10240), "r"(__float_as_uint(red[20])), "r"(__float_as_uint(red[21])), "r"(__float_as_uint(red[22])), "r"(__float_as_uint(red[23])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 12288), "r"(__float_as_uint(red[24])), "r"(__float_as_uint(red[25])), "r"(__float_as_uint(red[26])), "r"(__float_as_uint(red[27])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 14336), "r"(__float_as_uint(red[28])), "r"(__float_as_uint(red[29])), "r"(__float_as_uint(red[30])), "r"(__float_as_uint(red[31])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 16384), "r"(__float_as_uint(red[32])), "r"(__float_as_uint(red[33])), "r"(__float_as_uint(red[34])), "r"(__float_as_uint(red[35])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 18432), "r"(__float_as_uint(red[36])), "r"(__float_as_uint(red[37])), "r"(__float_as_uint(red[38])), "r"(__float_as_uint(red[39])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 20480), "r"(__float_as_uint(red[40])), "r"(__float_as_uint(red[41])), "r"(__float_as_uint(red[42])), "r"(__float_as_uint(red[43])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 22528), "r"(__float_as_uint(red[44])), "r"(__float_as_uint(red[45])), "r"(__float_as_uint(red[46])), "r"(__float_as_uint(red[47])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 24576), "r"(__float_as_uint(red[48])), "r"(__float_as_uint(red[49])), "r"(__float_as_uint(red[50])), "r"(__float_as_uint(red[51])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 26624), "r"(__float_as_uint(red[52])), "r"(__float_as_uint(red[53])), "r"(__float_as_uint(red[54])), "r"(__float_as_uint(red[55])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 28672), "r"(__float_as_uint(red[56])), "r"(__float_as_uint(red[57])), "r"(__float_as_uint(red[58])), "r"(__float_as_uint(red[59])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 30720), "r"(__float_as_uint(red[60])), "r"(__float_as_uint(red[61])), "r"(__float_as_uint(red[62])), "r"(__float_as_uint(red[63])), "r"(_mapa_0) : "memory");
                fin = 0;
            } else {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(red_bar_addr);
                    }
                }
                mbarrier_wait_cluster_hint(red_bar_addr, 0, 10000000);
                red[0] = red[0] + land[tid_1 * 4];
                red[1] = red[1] + land[tid_1 * 4 + 1];
                red[2] = red[2] + land[tid_1 * 4 + 2];
                red[3] = red[3] + land[tid_1 * 4 + 3];
                red[4] = red[4] + land[(128 + tid_1) * 4];
                red[5] = red[5] + land[(128 + tid_1) * 4 + 1];
                red[6] = red[6] + land[(128 + tid_1) * 4 + 2];
                red[7] = red[7] + land[(128 + tid_1) * 4 + 3];
                red[8] = red[8] + land[(256 + tid_1) * 4];
                red[9] = red[9] + land[(256 + tid_1) * 4 + 1];
                red[10] = red[10] + land[(256 + tid_1) * 4 + 2];
                red[11] = red[11] + land[(256 + tid_1) * 4 + 3];
                red[12] = red[12] + land[(384 + tid_1) * 4];
                red[13] = red[13] + land[(384 + tid_1) * 4 + 1];
                red[14] = red[14] + land[(384 + tid_1) * 4 + 2];
                red[15] = red[15] + land[(384 + tid_1) * 4 + 3];
                red[16] = red[16] + land[(512 + tid_1) * 4];
                red[17] = red[17] + land[(512 + tid_1) * 4 + 1];
                red[18] = red[18] + land[(512 + tid_1) * 4 + 2];
                red[19] = red[19] + land[(512 + tid_1) * 4 + 3];
                red[20] = red[20] + land[(640 + tid_1) * 4];
                red[21] = red[21] + land[(640 + tid_1) * 4 + 1];
                red[22] = red[22] + land[(640 + tid_1) * 4 + 2];
                red[23] = red[23] + land[(640 + tid_1) * 4 + 3];
                red[24] = red[24] + land[(768 + tid_1) * 4];
                red[25] = red[25] + land[(768 + tid_1) * 4 + 1];
                red[26] = red[26] + land[(768 + tid_1) * 4 + 2];
                red[27] = red[27] + land[(768 + tid_1) * 4 + 3];
                red[28] = red[28] + land[(896 + tid_1) * 4];
                red[29] = red[29] + land[(896 + tid_1) * 4 + 1];
                red[30] = red[30] + land[(896 + tid_1) * 4 + 2];
                red[31] = red[31] + land[(896 + tid_1) * 4 + 3];
                red[32] = red[32] + land[(1024 + tid_1) * 4];
                red[33] = red[33] + land[(1024 + tid_1) * 4 + 1];
                red[34] = red[34] + land[(1024 + tid_1) * 4 + 2];
                red[35] = red[35] + land[(1024 + tid_1) * 4 + 3];
                red[36] = red[36] + land[(1152 + tid_1) * 4];
                red[37] = red[37] + land[(1152 + tid_1) * 4 + 1];
                red[38] = red[38] + land[(1152 + tid_1) * 4 + 2];
                red[39] = red[39] + land[(1152 + tid_1) * 4 + 3];
                red[40] = red[40] + land[(1280 + tid_1) * 4];
                red[41] = red[41] + land[(1280 + tid_1) * 4 + 1];
                red[42] = red[42] + land[(1280 + tid_1) * 4 + 2];
                red[43] = red[43] + land[(1280 + tid_1) * 4 + 3];
                red[44] = red[44] + land[(1408 + tid_1) * 4];
                red[45] = red[45] + land[(1408 + tid_1) * 4 + 1];
                red[46] = red[46] + land[(1408 + tid_1) * 4 + 2];
                red[47] = red[47] + land[(1408 + tid_1) * 4 + 3];
                red[48] = red[48] + land[(1536 + tid_1) * 4];
                red[49] = red[49] + land[(1536 + tid_1) * 4 + 1];
                red[50] = red[50] + land[(1536 + tid_1) * 4 + 2];
                red[51] = red[51] + land[(1536 + tid_1) * 4 + 3];
                red[52] = red[52] + land[(1664 + tid_1) * 4];
                red[53] = red[53] + land[(1664 + tid_1) * 4 + 1];
                red[54] = red[54] + land[(1664 + tid_1) * 4 + 2];
                red[55] = red[55] + land[(1664 + tid_1) * 4 + 3];
                red[56] = red[56] + land[(1792 + tid_1) * 4];
                red[57] = red[57] + land[(1792 + tid_1) * 4 + 1];
                red[58] = red[58] + land[(1792 + tid_1) * 4 + 2];
                red[59] = red[59] + land[(1792 + tid_1) * 4 + 3];
                red[60] = red[60] + land[(1920 + tid_1) * 4];
                red[61] = red[61] + land[(1920 + tid_1) * 4 + 1];
                red[62] = red[62] + land[(1920 + tid_1) * 4 + 2];
                red[63] = red[63] + land[(1920 + tid_1) * 4 + 3];
            }
            if (fin == 1) {
                if (cls == 2) {
                    int tok = lane_pair * 2;
                    int frow = row_base + ((0) ? 8 : 0);
                    if (tok < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_68 = __float2bfloat16(red[0]);
                        float _cvt_f32_68 = __bfloat162float(_cvt_bf16_68);
                        float gk = _cvt_f32_68;
                        __nv_bfloat16 _cvt_bf16_69 = __float2bfloat16(red[32]);
                        float _cvt_f32_69 = __bfloat162float(_cvt_bf16_69);
                        float uk = _cvt_f32_69;
                        float _exp2_0 = approx_exp2((-gk) * 1.4426950408889634f);
                        float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                        float sig = _rcp_0;
                        float _tanh_0 = tanhf(gk * 0.25f);
                        float aa = 4.0f * _tanh_0 * sig;
                        float _tanh_1 = tanhf(uk * 0.04f);
                        float bb = 25.0f * _tanh_1;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok * 768 + row_s + frow)) + (0)) = __float2bfloat16_rn(aa * bb);
                    }
                    int tok_0 = lane_pair * 2 + 1;
                    int frow_1 = row_base + ((0) ? 8 : 0);
                    if (tok_0 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_70 = __float2bfloat16(red[1]);
                        float _cvt_f32_70 = __bfloat162float(_cvt_bf16_70);
                        float gk_1 = _cvt_f32_70;
                        __nv_bfloat16 _cvt_bf16_71 = __float2bfloat16(red[33]);
                        float _cvt_f32_71 = __bfloat162float(_cvt_bf16_71);
                        float uk_1 = _cvt_f32_71;
                        float _exp2_1 = approx_exp2((-gk_1) * 1.4426950408889634f);
                        float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                        float sig_1 = _rcp_1;
                        float _tanh_2 = tanhf(gk_1 * 0.25f);
                        float aa_1 = 4.0f * _tanh_2 * sig_1;
                        float _tanh_3 = tanhf(uk_1 * 0.04f);
                        float bb_1 = 25.0f * _tanh_3;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_0 * 768 + row_s + frow_1)) + (0)) = __float2bfloat16_rn(aa_1 * bb_1);
                    }
                    int tok_2 = lane_pair * 2;
                    int frow_3 = row_base + ((1) ? 8 : 0);
                    if (tok_2 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_72 = __float2bfloat16(red[2]);
                        float _cvt_f32_72 = __bfloat162float(_cvt_bf16_72);
                        float gk_2 = _cvt_f32_72;
                        __nv_bfloat16 _cvt_bf16_73 = __float2bfloat16(red[34]);
                        float _cvt_f32_73 = __bfloat162float(_cvt_bf16_73);
                        float uk_2 = _cvt_f32_73;
                        float _exp2_2 = approx_exp2((-gk_2) * 1.4426950408889634f);
                        float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                        float sig_2 = _rcp_2;
                        float _tanh_4 = tanhf(gk_2 * 0.25f);
                        float aa_2 = 4.0f * _tanh_4 * sig_2;
                        float _tanh_5 = tanhf(uk_2 * 0.04f);
                        float bb_2 = 25.0f * _tanh_5;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_2 * 768 + row_s + frow_3)) + (0)) = __float2bfloat16_rn(aa_2 * bb_2);
                    }
                    int tok_4 = lane_pair * 2 + 1;
                    int frow_5 = row_base + ((1) ? 8 : 0);
                    if (tok_4 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_74 = __float2bfloat16(red[3]);
                        float _cvt_f32_74 = __bfloat162float(_cvt_bf16_74);
                        float gk_3 = _cvt_f32_74;
                        __nv_bfloat16 _cvt_bf16_75 = __float2bfloat16(red[35]);
                        float _cvt_f32_75 = __bfloat162float(_cvt_bf16_75);
                        float uk_3 = _cvt_f32_75;
                        float _exp2_3 = approx_exp2((-gk_3) * 1.4426950408889634f);
                        float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                        float sig_3 = _rcp_3;
                        float _tanh_6 = tanhf(gk_3 * 0.25f);
                        float aa_3 = 4.0f * _tanh_6 * sig_3;
                        float _tanh_7 = tanhf(uk_3 * 0.04f);
                        float bb_3 = 25.0f * _tanh_7;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_4 * 768 + row_s + frow_5)) + (0)) = __float2bfloat16_rn(aa_3 * bb_3);
                    }
                    int tok_6 = 8 + lane_pair * 2;
                    int frow_7 = row_base + ((0) ? 8 : 0);
                    if (tok_6 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_76 = __float2bfloat16(red[4]);
                        float _cvt_f32_76 = __bfloat162float(_cvt_bf16_76);
                        float gk_4 = _cvt_f32_76;
                        __nv_bfloat16 _cvt_bf16_77 = __float2bfloat16(red[36]);
                        float _cvt_f32_77 = __bfloat162float(_cvt_bf16_77);
                        float uk_4 = _cvt_f32_77;
                        float _exp2_4 = approx_exp2((-gk_4) * 1.4426950408889634f);
                        float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                        float sig_4 = _rcp_4;
                        float _tanh_8 = tanhf(gk_4 * 0.25f);
                        float aa_4 = 4.0f * _tanh_8 * sig_4;
                        float _tanh_9 = tanhf(uk_4 * 0.04f);
                        float bb_4 = 25.0f * _tanh_9;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_6 * 768 + row_s + frow_7)) + (0)) = __float2bfloat16_rn(aa_4 * bb_4);
                    }
                    int tok_8 = 8 + lane_pair * 2 + 1;
                    int frow_9 = row_base + ((0) ? 8 : 0);
                    if (tok_8 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_78 = __float2bfloat16(red[5]);
                        float _cvt_f32_78 = __bfloat162float(_cvt_bf16_78);
                        float gk_5 = _cvt_f32_78;
                        __nv_bfloat16 _cvt_bf16_79 = __float2bfloat16(red[37]);
                        float _cvt_f32_79 = __bfloat162float(_cvt_bf16_79);
                        float uk_5 = _cvt_f32_79;
                        float _exp2_5 = approx_exp2((-gk_5) * 1.4426950408889634f);
                        float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                        float sig_5 = _rcp_5;
                        float _tanh_10 = tanhf(gk_5 * 0.25f);
                        float aa_5 = 4.0f * _tanh_10 * sig_5;
                        float _tanh_11 = tanhf(uk_5 * 0.04f);
                        float bb_5 = 25.0f * _tanh_11;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_8 * 768 + row_s + frow_9)) + (0)) = __float2bfloat16_rn(aa_5 * bb_5);
                    }
                    int tok_10 = 8 + lane_pair * 2;
                    int frow_11 = row_base + ((1) ? 8 : 0);
                    if (tok_10 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_80 = __float2bfloat16(red[6]);
                        float _cvt_f32_80 = __bfloat162float(_cvt_bf16_80);
                        float gk_6 = _cvt_f32_80;
                        __nv_bfloat16 _cvt_bf16_81 = __float2bfloat16(red[38]);
                        float _cvt_f32_81 = __bfloat162float(_cvt_bf16_81);
                        float uk_6 = _cvt_f32_81;
                        float _exp2_6 = approx_exp2((-gk_6) * 1.4426950408889634f);
                        float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                        float sig_6 = _rcp_6;
                        float _tanh_12 = tanhf(gk_6 * 0.25f);
                        float aa_6 = 4.0f * _tanh_12 * sig_6;
                        float _tanh_13 = tanhf(uk_6 * 0.04f);
                        float bb_6 = 25.0f * _tanh_13;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_10 * 768 + row_s + frow_11)) + (0)) = __float2bfloat16_rn(aa_6 * bb_6);
                    }
                    int tok_12 = 8 + lane_pair * 2 + 1;
                    int frow_13 = row_base + ((1) ? 8 : 0);
                    if (tok_12 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_82 = __float2bfloat16(red[7]);
                        float _cvt_f32_82 = __bfloat162float(_cvt_bf16_82);
                        float gk_7 = _cvt_f32_82;
                        __nv_bfloat16 _cvt_bf16_83 = __float2bfloat16(red[39]);
                        float _cvt_f32_83 = __bfloat162float(_cvt_bf16_83);
                        float uk_7 = _cvt_f32_83;
                        float _exp2_7 = approx_exp2((-gk_7) * 1.4426950408889634f);
                        float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                        float sig_7 = _rcp_7;
                        float _tanh_14 = tanhf(gk_7 * 0.25f);
                        float aa_7 = 4.0f * _tanh_14 * sig_7;
                        float _tanh_15 = tanhf(uk_7 * 0.04f);
                        float bb_7 = 25.0f * _tanh_15;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_12 * 768 + row_s + frow_13)) + (0)) = __float2bfloat16_rn(aa_7 * bb_7);
                    }
                    int tok_14 = 16 + lane_pair * 2;
                    int frow_15 = row_base + ((0) ? 8 : 0);
                    if (tok_14 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_84 = __float2bfloat16(red[8]);
                        float _cvt_f32_84 = __bfloat162float(_cvt_bf16_84);
                        float gk_8 = _cvt_f32_84;
                        __nv_bfloat16 _cvt_bf16_85 = __float2bfloat16(red[40]);
                        float _cvt_f32_85 = __bfloat162float(_cvt_bf16_85);
                        float uk_8 = _cvt_f32_85;
                        float _exp2_8 = approx_exp2((-gk_8) * 1.4426950408889634f);
                        float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                        float sig_8 = _rcp_8;
                        float _tanh_16 = tanhf(gk_8 * 0.25f);
                        float aa_8 = 4.0f * _tanh_16 * sig_8;
                        float _tanh_17 = tanhf(uk_8 * 0.04f);
                        float bb_8 = 25.0f * _tanh_17;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_14 * 768 + row_s + frow_15)) + (0)) = __float2bfloat16_rn(aa_8 * bb_8);
                    }
                    int tok_16 = 16 + lane_pair * 2 + 1;
                    int frow_17 = row_base + ((0) ? 8 : 0);
                    if (tok_16 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_86 = __float2bfloat16(red[9]);
                        float _cvt_f32_86 = __bfloat162float(_cvt_bf16_86);
                        float gk_9 = _cvt_f32_86;
                        __nv_bfloat16 _cvt_bf16_87 = __float2bfloat16(red[41]);
                        float _cvt_f32_87 = __bfloat162float(_cvt_bf16_87);
                        float uk_9 = _cvt_f32_87;
                        float _exp2_9 = approx_exp2((-gk_9) * 1.4426950408889634f);
                        float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                        float sig_9 = _rcp_9;
                        float _tanh_18 = tanhf(gk_9 * 0.25f);
                        float aa_9 = 4.0f * _tanh_18 * sig_9;
                        float _tanh_19 = tanhf(uk_9 * 0.04f);
                        float bb_9 = 25.0f * _tanh_19;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_16 * 768 + row_s + frow_17)) + (0)) = __float2bfloat16_rn(aa_9 * bb_9);
                    }
                    int tok_18 = 16 + lane_pair * 2;
                    int frow_19 = row_base + ((1) ? 8 : 0);
                    if (tok_18 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_88 = __float2bfloat16(red[10]);
                        float _cvt_f32_88 = __bfloat162float(_cvt_bf16_88);
                        float gk_10 = _cvt_f32_88;
                        __nv_bfloat16 _cvt_bf16_89 = __float2bfloat16(red[42]);
                        float _cvt_f32_89 = __bfloat162float(_cvt_bf16_89);
                        float uk_10 = _cvt_f32_89;
                        float _exp2_10 = approx_exp2((-gk_10) * 1.4426950408889634f);
                        float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                        float sig_10 = _rcp_10;
                        float _tanh_20 = tanhf(gk_10 * 0.25f);
                        float aa_10 = 4.0f * _tanh_20 * sig_10;
                        float _tanh_21 = tanhf(uk_10 * 0.04f);
                        float bb_10 = 25.0f * _tanh_21;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_18 * 768 + row_s + frow_19)) + (0)) = __float2bfloat16_rn(aa_10 * bb_10);
                    }
                    int tok_20 = 16 + lane_pair * 2 + 1;
                    int frow_21 = row_base + ((1) ? 8 : 0);
                    if (tok_20 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_90 = __float2bfloat16(red[11]);
                        float _cvt_f32_90 = __bfloat162float(_cvt_bf16_90);
                        float gk_11 = _cvt_f32_90;
                        __nv_bfloat16 _cvt_bf16_91 = __float2bfloat16(red[43]);
                        float _cvt_f32_91 = __bfloat162float(_cvt_bf16_91);
                        float uk_11 = _cvt_f32_91;
                        float _exp2_11 = approx_exp2((-gk_11) * 1.4426950408889634f);
                        float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                        float sig_11 = _rcp_11;
                        float _tanh_22 = tanhf(gk_11 * 0.25f);
                        float aa_11 = 4.0f * _tanh_22 * sig_11;
                        float _tanh_23 = tanhf(uk_11 * 0.04f);
                        float bb_11 = 25.0f * _tanh_23;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_20 * 768 + row_s + frow_21)) + (0)) = __float2bfloat16_rn(aa_11 * bb_11);
                    }
                    int tok_22 = 24 + lane_pair * 2;
                    int frow_23 = row_base + ((0) ? 8 : 0);
                    if (tok_22 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_92 = __float2bfloat16(red[12]);
                        float _cvt_f32_92 = __bfloat162float(_cvt_bf16_92);
                        float gk_12 = _cvt_f32_92;
                        __nv_bfloat16 _cvt_bf16_93 = __float2bfloat16(red[44]);
                        float _cvt_f32_93 = __bfloat162float(_cvt_bf16_93);
                        float uk_12 = _cvt_f32_93;
                        float _exp2_12 = approx_exp2((-gk_12) * 1.4426950408889634f);
                        float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                        float sig_12 = _rcp_12;
                        float _tanh_24 = tanhf(gk_12 * 0.25f);
                        float aa_12 = 4.0f * _tanh_24 * sig_12;
                        float _tanh_25 = tanhf(uk_12 * 0.04f);
                        float bb_12 = 25.0f * _tanh_25;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_22 * 768 + row_s + frow_23)) + (0)) = __float2bfloat16_rn(aa_12 * bb_12);
                    }
                    int tok_24 = 24 + lane_pair * 2 + 1;
                    int frow_25 = row_base + ((0) ? 8 : 0);
                    if (tok_24 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_94 = __float2bfloat16(red[13]);
                        float _cvt_f32_94 = __bfloat162float(_cvt_bf16_94);
                        float gk_13 = _cvt_f32_94;
                        __nv_bfloat16 _cvt_bf16_95 = __float2bfloat16(red[45]);
                        float _cvt_f32_95 = __bfloat162float(_cvt_bf16_95);
                        float uk_13 = _cvt_f32_95;
                        float _exp2_13 = approx_exp2((-gk_13) * 1.4426950408889634f);
                        float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                        float sig_13 = _rcp_13;
                        float _tanh_26 = tanhf(gk_13 * 0.25f);
                        float aa_13 = 4.0f * _tanh_26 * sig_13;
                        float _tanh_27 = tanhf(uk_13 * 0.04f);
                        float bb_13 = 25.0f * _tanh_27;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_24 * 768 + row_s + frow_25)) + (0)) = __float2bfloat16_rn(aa_13 * bb_13);
                    }
                    int tok_26 = 24 + lane_pair * 2;
                    int frow_27 = row_base + ((1) ? 8 : 0);
                    if (tok_26 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_96 = __float2bfloat16(red[14]);
                        float _cvt_f32_96 = __bfloat162float(_cvt_bf16_96);
                        float gk_14 = _cvt_f32_96;
                        __nv_bfloat16 _cvt_bf16_97 = __float2bfloat16(red[46]);
                        float _cvt_f32_97 = __bfloat162float(_cvt_bf16_97);
                        float uk_14 = _cvt_f32_97;
                        float _exp2_14 = approx_exp2((-gk_14) * 1.4426950408889634f);
                        float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                        float sig_14 = _rcp_14;
                        float _tanh_28 = tanhf(gk_14 * 0.25f);
                        float aa_14 = 4.0f * _tanh_28 * sig_14;
                        float _tanh_29 = tanhf(uk_14 * 0.04f);
                        float bb_14 = 25.0f * _tanh_29;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_26 * 768 + row_s + frow_27)) + (0)) = __float2bfloat16_rn(aa_14 * bb_14);
                    }
                    int tok_28 = 24 + lane_pair * 2 + 1;
                    int frow_29 = row_base + ((1) ? 8 : 0);
                    if (tok_28 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_98 = __float2bfloat16(red[15]);
                        float _cvt_f32_98 = __bfloat162float(_cvt_bf16_98);
                        float gk_15 = _cvt_f32_98;
                        __nv_bfloat16 _cvt_bf16_99 = __float2bfloat16(red[47]);
                        float _cvt_f32_99 = __bfloat162float(_cvt_bf16_99);
                        float uk_15 = _cvt_f32_99;
                        float _exp2_15 = approx_exp2((-gk_15) * 1.4426950408889634f);
                        float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                        float sig_15 = _rcp_15;
                        float _tanh_30 = tanhf(gk_15 * 0.25f);
                        float aa_15 = 4.0f * _tanh_30 * sig_15;
                        float _tanh_31 = tanhf(uk_15 * 0.04f);
                        float bb_15 = 25.0f * _tanh_31;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_28 * 768 + row_s + frow_29)) + (0)) = __float2bfloat16_rn(aa_15 * bb_15);
                    }
                    int tok_30 = 32 + lane_pair * 2;
                    int frow_31 = row_base + ((0) ? 8 : 0);
                    if (tok_30 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_100 = __float2bfloat16(red[16]);
                        float _cvt_f32_100 = __bfloat162float(_cvt_bf16_100);
                        float gk_16 = _cvt_f32_100;
                        __nv_bfloat16 _cvt_bf16_101 = __float2bfloat16(red[48]);
                        float _cvt_f32_101 = __bfloat162float(_cvt_bf16_101);
                        float uk_16 = _cvt_f32_101;
                        float _exp2_16 = approx_exp2((-gk_16) * 1.4426950408889634f);
                        float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                        float sig_16 = _rcp_16;
                        float _tanh_32 = tanhf(gk_16 * 0.25f);
                        float aa_16 = 4.0f * _tanh_32 * sig_16;
                        float _tanh_33 = tanhf(uk_16 * 0.04f);
                        float bb_16 = 25.0f * _tanh_33;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_30 * 768 + row_s + frow_31)) + (0)) = __float2bfloat16_rn(aa_16 * bb_16);
                    }
                    int tok_32 = 32 + lane_pair * 2 + 1;
                    int frow_33 = row_base + ((0) ? 8 : 0);
                    if (tok_32 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_102 = __float2bfloat16(red[17]);
                        float _cvt_f32_102 = __bfloat162float(_cvt_bf16_102);
                        float gk_17 = _cvt_f32_102;
                        __nv_bfloat16 _cvt_bf16_103 = __float2bfloat16(red[49]);
                        float _cvt_f32_103 = __bfloat162float(_cvt_bf16_103);
                        float uk_17 = _cvt_f32_103;
                        float _exp2_17 = approx_exp2((-gk_17) * 1.4426950408889634f);
                        float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                        float sig_17 = _rcp_17;
                        float _tanh_34 = tanhf(gk_17 * 0.25f);
                        float aa_17 = 4.0f * _tanh_34 * sig_17;
                        float _tanh_35 = tanhf(uk_17 * 0.04f);
                        float bb_17 = 25.0f * _tanh_35;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_32 * 768 + row_s + frow_33)) + (0)) = __float2bfloat16_rn(aa_17 * bb_17);
                    }
                    int tok_34 = 32 + lane_pair * 2;
                    int frow_35 = row_base + ((1) ? 8 : 0);
                    if (tok_34 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_104 = __float2bfloat16(red[18]);
                        float _cvt_f32_104 = __bfloat162float(_cvt_bf16_104);
                        float gk_18 = _cvt_f32_104;
                        __nv_bfloat16 _cvt_bf16_105 = __float2bfloat16(red[50]);
                        float _cvt_f32_105 = __bfloat162float(_cvt_bf16_105);
                        float uk_18 = _cvt_f32_105;
                        float _exp2_18 = approx_exp2((-gk_18) * 1.4426950408889634f);
                        float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                        float sig_18 = _rcp_18;
                        float _tanh_36 = tanhf(gk_18 * 0.25f);
                        float aa_18 = 4.0f * _tanh_36 * sig_18;
                        float _tanh_37 = tanhf(uk_18 * 0.04f);
                        float bb_18 = 25.0f * _tanh_37;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_34 * 768 + row_s + frow_35)) + (0)) = __float2bfloat16_rn(aa_18 * bb_18);
                    }
                    int tok_36 = 32 + lane_pair * 2 + 1;
                    int frow_37 = row_base + ((1) ? 8 : 0);
                    if (tok_36 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_106 = __float2bfloat16(red[19]);
                        float _cvt_f32_106 = __bfloat162float(_cvt_bf16_106);
                        float gk_19 = _cvt_f32_106;
                        __nv_bfloat16 _cvt_bf16_107 = __float2bfloat16(red[51]);
                        float _cvt_f32_107 = __bfloat162float(_cvt_bf16_107);
                        float uk_19 = _cvt_f32_107;
                        float _exp2_19 = approx_exp2((-gk_19) * 1.4426950408889634f);
                        float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                        float sig_19 = _rcp_19;
                        float _tanh_38 = tanhf(gk_19 * 0.25f);
                        float aa_19 = 4.0f * _tanh_38 * sig_19;
                        float _tanh_39 = tanhf(uk_19 * 0.04f);
                        float bb_19 = 25.0f * _tanh_39;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_36 * 768 + row_s + frow_37)) + (0)) = __float2bfloat16_rn(aa_19 * bb_19);
                    }
                    int tok_38 = 40 + lane_pair * 2;
                    int frow_39 = row_base + ((0) ? 8 : 0);
                    if (tok_38 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_108 = __float2bfloat16(red[20]);
                        float _cvt_f32_108 = __bfloat162float(_cvt_bf16_108);
                        float gk_20 = _cvt_f32_108;
                        __nv_bfloat16 _cvt_bf16_109 = __float2bfloat16(red[52]);
                        float _cvt_f32_109 = __bfloat162float(_cvt_bf16_109);
                        float uk_20 = _cvt_f32_109;
                        float _exp2_20 = approx_exp2((-gk_20) * 1.4426950408889634f);
                        float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                        float sig_20 = _rcp_20;
                        float _tanh_40 = tanhf(gk_20 * 0.25f);
                        float aa_20 = 4.0f * _tanh_40 * sig_20;
                        float _tanh_41 = tanhf(uk_20 * 0.04f);
                        float bb_20 = 25.0f * _tanh_41;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_38 * 768 + row_s + frow_39)) + (0)) = __float2bfloat16_rn(aa_20 * bb_20);
                    }
                    int tok_40 = 40 + lane_pair * 2 + 1;
                    int frow_41 = row_base + ((0) ? 8 : 0);
                    if (tok_40 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_110 = __float2bfloat16(red[21]);
                        float _cvt_f32_110 = __bfloat162float(_cvt_bf16_110);
                        float gk_21 = _cvt_f32_110;
                        __nv_bfloat16 _cvt_bf16_111 = __float2bfloat16(red[53]);
                        float _cvt_f32_111 = __bfloat162float(_cvt_bf16_111);
                        float uk_21 = _cvt_f32_111;
                        float _exp2_21 = approx_exp2((-gk_21) * 1.4426950408889634f);
                        float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                        float sig_21 = _rcp_21;
                        float _tanh_42 = tanhf(gk_21 * 0.25f);
                        float aa_21 = 4.0f * _tanh_42 * sig_21;
                        float _tanh_43 = tanhf(uk_21 * 0.04f);
                        float bb_21 = 25.0f * _tanh_43;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_40 * 768 + row_s + frow_41)) + (0)) = __float2bfloat16_rn(aa_21 * bb_21);
                    }
                    int tok_42 = 40 + lane_pair * 2;
                    int frow_43 = row_base + ((1) ? 8 : 0);
                    if (tok_42 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_112 = __float2bfloat16(red[22]);
                        float _cvt_f32_112 = __bfloat162float(_cvt_bf16_112);
                        float gk_22 = _cvt_f32_112;
                        __nv_bfloat16 _cvt_bf16_113 = __float2bfloat16(red[54]);
                        float _cvt_f32_113 = __bfloat162float(_cvt_bf16_113);
                        float uk_22 = _cvt_f32_113;
                        float _exp2_22 = approx_exp2((-gk_22) * 1.4426950408889634f);
                        float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                        float sig_22 = _rcp_22;
                        float _tanh_44 = tanhf(gk_22 * 0.25f);
                        float aa_22 = 4.0f * _tanh_44 * sig_22;
                        float _tanh_45 = tanhf(uk_22 * 0.04f);
                        float bb_22 = 25.0f * _tanh_45;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_42 * 768 + row_s + frow_43)) + (0)) = __float2bfloat16_rn(aa_22 * bb_22);
                    }
                    int tok_44 = 40 + lane_pair * 2 + 1;
                    int frow_45 = row_base + ((1) ? 8 : 0);
                    if (tok_44 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_114 = __float2bfloat16(red[23]);
                        float _cvt_f32_114 = __bfloat162float(_cvt_bf16_114);
                        float gk_23 = _cvt_f32_114;
                        __nv_bfloat16 _cvt_bf16_115 = __float2bfloat16(red[55]);
                        float _cvt_f32_115 = __bfloat162float(_cvt_bf16_115);
                        float uk_23 = _cvt_f32_115;
                        float _exp2_23 = approx_exp2((-gk_23) * 1.4426950408889634f);
                        float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                        float sig_23 = _rcp_23;
                        float _tanh_46 = tanhf(gk_23 * 0.25f);
                        float aa_23 = 4.0f * _tanh_46 * sig_23;
                        float _tanh_47 = tanhf(uk_23 * 0.04f);
                        float bb_23 = 25.0f * _tanh_47;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_44 * 768 + row_s + frow_45)) + (0)) = __float2bfloat16_rn(aa_23 * bb_23);
                    }
                    int tok_46 = 48 + lane_pair * 2;
                    int frow_47 = row_base + ((0) ? 8 : 0);
                    if (tok_46 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_116 = __float2bfloat16(red[24]);
                        float _cvt_f32_116 = __bfloat162float(_cvt_bf16_116);
                        float gk_24 = _cvt_f32_116;
                        __nv_bfloat16 _cvt_bf16_117 = __float2bfloat16(red[56]);
                        float _cvt_f32_117 = __bfloat162float(_cvt_bf16_117);
                        float uk_24 = _cvt_f32_117;
                        float _exp2_24 = approx_exp2((-gk_24) * 1.4426950408889634f);
                        float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                        float sig_24 = _rcp_24;
                        float _tanh_48 = tanhf(gk_24 * 0.25f);
                        float aa_24 = 4.0f * _tanh_48 * sig_24;
                        float _tanh_49 = tanhf(uk_24 * 0.04f);
                        float bb_24 = 25.0f * _tanh_49;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_46 * 768 + row_s + frow_47)) + (0)) = __float2bfloat16_rn(aa_24 * bb_24);
                    }
                    int tok_48 = 48 + lane_pair * 2 + 1;
                    int frow_49 = row_base + ((0) ? 8 : 0);
                    if (tok_48 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_118 = __float2bfloat16(red[25]);
                        float _cvt_f32_118 = __bfloat162float(_cvt_bf16_118);
                        float gk_25 = _cvt_f32_118;
                        __nv_bfloat16 _cvt_bf16_119 = __float2bfloat16(red[57]);
                        float _cvt_f32_119 = __bfloat162float(_cvt_bf16_119);
                        float uk_25 = _cvt_f32_119;
                        float _exp2_25 = approx_exp2((-gk_25) * 1.4426950408889634f);
                        float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                        float sig_25 = _rcp_25;
                        float _tanh_50 = tanhf(gk_25 * 0.25f);
                        float aa_25 = 4.0f * _tanh_50 * sig_25;
                        float _tanh_51 = tanhf(uk_25 * 0.04f);
                        float bb_25 = 25.0f * _tanh_51;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_48 * 768 + row_s + frow_49)) + (0)) = __float2bfloat16_rn(aa_25 * bb_25);
                    }
                    int tok_50 = 48 + lane_pair * 2;
                    int frow_51 = row_base + ((1) ? 8 : 0);
                    if (tok_50 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_120 = __float2bfloat16(red[26]);
                        float _cvt_f32_120 = __bfloat162float(_cvt_bf16_120);
                        float gk_26 = _cvt_f32_120;
                        __nv_bfloat16 _cvt_bf16_121 = __float2bfloat16(red[58]);
                        float _cvt_f32_121 = __bfloat162float(_cvt_bf16_121);
                        float uk_26 = _cvt_f32_121;
                        float _exp2_26 = approx_exp2((-gk_26) * 1.4426950408889634f);
                        float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                        float sig_26 = _rcp_26;
                        float _tanh_52 = tanhf(gk_26 * 0.25f);
                        float aa_26 = 4.0f * _tanh_52 * sig_26;
                        float _tanh_53 = tanhf(uk_26 * 0.04f);
                        float bb_26 = 25.0f * _tanh_53;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_50 * 768 + row_s + frow_51)) + (0)) = __float2bfloat16_rn(aa_26 * bb_26);
                    }
                    int tok_52 = 48 + lane_pair * 2 + 1;
                    int frow_53 = row_base + ((1) ? 8 : 0);
                    if (tok_52 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_122 = __float2bfloat16(red[27]);
                        float _cvt_f32_122 = __bfloat162float(_cvt_bf16_122);
                        float gk_27 = _cvt_f32_122;
                        __nv_bfloat16 _cvt_bf16_123 = __float2bfloat16(red[59]);
                        float _cvt_f32_123 = __bfloat162float(_cvt_bf16_123);
                        float uk_27 = _cvt_f32_123;
                        float _exp2_27 = approx_exp2((-gk_27) * 1.4426950408889634f);
                        float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                        float sig_27 = _rcp_27;
                        float _tanh_54 = tanhf(gk_27 * 0.25f);
                        float aa_27 = 4.0f * _tanh_54 * sig_27;
                        float _tanh_55 = tanhf(uk_27 * 0.04f);
                        float bb_27 = 25.0f * _tanh_55;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_52 * 768 + row_s + frow_53)) + (0)) = __float2bfloat16_rn(aa_27 * bb_27);
                    }
                    int tok_54 = 56 + lane_pair * 2;
                    int frow_55 = row_base + ((0) ? 8 : 0);
                    if (tok_54 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_124 = __float2bfloat16(red[28]);
                        float _cvt_f32_124 = __bfloat162float(_cvt_bf16_124);
                        float gk_28 = _cvt_f32_124;
                        __nv_bfloat16 _cvt_bf16_125 = __float2bfloat16(red[60]);
                        float _cvt_f32_125 = __bfloat162float(_cvt_bf16_125);
                        float uk_28 = _cvt_f32_125;
                        float _exp2_28 = approx_exp2((-gk_28) * 1.4426950408889634f);
                        float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                        float sig_28 = _rcp_28;
                        float _tanh_56 = tanhf(gk_28 * 0.25f);
                        float aa_28 = 4.0f * _tanh_56 * sig_28;
                        float _tanh_57 = tanhf(uk_28 * 0.04f);
                        float bb_28 = 25.0f * _tanh_57;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_54 * 768 + row_s + frow_55)) + (0)) = __float2bfloat16_rn(aa_28 * bb_28);
                    }
                    int tok_56 = 56 + lane_pair * 2 + 1;
                    int frow_57 = row_base + ((0) ? 8 : 0);
                    if (tok_56 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_126 = __float2bfloat16(red[29]);
                        float _cvt_f32_126 = __bfloat162float(_cvt_bf16_126);
                        float gk_29 = _cvt_f32_126;
                        __nv_bfloat16 _cvt_bf16_127 = __float2bfloat16(red[61]);
                        float _cvt_f32_127 = __bfloat162float(_cvt_bf16_127);
                        float uk_29 = _cvt_f32_127;
                        float _exp2_29 = approx_exp2((-gk_29) * 1.4426950408889634f);
                        float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                        float sig_29 = _rcp_29;
                        float _tanh_58 = tanhf(gk_29 * 0.25f);
                        float aa_29 = 4.0f * _tanh_58 * sig_29;
                        float _tanh_59 = tanhf(uk_29 * 0.04f);
                        float bb_29 = 25.0f * _tanh_59;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_56 * 768 + row_s + frow_57)) + (0)) = __float2bfloat16_rn(aa_29 * bb_29);
                    }
                    int tok_58 = 56 + lane_pair * 2;
                    int frow_59 = row_base + ((1) ? 8 : 0);
                    if (tok_58 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_128 = __float2bfloat16(red[30]);
                        float _cvt_f32_128 = __bfloat162float(_cvt_bf16_128);
                        float gk_30 = _cvt_f32_128;
                        __nv_bfloat16 _cvt_bf16_129 = __float2bfloat16(red[62]);
                        float _cvt_f32_129 = __bfloat162float(_cvt_bf16_129);
                        float uk_30 = _cvt_f32_129;
                        float _exp2_30 = approx_exp2((-gk_30) * 1.4426950408889634f);
                        float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                        float sig_30 = _rcp_30;
                        float _tanh_60 = tanhf(gk_30 * 0.25f);
                        float aa_30 = 4.0f * _tanh_60 * sig_30;
                        float _tanh_61 = tanhf(uk_30 * 0.04f);
                        float bb_30 = 25.0f * _tanh_61;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_58 * 768 + row_s + frow_59)) + (0)) = __float2bfloat16_rn(aa_30 * bb_30);
                    }
                    int tok_60 = 56 + lane_pair * 2 + 1;
                    int frow_61 = row_base + ((1) ? 8 : 0);
                    if (tok_60 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_130 = __float2bfloat16(red[31]);
                        float _cvt_f32_130 = __bfloat162float(_cvt_bf16_130);
                        float gk_31 = _cvt_f32_130;
                        __nv_bfloat16 _cvt_bf16_131 = __float2bfloat16(red[63]);
                        float _cvt_f32_131 = __bfloat162float(_cvt_bf16_131);
                        float uk_31 = _cvt_f32_131;
                        float _exp2_31 = approx_exp2((-gk_31) * 1.4426950408889634f);
                        float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                        float sig_31 = _rcp_31;
                        float _tanh_62 = tanhf(gk_31 * 0.25f);
                        float aa_31 = 4.0f * _tanh_62 * sig_31;
                        float _tanh_63 = tanhf(uk_31 * 0.04f);
                        float bb_31 = 25.0f * _tanh_63;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_60 * 768 + row_s + frow_61)) + (0)) = __float2bfloat16_rn(aa_31 * bb_31);
                    }
                } else {
                    if (num_tokens > 0) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (row_r + tid_1)) + (0)) = red[0];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[0]);
                        }
                    }
                    if (num_tokens > 1) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (896 + row_r + tid_1)) + (0)) = red[1];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (3584 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[1]);
                        }
                    }
                    if (num_tokens > 2) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (1792 + row_r + tid_1)) + (0)) = red[2];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (7168 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[2]);
                        }
                    }
                    if (num_tokens > 3) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (2688 + row_r + tid_1)) + (0)) = red[3];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (10752 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[3]);
                        }
                    }
                    if (num_tokens > 4) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (3584 + row_r + tid_1)) + (0)) = red[4];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (14336 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[4]);
                        }
                    }
                    if (num_tokens > 5) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (4480 + row_r + tid_1)) + (0)) = red[5];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (17920 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[5]);
                        }
                    }
                    if (num_tokens > 6) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (5376 + row_r + tid_1)) + (0)) = red[6];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (21504 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[6]);
                        }
                    }
                    if (num_tokens > 7) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (6272 + row_r + tid_1)) + (0)) = red[7];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (25088 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[7]);
                        }
                    }
                    if (num_tokens > 8) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (7168 + row_r + tid_1)) + (0)) = red[8];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (28672 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[8]);
                        }
                    }
                    if (num_tokens > 9) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (8064 + row_r + tid_1)) + (0)) = red[9];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (32256 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[9]);
                        }
                    }
                    if (num_tokens > 10) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (8960 + row_r + tid_1)) + (0)) = red[10];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (35840 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[10]);
                        }
                    }
                    if (num_tokens > 11) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (9856 + row_r + tid_1)) + (0)) = red[11];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (39424 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[11]);
                        }
                    }
                    if (num_tokens > 12) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (10752 + row_r + tid_1)) + (0)) = red[12];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (43008 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[12]);
                        }
                    }
                    if (num_tokens > 13) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (11648 + row_r + tid_1)) + (0)) = red[13];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (46592 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[13]);
                        }
                    }
                    if (num_tokens > 14) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (12544 + row_r + tid_1)) + (0)) = red[14];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (50176 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[14]);
                        }
                    }
                    if (num_tokens > 15) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (13440 + row_r + tid_1)) + (0)) = red[15];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (53760 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[15]);
                        }
                    }
                    if (num_tokens > 16) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (14336 + row_r + tid_1)) + (0)) = red[16];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (57344 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[16]);
                        }
                    }
                    if (num_tokens > 17) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (15232 + row_r + tid_1)) + (0)) = red[17];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (60928 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[17]);
                        }
                    }
                    if (num_tokens > 18) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (16128 + row_r + tid_1)) + (0)) = red[18];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (64512 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[18]);
                        }
                    }
                    if (num_tokens > 19) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (17024 + row_r + tid_1)) + (0)) = red[19];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (68096 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[19]);
                        }
                    }
                    if (num_tokens > 20) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (17920 + row_r + tid_1)) + (0)) = red[20];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (71680 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[20]);
                        }
                    }
                    if (num_tokens > 21) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (18816 + row_r + tid_1)) + (0)) = red[21];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (75264 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[21]);
                        }
                    }
                    if (num_tokens > 22) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (19712 + row_r + tid_1)) + (0)) = red[22];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (78848 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[22]);
                        }
                    }
                    if (num_tokens > 23) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (20608 + row_r + tid_1)) + (0)) = red[23];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (82432 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[23]);
                        }
                    }
                    if (num_tokens > 24) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (21504 + row_r + tid_1)) + (0)) = red[24];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (86016 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[24]);
                        }
                    }
                    if (num_tokens > 25) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (22400 + row_r + tid_1)) + (0)) = red[25];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (89600 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[25]);
                        }
                    }
                    if (num_tokens > 26) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (23296 + row_r + tid_1)) + (0)) = red[26];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (93184 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[26]);
                        }
                    }
                    if (num_tokens > 27) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (24192 + row_r + tid_1)) + (0)) = red[27];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (96768 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[27]);
                        }
                    }
                    if (num_tokens > 28) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (25088 + row_r + tid_1)) + (0)) = red[28];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (100352 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[28]);
                        }
                    }
                    if (num_tokens > 29) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (25984 + row_r + tid_1)) + (0)) = red[29];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (103936 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[29]);
                        }
                    }
                    if (num_tokens > 30) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (26880 + row_r + tid_1)) + (0)) = red[30];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (107520 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[30]);
                        }
                    }
                    if (num_tokens > 31) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (27776 + row_r + tid_1)) + (0)) = red[31];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (111104 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[31]);
                        }
                    }
                    if (num_tokens > 32) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (28672 + row_r + tid_1)) + (0)) = red[32];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (114688 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[32]);
                        }
                    }
                    if (num_tokens > 33) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (29568 + row_r + tid_1)) + (0)) = red[33];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (118272 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[33]);
                        }
                    }
                    if (num_tokens > 34) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (30464 + row_r + tid_1)) + (0)) = red[34];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (121856 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[34]);
                        }
                    }
                    if (num_tokens > 35) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (31360 + row_r + tid_1)) + (0)) = red[35];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (125440 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[35]);
                        }
                    }
                    if (num_tokens > 36) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (32256 + row_r + tid_1)) + (0)) = red[36];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (129024 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[36]);
                        }
                    }
                    if (num_tokens > 37) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (33152 + row_r + tid_1)) + (0)) = red[37];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (132608 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[37]);
                        }
                    }
                    if (num_tokens > 38) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (34048 + row_r + tid_1)) + (0)) = red[38];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (136192 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[38]);
                        }
                    }
                    if (num_tokens > 39) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (34944 + row_r + tid_1)) + (0)) = red[39];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (139776 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[39]);
                        }
                    }
                    if (num_tokens > 40) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (35840 + row_r + tid_1)) + (0)) = red[40];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (143360 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[40]);
                        }
                    }
                    if (num_tokens > 41) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (36736 + row_r + tid_1)) + (0)) = red[41];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (146944 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[41]);
                        }
                    }
                    if (num_tokens > 42) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (37632 + row_r + tid_1)) + (0)) = red[42];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (150528 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[42]);
                        }
                    }
                    if (num_tokens > 43) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (38528 + row_r + tid_1)) + (0)) = red[43];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (154112 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[43]);
                        }
                    }
                    if (num_tokens > 44) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (39424 + row_r + tid_1)) + (0)) = red[44];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (157696 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[44]);
                        }
                    }
                    if (num_tokens > 45) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (40320 + row_r + tid_1)) + (0)) = red[45];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (161280 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[45]);
                        }
                    }
                    if (num_tokens > 46) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (41216 + row_r + tid_1)) + (0)) = red[46];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (164864 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[46]);
                        }
                    }
                    if (num_tokens > 47) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (42112 + row_r + tid_1)) + (0)) = red[47];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (168448 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[47]);
                        }
                    }
                    if (num_tokens > 48) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (43008 + row_r + tid_1)) + (0)) = red[48];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (172032 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[48]);
                        }
                    }
                    if (num_tokens > 49) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (43904 + row_r + tid_1)) + (0)) = red[49];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (175616 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[49]);
                        }
                    }
                    if (num_tokens > 50) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (44800 + row_r + tid_1)) + (0)) = red[50];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (179200 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[50]);
                        }
                    }
                    if (num_tokens > 51) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (45696 + row_r + tid_1)) + (0)) = red[51];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (182784 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[51]);
                        }
                    }
                    if (num_tokens > 52) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (46592 + row_r + tid_1)) + (0)) = red[52];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (186368 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[52]);
                        }
                    }
                    if (num_tokens > 53) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (47488 + row_r + tid_1)) + (0)) = red[53];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (189952 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[53]);
                        }
                    }
                    if (num_tokens > 54) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (48384 + row_r + tid_1)) + (0)) = red[54];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (193536 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[54]);
                        }
                    }
                    if (num_tokens > 55) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (49280 + row_r + tid_1)) + (0)) = red[55];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (197120 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[55]);
                        }
                    }
                    if (num_tokens > 56) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (50176 + row_r + tid_1)) + (0)) = red[56];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (200704 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[56]);
                        }
                    }
                    if (num_tokens > 57) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (51072 + row_r + tid_1)) + (0)) = red[57];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (204288 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[57]);
                        }
                    }
                    if (num_tokens > 58) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (51968 + row_r + tid_1)) + (0)) = red[58];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (207872 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[58]);
                        }
                    }
                    if (num_tokens > 59) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (52864 + row_r + tid_1)) + (0)) = red[59];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (211456 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[59]);
                        }
                    }
                    if (num_tokens > 60) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (53760 + row_r + tid_1)) + (0)) = red[60];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (215040 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[60]);
                        }
                    }
                    if (num_tokens > 61) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (54656 + row_r + tid_1)) + (0)) = red[61];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (218624 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[61]);
                        }
                    }
                    if (num_tokens > 62) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (55552 + row_r + tid_1)) + (0)) = red[62];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (222208 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[62]);
                        }
                    }
                    if (num_tokens > 63) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (56448 + row_r + tid_1)) + (0)) = red[63];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (225792 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[63]);
                        }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int g_l = bid;
            int tl_l = g_l * 16;
            int tile_l = g_l;
            int rank_l = 0;
            {
                tile_l = g_l / 2;
                rank_l = g_l % 2;
            }
            int d_lo_l = ((rank_l == 0) ? 0 : 0);
            int d_cnt_l = ((rank_l == 0) ? 0 : 0);
            int u_lo_l = ((rank_l == 0) ? 0 : 56);
            int u_cnt_l = ((rank_l == 0) ? 56 : 56);
            int u_count_l = d_cnt_l + u_cnt_l;
            unsigned int stage = 0;
            int _min_0 = ((8) < (u_count_l) ? (8) : (u_count_l));
            int _min_1 = ((3) < (u_count_l) ? (3) : (u_count_l));
            int pro = ((0) ? _min_0 : ((0) ? _min_1 : 0));
            {
                if (elect_sync()) {
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_R))) : "memory");
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_L))) : "memory");
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_S))) : "memory");
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_2))) : "memory");
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_1))) : "memory");
                    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_2))) : "memory");
                }
            }
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int _phase_a_empty = 1;
            if (elect_sync()) {
                int norm_ok = ((0) ? 1 : 0);
                #pragma unroll 1
                for (int s = 0; s < u_count_l; s++) {
                    mbarrier_wait(a_empty_addr + (stage) * 8, _phase_a_empty);
                    int tile_1 = tile_l;
                    int m = ((d_cnt_l > s) ? d_lo_l + s : u_lo_l + (s - d_cnt_l));
                    int cls_1 = ((tile_1 < 7) ? 0 : ((tile_1 < 35) ? 1 : 2));
                    int row_r_1 = tile_1 * 128;
                    int row_l_1 = (tile_1 - 7) * 128;
                    int row_s_1 = (tile_1 - 7 - 28) * 64;
                    int need_a = ((pro > s) ? 0 : 1);
                    {
                        int kb1 = k1_off + m;
                        tma_3d_gmem2smem(smem_b_addr + stage * 8192, (&B_1), 0, 0, kb1, a_full_addr + (stage) * 8);
                    }
                    if (need_a == 1) {
                        {
                            int kc1 = k1_off + m;
                            if (cls_1 == 0) {
                                tma_3d_gmem2smem(smem_a_addr + stage * 16384, (&A_R), 0, row_r_1, kc1, a_full_addr + (stage) * 8);
                            }
                            if (cls_1 == 1) {
                                tma_3d_gmem2smem(smem_a_addr + stage * 16384, (&A_L), 0, row_l_1, kc1, a_full_addr + (stage) * 8);
                            }
                            if (cls_1 == 2) {
                                tma_3d_gmem2smem(smem_a_addr + stage * 16384, (&A_S), 0, row_s_1, kc1, a_full_addr + (stage) * 8);
                                tma_3d_gmem2smem(smem_a_addr + stage * 16384 + 8192, (&A_S), 0, 768 + row_s_1, kc1, a_full_addr + (stage) * 8);
                            }
                        }
                    }
                    {
                        mbarrier_arrive_expect_tx(a_full_addr + (stage) * 8, 24576);
                    }
                    stage += 1;
                    if (stage == 8) { stage = 0; _phase_a_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int g_m = bid;
            int tl_m = g_m * 16;
            int tile_2 = g_m;
            int rank_m = 0;
            {
                tile_2 = g_m / 2;
                rank_m = g_m % 2;
            }
            int u_count_m = ((rank_m == 0) ? 56 : 56);
            int d_cnt_m = ((rank_m == 0) ? 0 : 0);
            unsigned int stage_1 = 0;
            unsigned int epi_stage_1 = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_a_full = 0;
            if (elect_sync()) {
                int cls_2 = ((tile_2 < 7) ? 0 : ((tile_2 < 35) ? 1 : 2));
                mbarrier_wait(acc_empty_addr + (epi_stage_1) * 8, _phase_acc_empty);
                #pragma unroll 1
                for (int m_1 = 0; m_1 < u_count_m; m_1++) {
                    mbarrier_wait(a_full_addr + (stage_1) * 8, _phase_a_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((m_1 == 0) ? 1 : 0);
                    {
                        int init_sub = ((1) ? init_flag : 0);
                        if (cls_2 == 2) {
                            int _mma_a_lo_0 = (((smem_ag_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 512;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_acc + (64))), "r"(((init_sub) ? 0 : 1)));
                            int _mma_a_lo_1 = (((smem_au_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 512;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_acc + (128))), "r"(((init_sub) ? 0 : 1)));
                        } else {
                            int _mma_a_lo_2 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 512;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 135267472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_acc), "r"(((init_sub) ? 0 : 1)));
                        }
                    }
                    tcgen05_commit(a_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 8) { stage_1 = 0; _phase_a_full ^= 1; }
                }
                tcgen05_commit(acc_full_addr + (epi_stage_1) * 8);
            }
        }
    }

    // Cleanup
}

} // extern "C"
