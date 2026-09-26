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
#define TMEM_NCOLS 128
#define TMEM_ACC_OFFSET 0
#define NUM_TMA_PIPE_STAGES 10
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_LAND_OFF 2048
#define SMEM_LAND_STAGE_BYTES 16384
#define SMEM_LAND_STRIDE 16384
#define SMEM_SMEM_A_OFF 18432
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 182272
#define SMEM_SMEM_B_STAGE_BYTES 4096
#define SMEM_SMEM_B_STRIDE 4096
#define SMEM_SMEM_BN_OFF 223232
#define SMEM_SMEM_BN_STAGE_BYTES 4096
#define SMEM_SMEM_BN_STRIDE 4096
#define SMEM_SMEM_AG_OFF 18432
#define SMEM_SMEM_AG_STAGE_BYTES 8192
#define SMEM_SMEM_AG_STRIDE 16384
#define SMEM_SMEM_AU_OFF 26624
#define SMEM_SMEM_AU_STAGE_BYTES 8192
#define SMEM_SMEM_AU_STRIDE 16384
#define SMEM_TOTAL 223232

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

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_latent_moe_93829c60f71de95c5423(const __grid_constant__ CUtensorMap A_R, const __grid_constant__ CUtensorMap A_L, const __grid_constant__ CUtensorMap A_S, const __grid_constant__ CUtensorMap A_2, const __grid_constant__ CUtensorMap B_1, const __grid_constant__ CUtensorMap B_2, float* __restrict__ out_r, __nv_bfloat16* __restrict__ out_l, __nv_bfloat16* __restrict__ out_s, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out, unsigned long long* __restrict__ tl, int num_tokens, int k1_off, int num_partials, float eps)
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
    #define a_empty_addr (mbar_base + 80)
    #define acc_full_addr (mbar_base + 160)
    #define acc_empty_addr (mbar_base + 168)
    #define red_bar_addr (mbar_base + 176)
    #define issue_bar_addr (mbar_base + 184)
    #define rows_bar_addr (mbar_base + 192)
    #define bn_bar_addr (mbar_base + 200)

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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 18432);
    const int smem_a_addr = smem + 18432;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 182272);
    const int smem_b_addr = smem + 182272;
    __nv_bfloat16* smem_bn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 223232);
    const int smem_bn_addr = smem + 223232;
    __nv_bfloat16* smem_ag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 18432);
    const int smem_ag_addr = smem + 18432;
    __nv_bfloat16* smem_au = reinterpret_cast<__nv_bfloat16*>(smem_raw + 26624);
    const int smem_au_addr = smem + 26624;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // a_full: 10 barriers, init_count=1
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
            // a_empty: 10 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'epi_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // acc_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            // red_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // issue_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            // rows_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // bn_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 176, 16384);
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 0) {
        int _tmem_hold = smem + 208;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
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
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                {
                    int nrow = g_e + epi_warp * 112;
                    if (nrow < num_tokens) {
                        unsigned long long nrow_base = (unsigned long long)nrow * 3584;
                        unsigned long long partial_stride = (unsigned long long)num_tokens * 3584;
                        unsigned int nwp[56];
                        unsigned int xp[56];
                        float nacc[112];
                        {
                            #pragma unroll
                            for (int i = 0; i < 14; i++) {
                                int kk0 = (lane + i * 32) * 8;
                                {
                                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(routed + (nrow_base + (unsigned long long)kk0) + 0);
                                    uint4* _vdst_0 = reinterpret_cast<uint4*>(&xp[i * 4]);
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vdst_0[_blk] = _vptr_0[_blk];
                                    }
                                }
                            }
                        }
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 14; i_1++) {
                            int kw = (lane + i_1 * 32) * 8;
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(norm_w + kw + 0);
                                uint4* _vdst_1 = reinterpret_cast<uint4*>(&nwp[i_1 * 4]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_1[_blk] = _vptr_1[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int i_2 = 0; i_2 < 56; i_2++) {
                            {
                                float _bf16x2_add_f32_16[2];
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b16 lo, hi;\n\t"
                                    "mov.b32 {lo, hi}, %2;\n\t"
                                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                    "}\n"
                                    : "=&f"(_bf16x2_add_f32_16[0]), "=&f"(_bf16x2_add_f32_16[1]) : "r"(xp[i_2]), "f"(0.0f), "f"(0.0f));
                                nacc[2 * i_2] = _bf16x2_add_f32_16[0];
                                nacc[2 * i_2 + 1] = _bf16x2_add_f32_16[1];
                            }
                        }
                        {
                            #pragma unroll 1
                            for (int p = 1; p < num_partials; p++) {
                                unsigned long long src_base = (unsigned long long)p * partial_stride + nrow_base;
                                #pragma unroll
                                for (int i_3 = 0; i_3 < 14; i_3++) {
                                    int kk = (lane + i_3 * 32) * 8;
                                    float _vec_load_8[8];
                                    {
                                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(routed + (src_base + (unsigned long long)kk) + 0);
                                        uint4 _vld_2[1];
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vld_2[_blk] = _vptr_2[_blk];
                                            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[1])
                                                    : "r"(_vpairs_2[_pair]));
                                            }
                                        }
                                    }
                                    #pragma unroll
                                    for (int j = 0; j < 8; j++) {
                                        nacc[i_3 * 8 + j] = nacc[i_3 * 8 + j] + _vec_load_8[j];
                                    }
                                }
                            }
                        }
                        if (warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(issue_bar_addr);
                            }
                        }
                        float sum_sq = 0.0f;
                        #pragma unroll
                        for (int i_4 = 0; i_4 < 112; i_4++) {
                            __nv_bfloat16 _cvt_bf16_32 = __float2bfloat16(nacc[i_4]);
                            float _cvt_f32_32 = __bfloat162float(_cvt_bf16_32);
                            float sv = _cvt_f32_32;
                            sum_sq += sv * sv;
                        }
                        float _warp_reduce_8 = sum_sq;
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_8 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_8, offset);
                        float total = _warp_reduce_8;
                        float _rsqrt_8 = rsqrtf(total / 3584.0f + eps);
                        float rstd = _rsqrt_8;
                        #pragma unroll
                        for (int i_5 = 0; i_5 < 14; i_5++) {
                            int kk2 = (lane + i_5 * 32) * 8;
                            float nvals[8];
                            #pragma unroll
                            for (int jj = 0; jj < 4; jj++) {
                                float _bf16x2_add_f32_17[2];
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b16 lo, hi;\n\t"
                                    "mov.b32 {lo, hi}, %2;\n\t"
                                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                    "}\n"
                                    : "=&f"(_bf16x2_add_f32_17[0]), "=&f"(_bf16x2_add_f32_17[1]) : "r"(nwp[i_5 * 4 + jj]), "f"(0.0f), "f"(0.0f));
                                #pragma unroll
                                for (int h = 0; h < 2; h++) {
                                    __nv_bfloat16 _cvt_bf16_33 = __float2bfloat16(nacc[i_5 * 8 + jj * 2 + h]);
                                    float _cvt_f32_33 = __bfloat162float(_cvt_bf16_33);
                                    float s2 = _cvt_f32_33;
                                    __nv_bfloat16 _cvt_bf16_34 = __float2bfloat16(s2 * rstd);
                                    float _cvt_f32_34 = __bfloat162float(_cvt_bf16_34);
                                    float normed = _cvt_f32_34;
                                    __nv_bfloat16 _cvt_bf16_35 = __float2bfloat16(_bf16x2_add_f32_17[h] * normed);
                                    float _cvt_f32_35 = __bfloat162float(_cvt_bf16_35);
                                    nvals[jj * 2 + h] = _cvt_f32_35;
                                }
                            }
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(nvals[0 + 0], nvals[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(nvals[0 + 2], nvals[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(nvals[0 + 4], nvals[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(nvals[0 + 6], nvals[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(y_out + (nrow_base + (unsigned long long)kk2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                        asm volatile("fence.release.gpu;" ::: "memory");
                    } else if (warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(issue_bar_addr);
                        }
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(rows_bar_addr);
                        }
                    }
                }
            }
            int tile = g_e;
            int slot = 0;
            {
                tile = g_e / 2;
                slot = g_e % 2;
            }
            int cls = ((tile < 0) ? 0 : ((tile < 56) ? 1 : 2));
            int row_r = tile * 128;
            int row_l = tile * 128;
            int row_s = (tile - 56) * 64;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            unsigned int epi_stage = 0;
            float red[32];
            int fin = 1;
            unsigned int _phase_acc_full = 0;
            mbarrier_wait(acc_full_addr + (epi_stage) * 8, _phase_acc_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            {
                float _tmem_load_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                    : "r"(lane_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                red[0] = _tmem_load_3[0];
                red[1] = _tmem_load_3[1];
                red[2] = _tmem_load_3[2];
                red[3] = _tmem_load_3[3];
                red[4] = _tmem_load_3[4];
                red[5] = _tmem_load_3[5];
                red[6] = _tmem_load_3[6];
                red[7] = _tmem_load_3[7];
                red[8] = _tmem_load_3[8];
                red[9] = _tmem_load_3[9];
                red[10] = _tmem_load_3[10];
                red[11] = _tmem_load_3[11];
                red[12] = _tmem_load_3[12];
                red[13] = _tmem_load_3[13];
                red[14] = _tmem_load_3[14];
                red[15] = _tmem_load_3[15];
                red[16] = _tmem_load_3[16];
                red[17] = _tmem_load_3[17];
                red[18] = _tmem_load_3[18];
                red[19] = _tmem_load_3[19];
                red[20] = _tmem_load_3[20];
                red[21] = _tmem_load_3[21];
                red[22] = _tmem_load_3[22];
                red[23] = _tmem_load_3[23];
                red[24] = _tmem_load_3[24];
                red[25] = _tmem_load_3[25];
                red[26] = _tmem_load_3[26];
                red[27] = _tmem_load_3[27];
                red[28] = _tmem_load_3[28];
                red[29] = _tmem_load_3[29];
                red[30] = _tmem_load_3[30];
                red[31] = _tmem_load_3[31];
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
            }
            if (fin == 1) {
                {
                    if (num_tokens > 0) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[0]);
                        }
                    }
                    if (num_tokens > 1) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (7168 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[1]);
                        }
                    }
                    if (num_tokens > 2) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (14336 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[2]);
                        }
                    }
                    if (num_tokens > 3) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (21504 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[3]);
                        }
                    }
                    if (num_tokens > 4) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (28672 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[4]);
                        }
                    }
                    if (num_tokens > 5) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (35840 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[5]);
                        }
                    }
                    if (num_tokens > 6) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (43008 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[6]);
                        }
                    }
                    if (num_tokens > 7) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (50176 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[7]);
                        }
                    }
                    if (num_tokens > 8) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (57344 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[8]);
                        }
                    }
                    if (num_tokens > 9) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (64512 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[9]);
                        }
                    }
                    if (num_tokens > 10) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (71680 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[10]);
                        }
                    }
                    if (num_tokens > 11) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (78848 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[11]);
                        }
                    }
                    if (num_tokens > 12) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (86016 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[12]);
                        }
                    }
                    if (num_tokens > 13) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (93184 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[13]);
                        }
                    }
                    if (num_tokens > 14) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (100352 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[14]);
                        }
                    }
                    if (num_tokens > 15) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (107520 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[15]);
                        }
                    }
                    if (num_tokens > 16) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (114688 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[16]);
                        }
                    }
                    if (num_tokens > 17) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (121856 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[17]);
                        }
                    }
                    if (num_tokens > 18) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (129024 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[18]);
                        }
                    }
                    if (num_tokens > 19) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (136192 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[19]);
                        }
                    }
                    if (num_tokens > 20) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (143360 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[20]);
                        }
                    }
                    if (num_tokens > 21) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (150528 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[21]);
                        }
                    }
                    if (num_tokens > 22) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (157696 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[22]);
                        }
                    }
                    if (num_tokens > 23) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (164864 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[23]);
                        }
                    }
                    if (num_tokens > 24) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (172032 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[24]);
                        }
                    }
                    if (num_tokens > 25) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (179200 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[25]);
                        }
                    }
                    if (num_tokens > 26) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (186368 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[26]);
                        }
                    }
                    if (num_tokens > 27) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (193536 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[27]);
                        }
                    }
                    if (num_tokens > 28) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (200704 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[28]);
                        }
                    }
                    if (num_tokens > 29) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (207872 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[29]);
                        }
                    }
                    if (num_tokens > 30) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (215040 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[30]);
                        }
                    }
                    if (num_tokens > 31) {
                        {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (222208 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[31]);
                        }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(128));
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
            int d_lo_l = ((rank_l == 0) ? 0 : 48);
            int d_cnt_l = ((rank_l == 0) ? 48 : 48);
            int u_lo_l = ((rank_l == 0) ? 0 : 28);
            int u_cnt_l = ((rank_l == 0) ? 28 : 28);
            int u_count_l = d_cnt_l + u_cnt_l;
            unsigned int stage = 0;
            int _min_0 = ((10) < (u_count_l) ? (10) : (u_count_l));
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
                    int m = ((d_cnt_l > s) ? d_lo_l + s : 96 + u_lo_l + (s - d_cnt_l));
                    int cls_1 = ((tile_1 < 0) ? 0 : ((tile_1 < 56) ? 1 : 2));
                    int row_r_1 = tile_1 * 128;
                    int row_l_1 = tile_1 * 128;
                    int row_s_1 = (tile_1 - 56) * 64;
                    int need_a = ((pro > s) ? 0 : 1);
                    if (m < 96) {
                        int kb2 = m;
                        tma_3d_gmem2smem(smem_b_addr + stage * 4096, (&B_2), 0, 0, kb2, a_full_addr + (stage) * 8);
                    } else if (!0) {
                        if (norm_ok == 0) {
                            mbarrier_wait(rows_bar_addr, 0);
                            asm volatile("fence.release.gpu;" ::: "memory");
                            uint32_t _atomic_inc_old_0;
                            asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                                : "=r"(_atomic_inc_old_0) : "l"(&counters[0]), "r"(static_cast<uint32_t>(111)) : "memory");
                            unsigned int old = _atomic_inc_old_0;
                            if ((int)old != 111) {
                                #pragma unroll 1
                                for (int _spin = 0; _spin < 4194304; _spin++) {
                                    uint32_t _relaxed_ld_0;
                                    asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(_relaxed_ld_0) : "l"(counters + 0) : "memory");
                                    unsigned int cnt = _relaxed_ld_0;
                                    if (cnt == 0) {
                                        break;
                                    }
                                }
                            }
                            asm volatile("fence.acquire.gpu;" ::: "memory");
                            asm volatile("fence.proxy.async;");
                            norm_ok = 1;
                        }
                        int kb = k1_off + (m - 96);
                        tma_3d_gmem2smem(smem_b_addr + stage * 4096, (&B_1), 0, 0, kb, a_full_addr + (stage) * 8);
                    }
                    if (need_a == 1) {
                        if (m < 96) {
                            int kc2 = m;
                            tma_3d_gmem2smem(smem_a_addr + stage * 16384, (&A_2), 0, row_l_1, kc2, a_full_addr + (stage) * 8);
                        } else {
                            int kc = k1_off + (m - 96);
                            tma_3d_gmem2smem(smem_a_addr + stage * 16384, (&A_L), 0, row_l_1, kc, a_full_addr + (stage) * 8);
                        }
                    }
                    {
                        mbarrier_arrive_expect_tx(a_full_addr + (stage) * 8, 20480);
                    }
                    stage += 1;
                    if (stage == 10) { stage = 0; _phase_a_empty ^= 1; }
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
            int u_count_m = ((rank_m == 0) ? 76 : 76);
            int d_cnt_m = ((rank_m == 0) ? 48 : 48);
            unsigned int stage_1 = 0;
            unsigned int epi_stage_1 = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_a_full = 0;
            if (elect_sync()) {
                int cls_2 = ((tile_2 < 0) ? 0 : ((tile_2 < 56) ? 1 : 2));
                mbarrier_wait(acc_empty_addr + (epi_stage_1) * 8, _phase_acc_empty);
                #pragma unroll 1
                for (int m_1 = 0; m_1 < u_count_m; m_1++) {
                    mbarrier_wait(a_full_addr + (stage_1) * 8, _phase_a_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((m_1 == 0) ? 1 : 0);
                    {
                        int init_sub = ((1) ? init_flag : 0);
                        {
                            int _mma_a_lo_5 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_5 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 256;
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
                    "mov.b32 id, 134743184;\n\t"
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"(tmem_acc), "r"(((init_sub) ? 0 : 1)));
                        }
                    }
                    tcgen05_commit(a_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 10) { stage_1 = 0; _phase_a_full ^= 1; }
                }
                tcgen05_commit(acc_full_addr + (epi_stage_1) * 8);
            }
        }
    }

    // Cleanup
}

} // extern "C"
