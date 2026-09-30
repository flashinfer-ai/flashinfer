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
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 7
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_WORK_RESPONSE_OFF 230400
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FIX_FLAG_OFF 230464
#define SMEM_FIX_FLAG_STAGE_BYTES 16
#define SMEM_FIX_FLAG_STRIDE 16
#define SMEM_TOTAL 230528
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 64
#define CTA_GROUP 2
#define NUM_STAGES 7
#define GROUP_M 16
#define WORK_STAGES 4
#define WORK_CONSUMERS 546
#define NUM_K_ITERS 112
#define R_TILES 4
#define L_TILES 14
#define S_TILES 6
#define N_TILES 24
#define I_LOCAL 768
#define NUM_EXPERTS 896
#define LATENT 3584
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
        :: "l"(mbar_addr), "r"(count) : "memory");
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


__device__ __forceinline__ void tcgen05_mma_f16_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_latent_moe_c87d3bed9c6312a4cb48(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap WG, const __grid_constant__ CUtensorMap WD, const __grid_constant__ CUtensorMap WS, float* __restrict__ logits, __nv_bfloat16* __restrict__ latent, __nv_bfloat16* __restrict__ shared_act, float* __restrict__ ws, int* __restrict__ counters, int M, int m_tiles, int num_items, int full_items, int sk_ipc, int sk_max_seg, int sk_total)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 56)
    #define mainloop_done_addr (mbar_base + 112)
    #define epilogue_done_addr (mbar_base + 128)
    #define work_full_addr (mbar_base + 144)
    #define work_empty_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 230400);
    const int work_response_addr = smem + 230400;
    int* fix_flag = reinterpret_cast<int*>(smem_raw + 230464);
    const int fix_flag_addr = smem + 230464;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 7 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // mma_done: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // epilogue_done: 2 barriers, init_count=16
            mbarrier_init(smem + 128, 16);
            mbarrier_init(smem + 136, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 176, 546);
            mbarrier_init(smem + 184, 546);
            mbarrier_init(smem + 192, 546);
            mbarrier_init(smem + 200, 546);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 0) {
        int _tmem_hold = smem + 208;
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

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < num_items; _tile_iter++) {
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
                    int c = (int)this_bid / CTA_GROUP;
                    int rank_i = (int)this_bid % CTA_GROUP;
                    int is_sk = ((c >= full_items) ? 1 : 0);
                    int j = c - full_items;
                    int start = ((is_sk == 1) ? j * sk_ipc : c * NUM_K_ITERS);
                    int _min_0 = ((sk_total) < (start + sk_ipc) ? (sk_total) : (start + sk_ipc));
                    int end = ((is_sk == 1) ? _min_0 : start + NUM_K_ITERS);
                    int first_tl = start / NUM_K_ITERS;
                    int nsegs = (end - 1) / NUM_K_ITERS - first_tl + 1;
                    #pragma unroll 1
                    for (int seg_l = 0; seg_l < nsegs; seg_l++) {
                        int tl = first_tl + seg_l;
                        int tile_begin = tl * NUM_K_ITERS;
                        int _max_0 = ((start - tile_begin) > (0) ? (start - tile_begin) : (0));
                        int k_lo = _max_0;
                        int _min_1 = ((NUM_K_ITERS) < (end - tile_begin) ? (NUM_K_ITERS) : (end - tile_begin));
                        int k_hi = _min_1;
                        int tile = ((is_sk == 1) ? full_items + tl : tl);
                        int j_first = tile_begin / sk_ipc;
                        int j_last = (tile_begin + NUM_K_ITERS - 1) / sk_ipc;
                        int nseg = ((is_sk == 1) ? j_last - j_first + 1 : 1);
                        int seg = ((is_sk == 1) ? j - j_first : 0);
                        int pseudo = tile * CTA_GROUP + rank_i;
                        int group = pseudo / tiles_per_group;
                        int first_m = group * GROUP_M;
                        int remaining = m_tiles - first_m;
                        int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                        int local = pseudo % tiles_per_group;
                        int bid_m = first_m + local % group_size;
                        int bid_n = local / group_size;
                        int off_m = bid_m * BLOCK_M;
                        int cls = ((bid_n < R_TILES) ? 0 : ((bid_n < R_TILES + L_TILES) ? 1 : 2));
                        int row_r = bid_n * BLOCK_N + cta_rank * B_HALF_N;
                        int row_l = (bid_n - R_TILES) * BLOCK_N + cta_rank * B_HALF_N;
                        int row_s = cta_rank * I_LOCAL + (bid_n - R_TILES - L_TILES) * B_HALF_N;
                        #pragma unroll 1
                        for (int iter_k = k_lo; iter_k < k_hi; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            int off_k = iter_k * BLOCK_K;
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, off_m, off_k / 64, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            if (cls == 0) {
                                tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&WG), 0, row_r, off_k / 64, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            }
                            if (cls == 1) {
                                tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&WD), 0, row_l, off_k / 64, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            }
                            if (cls == 2) {
                                tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&WS), 0, row_s, off_k / 64, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            }
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            load_stage += 1;
                            if (load_stage == 7) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
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
            unsigned int mma_epi_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int this_bid_m = bid;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_items; _tile_iter_1++) {
                    int c_1 = (int)this_bid_m / CTA_GROUP;
                    int rank_i_1 = (int)this_bid_m % CTA_GROUP;
                    int is_sk_1 = ((c_1 >= full_items) ? 1 : 0);
                    int j_1 = c_1 - full_items;
                    int start_1 = ((is_sk_1 == 1) ? j_1 * sk_ipc : c_1 * NUM_K_ITERS);
                    int _min_2 = ((sk_total) < (start_1 + sk_ipc) ? (sk_total) : (start_1 + sk_ipc));
                    int end_1 = ((is_sk_1 == 1) ? _min_2 : start_1 + NUM_K_ITERS);
                    int first_tl_1 = start_1 / NUM_K_ITERS;
                    int nsegs_1 = (end_1 - 1) / NUM_K_ITERS - first_tl_1 + 1;
                    #pragma unroll 1
                    for (int seg_m = 0; seg_m < nsegs_1; seg_m++) {
                        mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                        int tl_1 = first_tl_1 + seg_m;
                        int tile_begin_1 = tl_1 * NUM_K_ITERS;
                        int _max_1 = ((start_1 - tile_begin_1) > (0) ? (start_1 - tile_begin_1) : (0));
                        int k_lo_1 = _max_1;
                        int _min_3 = ((NUM_K_ITERS) < (end_1 - tile_begin_1) ? (NUM_K_ITERS) : (end_1 - tile_begin_1));
                        int k_hi_1 = _min_3;
                        int tile_1 = ((is_sk_1 == 1) ? full_items + tl_1 : tl_1);
                        int j_first_1 = tile_begin_1 / sk_ipc;
                        int j_last_1 = (tile_begin_1 + NUM_K_ITERS - 1) / sk_ipc;
                        int nseg_1 = ((is_sk_1 == 1) ? j_last_1 - j_first_1 + 1 : 1);
                        int seg_1 = ((is_sk_1 == 1) ? j_1 - j_first_1 : 0);
                        int pseudo_1 = tile_1 * CTA_GROUP + rank_i_1;
                        #pragma unroll 1
                        for (int iter_k_1 = k_lo_1; iter_k_1 < k_hi_1; iter_k_1++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag = ((iter_k_1 == k_lo_1) ? 1 : 0);
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 2048;
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
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (mma_epi_stage * 256))), "r"(((init_flag) ? 0 : 1)));
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 7) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                        mma_epi_stage += 1;
                        if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    }
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
                    this_bid_m = _clc_ctaid_1 + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_half = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            unsigned int this_bid_1 = bid;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_items; _tile_iter_2++) {
                int c_2 = (int)this_bid_1 / CTA_GROUP;
                int rank_i_2 = (int)this_bid_1 % CTA_GROUP;
                int is_sk_2 = ((c_2 >= full_items) ? 1 : 0);
                int j_2 = c_2 - full_items;
                int start_2 = ((is_sk_2 == 1) ? j_2 * sk_ipc : c_2 * NUM_K_ITERS);
                int _min_4 = ((sk_total) < (start_2 + sk_ipc) ? (sk_total) : (start_2 + sk_ipc));
                int end_2 = ((is_sk_2 == 1) ? _min_4 : start_2 + NUM_K_ITERS);
                int first_tl_2 = start_2 / NUM_K_ITERS;
                int nsegs_2 = (end_2 - 1) / NUM_K_ITERS - first_tl_2 + 1;
                #pragma unroll 1
                for (int seg_i = 0; seg_i < nsegs_2; seg_i++) {
                    mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int tl_2 = first_tl_2 + seg_i;
                    int tile_begin_2 = tl_2 * NUM_K_ITERS;
                    int _max_2 = ((start_2 - tile_begin_2) > (0) ? (start_2 - tile_begin_2) : (0));
                    int k_lo_2 = _max_2;
                    int _min_5 = ((NUM_K_ITERS) < (end_2 - tile_begin_2) ? (NUM_K_ITERS) : (end_2 - tile_begin_2));
                    int k_hi_2 = _min_5;
                    int tile_2 = ((is_sk_2 == 1) ? full_items + tl_2 : tl_2);
                    int j_first_2 = tile_begin_2 / sk_ipc;
                    int j_last_2 = (tile_begin_2 + NUM_K_ITERS - 1) / sk_ipc;
                    int nseg_2 = ((is_sk_2 == 1) ? j_last_2 - j_first_2 + 1 : 1);
                    int seg_2 = ((is_sk_2 == 1) ? j_2 - j_first_2 : 0);
                    int pseudo_2 = tile_2 * CTA_GROUP + rank_i_2;
                    int group_1 = pseudo_2 / tiles_per_group;
                    int first_m_1 = group_1 * GROUP_M;
                    int remaining_1 = m_tiles - first_m_1;
                    int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                    int local_1 = pseudo_2 % tiles_per_group;
                    int bid_m_1 = first_m_1 + local_1 % group_size_1;
                    int bid_n_1 = local_1 / group_size_1;
                    int off_m_1 = bid_m_1 * BLOCK_M;
                    int cls_1 = ((bid_n_1 < R_TILES) ? 0 : ((bid_n_1 < R_TILES + L_TILES) ? 1 : 2));
                    int global_row = off_m_1 + local_row;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + epi_stage * (unsigned int)BLOCK_N;
                    int row_ok = ((global_row < M) ? 1 : 0);
                    unsigned long long slot0_e = (unsigned long long)(tl_2 * sk_max_seg * CTA_GROUP + cta_rank) * (unsigned long long)(BLOCK_M * BLOCK_N) + (unsigned long long)local_row * 16;
                    int last_e = 1;
                    int sym_e = ((nseg_2 == 2) ? 1 : 0);
                    if (nseg_2 > 1) {
                        unsigned long long pbase_e = slot0_e + (unsigned long long)seg_2 * (unsigned long long)(CTA_GROUP * BLOCK_M * BLOCK_N);
                        int cidx_e = tl_2 * CTA_GROUP + cta_rank;
                        if (sym_e == 1) {
                            if (cls_1 < 2) {
                                int pcol0_e = col_half * B_HALF_N + (1 - seg_2) * 64;
                                #pragma unroll 1
                                for (int n_chunk = 0; n_chunk < 4; n_chunk++) {
                                    int colq = pcol0_e + n_chunk * 16;
                                    float _tmem_load_0[16];
                                    tmem_ld_x16(&_tmem_load_0[0], lane_addr + colq);
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    {
                                        unsigned _stv8_0_0 = __float_as_uint(_tmem_load_0[0 + 0]);
                                        unsigned _stv8_0_1 = __float_as_uint(_tmem_load_0[0 + 1]);
                                        unsigned _stv8_0_2 = __float_as_uint(_tmem_load_0[0 + 2]);
                                        unsigned _stv8_0_3 = __float_as_uint(_tmem_load_0[0 + 3]);
                                        unsigned _stv8_0_4 = __float_as_uint(_tmem_load_0[0 + 4]);
                                        unsigned _stv8_0_5 = __float_as_uint(_tmem_load_0[0 + 5]);
                                        unsigned _stv8_0_6 = __float_as_uint(_tmem_load_0[0 + 6]);
                                        unsigned _stv8_0_7 = __float_as_uint(_tmem_load_0[0 + 7]);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(ws + (pbase_e + (unsigned long long)(colq * BLOCK_M)) + (0))), "r"(_stv8_0_0), "r"(_stv8_0_1), "r"(_stv8_0_2), "r"(_stv8_0_3), "r"(_stv8_0_4), "r"(_stv8_0_5), "r"(_stv8_0_6), "r"(_stv8_0_7) : "memory");
                                    }
                                    {
                                        unsigned _stv8_1_0 = __float_as_uint(_tmem_load_0[8 + 0]);
                                        unsigned _stv8_1_1 = __float_as_uint(_tmem_load_0[8 + 1]);
                                        unsigned _stv8_1_2 = __float_as_uint(_tmem_load_0[8 + 2]);
                                        unsigned _stv8_1_3 = __float_as_uint(_tmem_load_0[8 + 3]);
                                        unsigned _stv8_1_4 = __float_as_uint(_tmem_load_0[8 + 4]);
                                        unsigned _stv8_1_5 = __float_as_uint(_tmem_load_0[8 + 5]);
                                        unsigned _stv8_1_6 = __float_as_uint(_tmem_load_0[8 + 6]);
                                        unsigned _stv8_1_7 = __float_as_uint(_tmem_load_0[8 + 7]);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(ws + (pbase_e + (unsigned long long)(colq * BLOCK_M) + 8) + (0))), "r"(_stv8_1_0), "r"(_stv8_1_1), "r"(_stv8_1_2), "r"(_stv8_1_3), "r"(_stv8_1_4), "r"(_stv8_1_5), "r"(_stv8_1_6), "r"(_stv8_1_7) : "memory");
                                    }
                                }
                            }
                            if (cls_1 == 2) {
                                int pcolg_e = col_half * (B_HALF_N / 2) + (1 - seg_2) * 32;
                                #pragma unroll 1
                                for (int n_chunk_1 = 0; n_chunk_1 < 2; n_chunk_1++) {
                                    int colq_g = pcolg_e + n_chunk_1 * 16;
                                    float _tmem_load_1[16];
                                    tmem_ld_x16(&_tmem_load_1[0], lane_addr + colq_g);
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    {
                                        unsigned _stv8_2_0 = __float_as_uint(_tmem_load_1[0 + 0]);
                                        unsigned _stv8_2_1 = __float_as_uint(_tmem_load_1[0 + 1]);
                                        unsigned _stv8_2_2 = __float_as_uint(_tmem_load_1[0 + 2]);
                                        unsigned _stv8_2_3 = __float_as_uint(_tmem_load_1[0 + 3]);
                                        unsigned _stv8_2_4 = __float_as_uint(_tmem_load_1[0 + 4]);
                                        unsigned _stv8_2_5 = __float_as_uint(_tmem_load_1[0 + 5]);
                                        unsigned _stv8_2_6 = __float_as_uint(_tmem_load_1[0 + 6]);
                                        unsigned _stv8_2_7 = __float_as_uint(_tmem_load_1[0 + 7]);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(ws + (pbase_e + (unsigned long long)(colq_g * BLOCK_M)) + (0))), "r"(_stv8_2_0), "r"(_stv8_2_1), "r"(_stv8_2_2), "r"(_stv8_2_3), "r"(_stv8_2_4), "r"(_stv8_2_5), "r"(_stv8_2_6), "r"(_stv8_2_7) : "memory");
                                    }
                                    {
                                        unsigned _stv8_3_0 = __float_as_uint(_tmem_load_1[8 + 0]);
                                        unsigned _stv8_3_1 = __float_as_uint(_tmem_load_1[8 + 1]);
                                        unsigned _stv8_3_2 = __float_as_uint(_tmem_load_1[8 + 2]);
                                        unsigned _stv8_3_3 = __float_as_uint(_tmem_load_1[8 + 3]);
                                        unsigned _stv8_3_4 = __float_as_uint(_tmem_load_1[8 + 4]);
                                        unsigned _stv8_3_5 = __float_as_uint(_tmem_load_1[8 + 5]);
                                        unsigned _stv8_3_6 = __float_as_uint(_tmem_load_1[8 + 6]);
                                        unsigned _stv8_3_7 = __float_as_uint(_tmem_load_1[8 + 7]);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(ws + (pbase_e + (unsigned long long)(colq_g * BLOCK_M) + 8) + (0))), "r"(_stv8_3_0), "r"(_stv8_3_1), "r"(_stv8_3_2), "r"(_stv8_3_3), "r"(_stv8_3_4), "r"(_stv8_3_5), "r"(_stv8_3_6), "r"(_stv8_3_7) : "memory");
                                    }
                                    int colq_u = B_HALF_N + pcolg_e + n_chunk_1 * 16;
                                    float _tmem_load_2[16];
                                    tmem_ld_x16(&_tmem_load_2[0], lane_addr + colq_u);
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    {
                                        unsigned _stv8_4_0 = __float_as_uint(_tmem_load_2[0 + 0]);
                                        unsigned _stv8_4_1 = __float_as_uint(_tmem_load_2[0 + 1]);
                                        unsigned _stv8_4_2 = __float_as_uint(_tmem_load_2[0 + 2]);
                                        unsigned _stv8_4_3 = __float_as_uint(_tmem_load_2[0 + 3]);
                                        unsigned _stv8_4_4 = __float_as_uint(_tmem_load_2[0 + 4]);
                                        unsigned _stv8_4_5 = __float_as_uint(_tmem_load_2[0 + 5]);
                                        unsigned _stv8_4_6 = __float_as_uint(_tmem_load_2[0 + 6]);
                                        unsigned _stv8_4_7 = __float_as_uint(_tmem_load_2[0 + 7]);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(ws + (pbase_e + (unsigned long long)(colq_u * BLOCK_M)) + (0))), "r"(_stv8_4_0), "r"(_stv8_4_1), "r"(_stv8_4_2), "r"(_stv8_4_3), "r"(_stv8_4_4), "r"(_stv8_4_5), "r"(_stv8_4_6), "r"(_stv8_4_7) : "memory");
                                    }
                                    {
                                        unsigned _stv8_5_0 = __float_as_uint(_tmem_load_2[8 + 0]);
                                        unsigned _stv8_5_1 = __float_as_uint(_tmem_load_2[8 + 1]);
                                        unsigned _stv8_5_2 = __float_as_uint(_tmem_load_2[8 + 2]);
                                        unsigned _stv8_5_3 = __float_as_uint(_tmem_load_2[8 + 3]);
                                        unsigned _stv8_5_4 = __float_as_uint(_tmem_load_2[8 + 4]);
                                        unsigned _stv8_5_5 = __float_as_uint(_tmem_load_2[8 + 5]);
                                        unsigned _stv8_5_6 = __float_as_uint(_tmem_load_2[8 + 6]);
                                        unsigned _stv8_5_7 = __float_as_uint(_tmem_load_2[8 + 7]);
                                        asm volatile(
                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "l"((void*)(ws + (pbase_e + (unsigned long long)(colq_u * BLOCK_M) + 8) + (0))), "r"(_stv8_5_0), "r"(_stv8_5_1), "r"(_stv8_5_2), "r"(_stv8_5_3), "r"(_stv8_5_4), "r"(_stv8_5_5), "r"(_stv8_5_6), "r"(_stv8_5_7) : "memory");
                                    }
                                }
                            }
                        } else {
                            #pragma unroll 1
                            for (int n_chunk_2 = 0; n_chunk_2 < B_HALF_N / 16; n_chunk_2++) {
                                int colp = col_half * B_HALF_N + n_chunk_2 * 16;
                                float _tmem_load_3[16];
                                tmem_ld_x16(&_tmem_load_3[0], lane_addr + colp);
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                {
                                    unsigned _stv8_6_0 = __float_as_uint(_tmem_load_3[0 + 0]);
                                    unsigned _stv8_6_1 = __float_as_uint(_tmem_load_3[0 + 1]);
                                    unsigned _stv8_6_2 = __float_as_uint(_tmem_load_3[0 + 2]);
                                    unsigned _stv8_6_3 = __float_as_uint(_tmem_load_3[0 + 3]);
                                    unsigned _stv8_6_4 = __float_as_uint(_tmem_load_3[0 + 4]);
                                    unsigned _stv8_6_5 = __float_as_uint(_tmem_load_3[0 + 5]);
                                    unsigned _stv8_6_6 = __float_as_uint(_tmem_load_3[0 + 6]);
                                    unsigned _stv8_6_7 = __float_as_uint(_tmem_load_3[0 + 7]);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(ws + (pbase_e + (unsigned long long)(colp * BLOCK_M)) + (0))), "r"(_stv8_6_0), "r"(_stv8_6_1), "r"(_stv8_6_2), "r"(_stv8_6_3), "r"(_stv8_6_4), "r"(_stv8_6_5), "r"(_stv8_6_6), "r"(_stv8_6_7) : "memory");
                                }
                                {
                                    unsigned _stv8_7_0 = __float_as_uint(_tmem_load_3[8 + 0]);
                                    unsigned _stv8_7_1 = __float_as_uint(_tmem_load_3[8 + 1]);
                                    unsigned _stv8_7_2 = __float_as_uint(_tmem_load_3[8 + 2]);
                                    unsigned _stv8_7_3 = __float_as_uint(_tmem_load_3[8 + 3]);
                                    unsigned _stv8_7_4 = __float_as_uint(_tmem_load_3[8 + 4]);
                                    unsigned _stv8_7_5 = __float_as_uint(_tmem_load_3[8 + 5]);
                                    unsigned _stv8_7_6 = __float_as_uint(_tmem_load_3[8 + 6]);
                                    unsigned _stv8_7_7 = __float_as_uint(_tmem_load_3[8 + 7]);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(ws + (pbase_e + (unsigned long long)(colp * BLOCK_M) + 8) + (0))), "r"(_stv8_7_0), "r"(_stv8_7_1), "r"(_stv8_7_2), "r"(_stv8_7_3), "r"(_stv8_7_4), "r"(_stv8_7_5), "r"(_stv8_7_6), "r"(_stv8_7_7) : "memory");
                                }
                            }
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (warp == 2) {
                            if (elect_sync()) {
                                asm volatile("fence.release.gpu;" ::: "memory");
                                int _atomic_old_0 = atomicAdd(&counters[cidx_e], 1);
                                int old_e = _atomic_old_0;
                                fix_flag[0] = old_e;
                            }
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (sym_e == 1) {
                            last_e = 0;
                            if (warp == 2) {
                                if (elect_sync()) {
                                    int32_t _relaxed_ld_0;
                                    asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_0) : "l"(counters + cidx_e) : "memory");
                                    int cnt_e = _relaxed_ld_0;
                                    #pragma unroll 1
                                    for (int _spin_e = 0; _spin_e < 4194304; _spin_e++) {
                                        if (cnt_e >= 2) {
                                            break;
                                        }
                                        int32_t _relaxed_ld_1;
                                        asm volatile("ld.relaxed.gpu.s32 %0, [%1];" : "=r"(_relaxed_ld_1) : "l"(counters + cidx_e) : "memory");
                                        cnt_e = _relaxed_ld_1;
                                    }
                                    if (cnt_e < 2) {
                                        asm volatile("trap;" ::: "memory");
                                    }
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                            }
                            asm volatile("barrier.sync 8, 256;" ::: "memory");
                        } else {
                            last_e = ((fix_flag[0] == nseg_2 - 1) ? 1 : 0);
                        }
                        if (last_e == 1) {
                            if (warp == 2) {
                                if (elect_sync()) {
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                    counters[cidx_e] = 0;
                                }
                            }
                            asm volatile("barrier.sync 8, 256;" ::: "memory");
                        }
                    }
                    if (sym_e == 1) {
                        int cidx_y = tl_2 * CTA_GROUP + cta_rank;
                        if (cls_1 < 2) {
                            int col_base_ry = bid_n_1 * BLOCK_N + col_half * B_HALF_N;
                            unsigned long long row_out_ry = (unsigned long long)global_row * (unsigned long long)NUM_EXPERTS + (unsigned long long)col_base_ry;
                            int store_ry = ((col_base_ry < NUM_EXPERTS) ? 1 : 0);
                            int col_base_ly = (bid_n_1 - R_TILES) * BLOCK_N + col_half * B_HALF_N;
                            unsigned long long row_out_ly = (unsigned long long)global_row * (unsigned long long)LATENT + (unsigned long long)col_base_ly;
                            int colh_y = col_half * B_HALF_N + seg_2 * 64;
                            float tot_y[64];
                            #pragma unroll
                            for (int e = 0; e < 64; e++) {
                                tot_y[e] = 0.0f;
                            }
                            #pragma unroll
                            for (int s2 = 0; s2 < 2; s2++) {
                                if (s2 == seg_2) {
                                    float _tmem_load_4[64];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                                        : "r"(lane_addr + colh_y));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(_tmem_load_4[32]), "=f"(_tmem_load_4[33]), "=f"(_tmem_load_4[34]), "=f"(_tmem_load_4[35]), "=f"(_tmem_load_4[36]), "=f"(_tmem_load_4[37]), "=f"(_tmem_load_4[38]), "=f"(_tmem_load_4[39]), "=f"(_tmem_load_4[40]), "=f"(_tmem_load_4[41]), "=f"(_tmem_load_4[42]), "=f"(_tmem_load_4[43]), "=f"(_tmem_load_4[44]), "=f"(_tmem_load_4[45]), "=f"(_tmem_load_4[46]), "=f"(_tmem_load_4[47]), "=f"(_tmem_load_4[48]), "=f"(_tmem_load_4[49]), "=f"(_tmem_load_4[50]), "=f"(_tmem_load_4[51]), "=f"(_tmem_load_4[52]), "=f"(_tmem_load_4[53]), "=f"(_tmem_load_4[54]), "=f"(_tmem_load_4[55]), "=f"(_tmem_load_4[56]), "=f"(_tmem_load_4[57]), "=f"(_tmem_load_4[58]), "=f"(_tmem_load_4[59]), "=f"(_tmem_load_4[60]), "=f"(_tmem_load_4[61]), "=f"(_tmem_load_4[62]), "=f"(_tmem_load_4[63])
                                        : "r"(lane_addr + colh_y + 32));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    #pragma unroll
                                    for (int e_1 = 0; e_1 < 64; e_1++) {
                                        tot_y[e_1] = tot_y[e_1] + _tmem_load_4[e_1];
                                    }
                                } else {
                                    unsigned long long obase_y = slot0_e + (unsigned long long)(s2 * (CTA_GROUP * BLOCK_M * BLOCK_N)) + (unsigned long long)(colh_y * BLOCK_M);
                                    #pragma unroll
                                    for (int q = 0; q < 8; q++) {
                                        float _vec_load_0[8];
                                        {
                                            unsigned _ldv8_8_0;
                                            unsigned _ldv8_8_1;
                                            unsigned _ldv8_8_2;
                                            unsigned _ldv8_8_3;
                                            unsigned _ldv8_8_4;
                                            unsigned _ldv8_8_5;
                                            unsigned _ldv8_8_6;
                                            unsigned _ldv8_8_7;
                                            asm volatile(
                                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                : "=r"(_ldv8_8_0), "=r"(_ldv8_8_1), "=r"(_ldv8_8_2), "=r"(_ldv8_8_3), "=r"(_ldv8_8_4), "=r"(_ldv8_8_5), "=r"(_ldv8_8_6), "=r"(_ldv8_8_7) : "l"((const void*)(ws + (obase_y + (unsigned long long)(q / 2 * (16 * BLOCK_M)) + (unsigned long long)(q % 2 * 8)) + (0))) : "memory");
                                            _vec_load_0[0 + 0] = __uint_as_float(_ldv8_8_0);
                                            _vec_load_0[0 + 1] = __uint_as_float(_ldv8_8_1);
                                            _vec_load_0[0 + 2] = __uint_as_float(_ldv8_8_2);
                                            _vec_load_0[0 + 3] = __uint_as_float(_ldv8_8_3);
                                            _vec_load_0[0 + 4] = __uint_as_float(_ldv8_8_4);
                                            _vec_load_0[0 + 5] = __uint_as_float(_ldv8_8_5);
                                            _vec_load_0[0 + 6] = __uint_as_float(_ldv8_8_6);
                                            _vec_load_0[0 + 7] = __uint_as_float(_ldv8_8_7);
                                        }
                                        #pragma unroll
                                        for (int e_2 = 0; e_2 < 8; e_2++) {
                                            tot_y[q * 8 + e_2] = tot_y[q * 8 + e_2] + _vec_load_0[e_2];
                                        }
                                    }
                                }
                            }
                            #pragma unroll
                            for (int q_1 = 0; q_1 < 4; q_1++) {
                                float outy[16];
                                #pragma unroll
                                for (int e_3 = 0; e_3 < 16; e_3++) {
                                    outy[e_3] = tot_y[q_1 * 16 + e_3];
                                }
                                if (cls_1 == 0) {
                                    if (row_ok + store_ry == 2) {
                                        {
                                            unsigned _stv8_9_0 = __float_as_uint(outy[0 + 0]);
                                            unsigned _stv8_9_1 = __float_as_uint(outy[0 + 1]);
                                            unsigned _stv8_9_2 = __float_as_uint(outy[0 + 2]);
                                            unsigned _stv8_9_3 = __float_as_uint(outy[0 + 3]);
                                            unsigned _stv8_9_4 = __float_as_uint(outy[0 + 4]);
                                            unsigned _stv8_9_5 = __float_as_uint(outy[0 + 5]);
                                            unsigned _stv8_9_6 = __float_as_uint(outy[0 + 6]);
                                            unsigned _stv8_9_7 = __float_as_uint(outy[0 + 7]);
                                            asm volatile(
                                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                :: "l"((void*)(logits + (row_out_ry + (unsigned long long)(seg_2 * 64) + (unsigned long long)(q_1 * 16)) + (0))), "r"(_stv8_9_0), "r"(_stv8_9_1), "r"(_stv8_9_2), "r"(_stv8_9_3), "r"(_stv8_9_4), "r"(_stv8_9_5), "r"(_stv8_9_6), "r"(_stv8_9_7) : "memory");
                                        }
                                        {
                                            unsigned _stv8_10_0 = __float_as_uint(outy[8 + 0]);
                                            unsigned _stv8_10_1 = __float_as_uint(outy[8 + 1]);
                                            unsigned _stv8_10_2 = __float_as_uint(outy[8 + 2]);
                                            unsigned _stv8_10_3 = __float_as_uint(outy[8 + 3]);
                                            unsigned _stv8_10_4 = __float_as_uint(outy[8 + 4]);
                                            unsigned _stv8_10_5 = __float_as_uint(outy[8 + 5]);
                                            unsigned _stv8_10_6 = __float_as_uint(outy[8 + 6]);
                                            unsigned _stv8_10_7 = __float_as_uint(outy[8 + 7]);
                                            asm volatile(
                                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                :: "l"((void*)(logits + (row_out_ry + (unsigned long long)(seg_2 * 64) + (unsigned long long)(q_1 * 16) + 8) + (0))), "r"(_stv8_10_0), "r"(_stv8_10_1), "r"(_stv8_10_2), "r"(_stv8_10_3), "r"(_stv8_10_4), "r"(_stv8_10_5), "r"(_stv8_10_6), "r"(_stv8_10_7) : "memory");
                                        }
                                    }
                                }
                                if (cls_1 == 1) {
                                    if (row_ok == 1) {
                                        {
                                            {
                                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(outy[0 + 0], outy[0 + 1]);
                                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(outy[0 + 2], outy[0 + 3]);
                                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(outy[0 + 4], outy[0 + 5]);
                                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(outy[0 + 6], outy[0 + 7]);
                                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(outy[0 + 8], outy[0 + 9]);
                                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(outy[0 + 10], outy[0 + 11]);
                                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(outy[0 + 12], outy[0 + 13]);
                                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(outy[0 + 14], outy[0 + 15]);
                                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                asm volatile(
                                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                    :: "l"((void*)(&((__nv_bfloat16*)(latent + (row_out_ly + (unsigned long long)(seg_2 * 64) + (unsigned long long)(q_1 * 16))))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if (cls_1 == 2) {
                            int col_base_sy = (bid_n_1 - R_TILES - L_TILES) * B_HALF_N + col_half * (B_HALF_N / 2);
                            unsigned long long row_out_sy = (unsigned long long)global_row * (unsigned long long)I_LOCAL + (unsigned long long)col_base_sy;
                            int colg_y = col_half * (B_HALF_N / 2) + seg_2 * 32;
                            float tot_gy[32];
                            float tot_uy[32];
                            #pragma unroll
                            for (int e_4 = 0; e_4 < 32; e_4++) {
                                tot_gy[e_4] = 0.0f;
                                tot_uy[e_4] = 0.0f;
                            }
                            #pragma unroll
                            for (int s2_1 = 0; s2_1 < 2; s2_1++) {
                                if (s2_1 == seg_2) {
                                    float _tmem_load_5[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                                        : "r"(lane_addr + colg_y));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    float _tmem_load_6[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                                        : "r"(lane_addr + B_HALF_N + colg_y));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    #pragma unroll
                                    for (int e_5 = 0; e_5 < 32; e_5++) {
                                        tot_gy[e_5] = tot_gy[e_5] + _tmem_load_5[e_5];
                                        tot_uy[e_5] = tot_uy[e_5] + _tmem_load_6[e_5];
                                    }
                                } else {
                                    unsigned long long obase_sy = slot0_e + (unsigned long long)(s2_1 * (CTA_GROUP * BLOCK_M * BLOCK_N)) + (unsigned long long)(colg_y * BLOCK_M);
                                    #pragma unroll
                                    for (int q_2 = 0; q_2 < 4; q_2++) {
                                        float _vec_load_1[8];
                                        {
                                            unsigned _ldv8_11_0;
                                            unsigned _ldv8_11_1;
                                            unsigned _ldv8_11_2;
                                            unsigned _ldv8_11_3;
                                            unsigned _ldv8_11_4;
                                            unsigned _ldv8_11_5;
                                            unsigned _ldv8_11_6;
                                            unsigned _ldv8_11_7;
                                            asm volatile(
                                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                : "=r"(_ldv8_11_0), "=r"(_ldv8_11_1), "=r"(_ldv8_11_2), "=r"(_ldv8_11_3), "=r"(_ldv8_11_4), "=r"(_ldv8_11_5), "=r"(_ldv8_11_6), "=r"(_ldv8_11_7) : "l"((const void*)(ws + (obase_sy + (unsigned long long)(q_2 / 2 * (16 * BLOCK_M)) + (unsigned long long)(q_2 % 2 * 8)) + (0))) : "memory");
                                            _vec_load_1[0 + 0] = __uint_as_float(_ldv8_11_0);
                                            _vec_load_1[0 + 1] = __uint_as_float(_ldv8_11_1);
                                            _vec_load_1[0 + 2] = __uint_as_float(_ldv8_11_2);
                                            _vec_load_1[0 + 3] = __uint_as_float(_ldv8_11_3);
                                            _vec_load_1[0 + 4] = __uint_as_float(_ldv8_11_4);
                                            _vec_load_1[0 + 5] = __uint_as_float(_ldv8_11_5);
                                            _vec_load_1[0 + 6] = __uint_as_float(_ldv8_11_6);
                                            _vec_load_1[0 + 7] = __uint_as_float(_ldv8_11_7);
                                        }
                                        float _vec_load_2[8];
                                        {
                                            unsigned _ldv8_12_0;
                                            unsigned _ldv8_12_1;
                                            unsigned _ldv8_12_2;
                                            unsigned _ldv8_12_3;
                                            unsigned _ldv8_12_4;
                                            unsigned _ldv8_12_5;
                                            unsigned _ldv8_12_6;
                                            unsigned _ldv8_12_7;
                                            asm volatile(
                                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                : "=r"(_ldv8_12_0), "=r"(_ldv8_12_1), "=r"(_ldv8_12_2), "=r"(_ldv8_12_3), "=r"(_ldv8_12_4), "=r"(_ldv8_12_5), "=r"(_ldv8_12_6), "=r"(_ldv8_12_7) : "l"((const void*)(ws + (obase_sy + (unsigned long long)(B_HALF_N * BLOCK_M) + (unsigned long long)(q_2 / 2 * (16 * BLOCK_M)) + (unsigned long long)(q_2 % 2 * 8)) + (0))) : "memory");
                                            _vec_load_2[0 + 0] = __uint_as_float(_ldv8_12_0);
                                            _vec_load_2[0 + 1] = __uint_as_float(_ldv8_12_1);
                                            _vec_load_2[0 + 2] = __uint_as_float(_ldv8_12_2);
                                            _vec_load_2[0 + 3] = __uint_as_float(_ldv8_12_3);
                                            _vec_load_2[0 + 4] = __uint_as_float(_ldv8_12_4);
                                            _vec_load_2[0 + 5] = __uint_as_float(_ldv8_12_5);
                                            _vec_load_2[0 + 6] = __uint_as_float(_ldv8_12_6);
                                            _vec_load_2[0 + 7] = __uint_as_float(_ldv8_12_7);
                                        }
                                        #pragma unroll
                                        for (int e_6 = 0; e_6 < 8; e_6++) {
                                            tot_gy[q_2 * 8 + e_6] = tot_gy[q_2 * 8 + e_6] + _vec_load_1[e_6];
                                            tot_uy[q_2 * 8 + e_6] = tot_uy[q_2 * 8 + e_6] + _vec_load_2[e_6];
                                        }
                                    }
                                }
                            }
                            #pragma unroll
                            for (int q_3 = 0; q_3 < 2; q_3++) {
                                float out_y[16];
                                #pragma unroll
                                for (int j_3 = 0; j_3 < 16; j_3++) {
                                    __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(tot_gy[q_3 * 16 + j_3]);
                                    float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                                    float gy = _cvt_f32_0;
                                    __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(tot_uy[q_3 * 16 + j_3]);
                                    float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                                    float uy = _cvt_f32_1;
                                    float _exp2_0 = approx_exp2((-gy) * 1.4426950408889634f);
                                    float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                                    float sigy = _rcp_0;
                                    float _tanh_0 = tanhf(gy * 0.25f);
                                    float ay = 4.0f * _tanh_0 * sigy;
                                    float _tanh_1 = tanhf(uy * 0.04f);
                                    float by = 25.0f * _tanh_1;
                                    out_y[j_3] = ay * by;
                                }
                                if (row_ok == 1) {
                                    {
                                        {
                                            __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_y[0 + 0], out_y[0 + 1]);
                                            unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                            __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_y[0 + 2], out_y[0 + 3]);
                                            unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                            __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_y[0 + 4], out_y[0 + 5]);
                                            unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                            __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_y[0 + 6], out_y[0 + 7]);
                                            unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                            __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_y[0 + 8], out_y[0 + 9]);
                                            unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                            __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_y[0 + 10], out_y[0 + 11]);
                                            unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                            __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_y[0 + 12], out_y[0 + 13]);
                                            unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                            __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_y[0 + 14], out_y[0 + 15]);
                                            unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                            asm volatile(
                                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                :: "l"((void*)(&((__nv_bfloat16*)(shared_act + (row_out_sy + (unsigned long long)(seg_2 * 32) + (unsigned long long)(q_3 * 16))))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                        if (warp == 2) {
                            if (elect_sync()) {
                                int _atomic_old_1 = atomicAdd(&counters[cidx_y], 1);
                                int old2_e = _atomic_old_1;
                                if (old2_e == 3) {
                                    counters[cidx_y] = 0;
                                }
                            }
                        }
                    }
                    if (last_e == 1) {
                        if (nseg_2 > 1) {
                            if (cls_1 < 2) {
                                int col_base_r = bid_n_1 * BLOCK_N + col_half * B_HALF_N;
                                unsigned long long row_out_r = (unsigned long long)global_row * (unsigned long long)NUM_EXPERTS + (unsigned long long)col_base_r;
                                int store_r = ((col_base_r < NUM_EXPERTS) ? 1 : 0);
                                int col_base_l = (bid_n_1 - R_TILES) * BLOCK_N + col_half * B_HALF_N;
                                unsigned long long row_out_l = (unsigned long long)global_row * (unsigned long long)LATENT + (unsigned long long)col_base_l;
                                #pragma unroll
                                for (int half = 0; half < 2; half++) {
                                    int colh = col_half * B_HALF_N + half * 64;
                                    float tot[64];
                                    #pragma unroll
                                    for (int e_7 = 0; e_7 < 64; e_7++) {
                                        tot[e_7] = 0.0f;
                                    }
                                    #pragma unroll 1
                                    for (int s2_2 = 0; s2_2 < nseg_2; s2_2++) {
                                        if (s2_2 == seg_2) {
                                            float _tmem_load_7[64];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                                : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
                                                : "r"(lane_addr + colh));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                                : "=f"(_tmem_load_7[32]), "=f"(_tmem_load_7[33]), "=f"(_tmem_load_7[34]), "=f"(_tmem_load_7[35]), "=f"(_tmem_load_7[36]), "=f"(_tmem_load_7[37]), "=f"(_tmem_load_7[38]), "=f"(_tmem_load_7[39]), "=f"(_tmem_load_7[40]), "=f"(_tmem_load_7[41]), "=f"(_tmem_load_7[42]), "=f"(_tmem_load_7[43]), "=f"(_tmem_load_7[44]), "=f"(_tmem_load_7[45]), "=f"(_tmem_load_7[46]), "=f"(_tmem_load_7[47]), "=f"(_tmem_load_7[48]), "=f"(_tmem_load_7[49]), "=f"(_tmem_load_7[50]), "=f"(_tmem_load_7[51]), "=f"(_tmem_load_7[52]), "=f"(_tmem_load_7[53]), "=f"(_tmem_load_7[54]), "=f"(_tmem_load_7[55]), "=f"(_tmem_load_7[56]), "=f"(_tmem_load_7[57]), "=f"(_tmem_load_7[58]), "=f"(_tmem_load_7[59]), "=f"(_tmem_load_7[60]), "=f"(_tmem_load_7[61]), "=f"(_tmem_load_7[62]), "=f"(_tmem_load_7[63])
                                                : "r"(lane_addr + colh + 32));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            #pragma unroll
                                            for (int e_8 = 0; e_8 < 64; e_8++) {
                                                tot[e_8] = tot[e_8] + _tmem_load_7[e_8];
                                            }
                                        } else {
                                            unsigned long long obase = slot0_e + (unsigned long long)s2_2 * (unsigned long long)(CTA_GROUP * BLOCK_M * BLOCK_N) + (unsigned long long)(colh * BLOCK_M);
                                            #pragma unroll
                                            for (int q_4 = 0; q_4 < 8; q_4++) {
                                                float _vec_load_3[8];
                                                {
                                                    unsigned _ldv8_13_0;
                                                    unsigned _ldv8_13_1;
                                                    unsigned _ldv8_13_2;
                                                    unsigned _ldv8_13_3;
                                                    unsigned _ldv8_13_4;
                                                    unsigned _ldv8_13_5;
                                                    unsigned _ldv8_13_6;
                                                    unsigned _ldv8_13_7;
                                                    asm volatile(
                                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                        : "=r"(_ldv8_13_0), "=r"(_ldv8_13_1), "=r"(_ldv8_13_2), "=r"(_ldv8_13_3), "=r"(_ldv8_13_4), "=r"(_ldv8_13_5), "=r"(_ldv8_13_6), "=r"(_ldv8_13_7) : "l"((const void*)(ws + (obase + (unsigned long long)(q_4 / 2 * (16 * BLOCK_M)) + (unsigned long long)(q_4 % 2 * 8)) + (0))) : "memory");
                                                    _vec_load_3[0 + 0] = __uint_as_float(_ldv8_13_0);
                                                    _vec_load_3[0 + 1] = __uint_as_float(_ldv8_13_1);
                                                    _vec_load_3[0 + 2] = __uint_as_float(_ldv8_13_2);
                                                    _vec_load_3[0 + 3] = __uint_as_float(_ldv8_13_3);
                                                    _vec_load_3[0 + 4] = __uint_as_float(_ldv8_13_4);
                                                    _vec_load_3[0 + 5] = __uint_as_float(_ldv8_13_5);
                                                    _vec_load_3[0 + 6] = __uint_as_float(_ldv8_13_6);
                                                    _vec_load_3[0 + 7] = __uint_as_float(_ldv8_13_7);
                                                }
                                                #pragma unroll
                                                for (int e_9 = 0; e_9 < 8; e_9++) {
                                                    tot[q_4 * 8 + e_9] = tot[q_4 * 8 + e_9] + _vec_load_3[e_9];
                                                }
                                            }
                                        }
                                    }
                                    #pragma unroll
                                    for (int q_5 = 0; q_5 < 4; q_5++) {
                                        float outv[16];
                                        #pragma unroll
                                        for (int e_10 = 0; e_10 < 16; e_10++) {
                                            outv[e_10] = tot[q_5 * 16 + e_10];
                                        }
                                        if (cls_1 == 0) {
                                            if (row_ok + store_r == 2) {
                                                {
                                                    unsigned _stv8_14_0 = __float_as_uint(outv[0 + 0]);
                                                    unsigned _stv8_14_1 = __float_as_uint(outv[0 + 1]);
                                                    unsigned _stv8_14_2 = __float_as_uint(outv[0 + 2]);
                                                    unsigned _stv8_14_3 = __float_as_uint(outv[0 + 3]);
                                                    unsigned _stv8_14_4 = __float_as_uint(outv[0 + 4]);
                                                    unsigned _stv8_14_5 = __float_as_uint(outv[0 + 5]);
                                                    unsigned _stv8_14_6 = __float_as_uint(outv[0 + 6]);
                                                    unsigned _stv8_14_7 = __float_as_uint(outv[0 + 7]);
                                                    asm volatile(
                                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                        :: "l"((void*)(logits + (row_out_r + (unsigned long long)(half * 64) + (unsigned long long)(q_5 * 16)) + (0))), "r"(_stv8_14_0), "r"(_stv8_14_1), "r"(_stv8_14_2), "r"(_stv8_14_3), "r"(_stv8_14_4), "r"(_stv8_14_5), "r"(_stv8_14_6), "r"(_stv8_14_7) : "memory");
                                                }
                                                {
                                                    unsigned _stv8_15_0 = __float_as_uint(outv[8 + 0]);
                                                    unsigned _stv8_15_1 = __float_as_uint(outv[8 + 1]);
                                                    unsigned _stv8_15_2 = __float_as_uint(outv[8 + 2]);
                                                    unsigned _stv8_15_3 = __float_as_uint(outv[8 + 3]);
                                                    unsigned _stv8_15_4 = __float_as_uint(outv[8 + 4]);
                                                    unsigned _stv8_15_5 = __float_as_uint(outv[8 + 5]);
                                                    unsigned _stv8_15_6 = __float_as_uint(outv[8 + 6]);
                                                    unsigned _stv8_15_7 = __float_as_uint(outv[8 + 7]);
                                                    asm volatile(
                                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                        :: "l"((void*)(logits + (row_out_r + (unsigned long long)(half * 64) + (unsigned long long)(q_5 * 16) + 8) + (0))), "r"(_stv8_15_0), "r"(_stv8_15_1), "r"(_stv8_15_2), "r"(_stv8_15_3), "r"(_stv8_15_4), "r"(_stv8_15_5), "r"(_stv8_15_6), "r"(_stv8_15_7) : "memory");
                                                }
                                            }
                                        }
                                        if (cls_1 == 1) {
                                            if (row_ok == 1) {
                                                {
                                                    {
                                                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(outv[0 + 0], outv[0 + 1]);
                                                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(outv[0 + 2], outv[0 + 3]);
                                                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(outv[0 + 4], outv[0 + 5]);
                                                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(outv[0 + 6], outv[0 + 7]);
                                                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(outv[0 + 8], outv[0 + 9]);
                                                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(outv[0 + 10], outv[0 + 11]);
                                                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(outv[0 + 12], outv[0 + 13]);
                                                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(outv[0 + 14], outv[0 + 15]);
                                                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                        asm volatile(
                                                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                            :: "l"((void*)(&((__nv_bfloat16*)(latent + (row_out_l + (unsigned long long)(half * 64) + (unsigned long long)(q_5 * 16))))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if (cls_1 == 2) {
                                int col_base_s = (bid_n_1 - R_TILES - L_TILES) * B_HALF_N + col_half * (B_HALF_N / 2);
                                unsigned long long row_out_s = (unsigned long long)global_row * (unsigned long long)I_LOCAL + (unsigned long long)col_base_s;
                                #pragma unroll
                                for (int half_1 = 0; half_1 < 2; half_1++) {
                                    int colg = col_half * (B_HALF_N / 2) + half_1 * 32;
                                    float tot_g[32];
                                    float tot_u[32];
                                    #pragma unroll
                                    for (int e_11 = 0; e_11 < 32; e_11++) {
                                        tot_g[e_11] = 0.0f;
                                        tot_u[e_11] = 0.0f;
                                    }
                                    #pragma unroll 1
                                    for (int s2_3 = 0; s2_3 < nseg_2; s2_3++) {
                                        if (s2_3 == seg_2) {
                                            float _tmem_load_8[32];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                                : "=f"(_tmem_load_8[0]), "=f"(_tmem_load_8[1]), "=f"(_tmem_load_8[2]), "=f"(_tmem_load_8[3]), "=f"(_tmem_load_8[4]), "=f"(_tmem_load_8[5]), "=f"(_tmem_load_8[6]), "=f"(_tmem_load_8[7]), "=f"(_tmem_load_8[8]), "=f"(_tmem_load_8[9]), "=f"(_tmem_load_8[10]), "=f"(_tmem_load_8[11]), "=f"(_tmem_load_8[12]), "=f"(_tmem_load_8[13]), "=f"(_tmem_load_8[14]), "=f"(_tmem_load_8[15]), "=f"(_tmem_load_8[16]), "=f"(_tmem_load_8[17]), "=f"(_tmem_load_8[18]), "=f"(_tmem_load_8[19]), "=f"(_tmem_load_8[20]), "=f"(_tmem_load_8[21]), "=f"(_tmem_load_8[22]), "=f"(_tmem_load_8[23]), "=f"(_tmem_load_8[24]), "=f"(_tmem_load_8[25]), "=f"(_tmem_load_8[26]), "=f"(_tmem_load_8[27]), "=f"(_tmem_load_8[28]), "=f"(_tmem_load_8[29]), "=f"(_tmem_load_8[30]), "=f"(_tmem_load_8[31])
                                                : "r"(lane_addr + colg));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            float _tmem_load_9[32];
                                            asm volatile(
                                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                                : "=f"(_tmem_load_9[0]), "=f"(_tmem_load_9[1]), "=f"(_tmem_load_9[2]), "=f"(_tmem_load_9[3]), "=f"(_tmem_load_9[4]), "=f"(_tmem_load_9[5]), "=f"(_tmem_load_9[6]), "=f"(_tmem_load_9[7]), "=f"(_tmem_load_9[8]), "=f"(_tmem_load_9[9]), "=f"(_tmem_load_9[10]), "=f"(_tmem_load_9[11]), "=f"(_tmem_load_9[12]), "=f"(_tmem_load_9[13]), "=f"(_tmem_load_9[14]), "=f"(_tmem_load_9[15]), "=f"(_tmem_load_9[16]), "=f"(_tmem_load_9[17]), "=f"(_tmem_load_9[18]), "=f"(_tmem_load_9[19]), "=f"(_tmem_load_9[20]), "=f"(_tmem_load_9[21]), "=f"(_tmem_load_9[22]), "=f"(_tmem_load_9[23]), "=f"(_tmem_load_9[24]), "=f"(_tmem_load_9[25]), "=f"(_tmem_load_9[26]), "=f"(_tmem_load_9[27]), "=f"(_tmem_load_9[28]), "=f"(_tmem_load_9[29]), "=f"(_tmem_load_9[30]), "=f"(_tmem_load_9[31])
                                                : "r"(lane_addr + B_HALF_N + colg));
                                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                                            #pragma unroll
                                            for (int e_12 = 0; e_12 < 32; e_12++) {
                                                tot_g[e_12] = tot_g[e_12] + _tmem_load_8[e_12];
                                                tot_u[e_12] = tot_u[e_12] + _tmem_load_9[e_12];
                                            }
                                        } else {
                                            unsigned long long obase_s = slot0_e + (unsigned long long)s2_3 * (unsigned long long)(CTA_GROUP * BLOCK_M * BLOCK_N) + (unsigned long long)(colg * BLOCK_M);
                                            #pragma unroll
                                            for (int q_6 = 0; q_6 < 4; q_6++) {
                                                float _vec_load_4[8];
                                                {
                                                    unsigned _ldv8_16_0;
                                                    unsigned _ldv8_16_1;
                                                    unsigned _ldv8_16_2;
                                                    unsigned _ldv8_16_3;
                                                    unsigned _ldv8_16_4;
                                                    unsigned _ldv8_16_5;
                                                    unsigned _ldv8_16_6;
                                                    unsigned _ldv8_16_7;
                                                    asm volatile(
                                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                        : "=r"(_ldv8_16_0), "=r"(_ldv8_16_1), "=r"(_ldv8_16_2), "=r"(_ldv8_16_3), "=r"(_ldv8_16_4), "=r"(_ldv8_16_5), "=r"(_ldv8_16_6), "=r"(_ldv8_16_7) : "l"((const void*)(ws + (obase_s + (unsigned long long)(q_6 / 2 * (16 * BLOCK_M)) + (unsigned long long)(q_6 % 2 * 8)) + (0))) : "memory");
                                                    _vec_load_4[0 + 0] = __uint_as_float(_ldv8_16_0);
                                                    _vec_load_4[0 + 1] = __uint_as_float(_ldv8_16_1);
                                                    _vec_load_4[0 + 2] = __uint_as_float(_ldv8_16_2);
                                                    _vec_load_4[0 + 3] = __uint_as_float(_ldv8_16_3);
                                                    _vec_load_4[0 + 4] = __uint_as_float(_ldv8_16_4);
                                                    _vec_load_4[0 + 5] = __uint_as_float(_ldv8_16_5);
                                                    _vec_load_4[0 + 6] = __uint_as_float(_ldv8_16_6);
                                                    _vec_load_4[0 + 7] = __uint_as_float(_ldv8_16_7);
                                                }
                                                float _vec_load_5[8];
                                                {
                                                    unsigned _ldv8_17_0;
                                                    unsigned _ldv8_17_1;
                                                    unsigned _ldv8_17_2;
                                                    unsigned _ldv8_17_3;
                                                    unsigned _ldv8_17_4;
                                                    unsigned _ldv8_17_5;
                                                    unsigned _ldv8_17_6;
                                                    unsigned _ldv8_17_7;
                                                    asm volatile(
                                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                        : "=r"(_ldv8_17_0), "=r"(_ldv8_17_1), "=r"(_ldv8_17_2), "=r"(_ldv8_17_3), "=r"(_ldv8_17_4), "=r"(_ldv8_17_5), "=r"(_ldv8_17_6), "=r"(_ldv8_17_7) : "l"((const void*)(ws + (obase_s + (unsigned long long)(B_HALF_N * BLOCK_M) + (unsigned long long)(q_6 / 2 * (16 * BLOCK_M)) + (unsigned long long)(q_6 % 2 * 8)) + (0))) : "memory");
                                                    _vec_load_5[0 + 0] = __uint_as_float(_ldv8_17_0);
                                                    _vec_load_5[0 + 1] = __uint_as_float(_ldv8_17_1);
                                                    _vec_load_5[0 + 2] = __uint_as_float(_ldv8_17_2);
                                                    _vec_load_5[0 + 3] = __uint_as_float(_ldv8_17_3);
                                                    _vec_load_5[0 + 4] = __uint_as_float(_ldv8_17_4);
                                                    _vec_load_5[0 + 5] = __uint_as_float(_ldv8_17_5);
                                                    _vec_load_5[0 + 6] = __uint_as_float(_ldv8_17_6);
                                                    _vec_load_5[0 + 7] = __uint_as_float(_ldv8_17_7);
                                                }
                                                #pragma unroll
                                                for (int e_13 = 0; e_13 < 8; e_13++) {
                                                    tot_g[q_6 * 8 + e_13] = tot_g[q_6 * 8 + e_13] + _vec_load_4[e_13];
                                                    tot_u[q_6 * 8 + e_13] = tot_u[q_6 * 8 + e_13] + _vec_load_5[e_13];
                                                }
                                            }
                                        }
                                    }
                                    #pragma unroll
                                    for (int q_7 = 0; q_7 < 2; q_7++) {
                                        float out_vals[16];
                                        #pragma unroll
                                        for (int j_4 = 0; j_4 < 16; j_4++) {
                                            __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(tot_g[q_7 * 16 + j_4]);
                                            float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                                            float g = _cvt_f32_2;
                                            __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(tot_u[q_7 * 16 + j_4]);
                                            float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                                            float u = _cvt_f32_3;
                                            float _exp2_1 = approx_exp2((-g) * 1.4426950408889634f);
                                            float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                                            float sig = _rcp_1;
                                            float _tanh_2 = tanhf(g * 0.25f);
                                            float a = 4.0f * _tanh_2 * sig;
                                            float _tanh_3 = tanhf(u * 0.04f);
                                            float b = 25.0f * _tanh_3;
                                            out_vals[j_4] = a * b;
                                        }
                                        if (row_ok == 1) {
                                            {
                                                {
                                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_vals[0 + 0], out_vals[0 + 1]);
                                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_vals[0 + 2], out_vals[0 + 3]);
                                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_vals[0 + 4], out_vals[0 + 5]);
                                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_vals[0 + 6], out_vals[0 + 7]);
                                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_vals[0 + 8], out_vals[0 + 9]);
                                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_vals[0 + 10], out_vals[0 + 11]);
                                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_vals[0 + 12], out_vals[0 + 13]);
                                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_vals[0 + 14], out_vals[0 + 15]);
                                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                    asm volatile(
                                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                        :: "l"((void*)(&((__nv_bfloat16*)(shared_act + (row_out_s + (unsigned long long)(half_1 * 32) + (unsigned long long)(q_7 * 16))))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (cls_1 == 0) {
                                int col_base_r_1 = bid_n_1 * BLOCK_N + col_half * B_HALF_N;
                                unsigned long long row_out_r_1 = (unsigned long long)global_row * (unsigned long long)NUM_EXPERTS + (unsigned long long)col_base_r_1;
                                int store_r_1 = ((col_base_r_1 < NUM_EXPERTS) ? 1 : 0);
                                #pragma unroll 1
                                for (int n_chunk_3 = 0; n_chunk_3 < B_HALF_N / 16; n_chunk_3++) {
                                    int col = n_chunk_3 * 16;
                                    float _tmem_load_10[16];
                                    tmem_ld_x16(&_tmem_load_10[0], lane_addr + col_half * B_HALF_N + col);
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    if (row_ok + store_r_1 == 2) {
                                        {
                                            unsigned _stv8_18_0 = __float_as_uint(_tmem_load_10[0 + 0]);
                                            unsigned _stv8_18_1 = __float_as_uint(_tmem_load_10[0 + 1]);
                                            unsigned _stv8_18_2 = __float_as_uint(_tmem_load_10[0 + 2]);
                                            unsigned _stv8_18_3 = __float_as_uint(_tmem_load_10[0 + 3]);
                                            unsigned _stv8_18_4 = __float_as_uint(_tmem_load_10[0 + 4]);
                                            unsigned _stv8_18_5 = __float_as_uint(_tmem_load_10[0 + 5]);
                                            unsigned _stv8_18_6 = __float_as_uint(_tmem_load_10[0 + 6]);
                                            unsigned _stv8_18_7 = __float_as_uint(_tmem_load_10[0 + 7]);
                                            asm volatile(
                                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                :: "l"((void*)(logits + (row_out_r_1 + (unsigned long long)col) + (0))), "r"(_stv8_18_0), "r"(_stv8_18_1), "r"(_stv8_18_2), "r"(_stv8_18_3), "r"(_stv8_18_4), "r"(_stv8_18_5), "r"(_stv8_18_6), "r"(_stv8_18_7) : "memory");
                                        }
                                        {
                                            unsigned _stv8_19_0 = __float_as_uint(_tmem_load_10[8 + 0]);
                                            unsigned _stv8_19_1 = __float_as_uint(_tmem_load_10[8 + 1]);
                                            unsigned _stv8_19_2 = __float_as_uint(_tmem_load_10[8 + 2]);
                                            unsigned _stv8_19_3 = __float_as_uint(_tmem_load_10[8 + 3]);
                                            unsigned _stv8_19_4 = __float_as_uint(_tmem_load_10[8 + 4]);
                                            unsigned _stv8_19_5 = __float_as_uint(_tmem_load_10[8 + 5]);
                                            unsigned _stv8_19_6 = __float_as_uint(_tmem_load_10[8 + 6]);
                                            unsigned _stv8_19_7 = __float_as_uint(_tmem_load_10[8 + 7]);
                                            asm volatile(
                                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                :: "l"((void*)(logits + (row_out_r_1 + (unsigned long long)col + 8) + (0))), "r"(_stv8_19_0), "r"(_stv8_19_1), "r"(_stv8_19_2), "r"(_stv8_19_3), "r"(_stv8_19_4), "r"(_stv8_19_5), "r"(_stv8_19_6), "r"(_stv8_19_7) : "memory");
                                        }
                                    }
                                }
                            }
                            if (cls_1 == 1) {
                                int col_base_l_1 = (bid_n_1 - R_TILES) * BLOCK_N + col_half * B_HALF_N;
                                unsigned long long row_out_l_1 = (unsigned long long)global_row * (unsigned long long)LATENT + (unsigned long long)col_base_l_1;
                                #pragma unroll 1
                                for (int n_chunk_4 = 0; n_chunk_4 < B_HALF_N / 16; n_chunk_4++) {
                                    int col_1 = n_chunk_4 * 16;
                                    float _tmem_load_11[16];
                                    tmem_ld_x16(&_tmem_load_11[0], lane_addr + col_half * B_HALF_N + col_1);
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    if (row_ok == 1) {
                                        {
                                            {
                                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(_tmem_load_11[0 + 0], _tmem_load_11[0 + 1]);
                                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(_tmem_load_11[0 + 2], _tmem_load_11[0 + 3]);
                                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(_tmem_load_11[0 + 4], _tmem_load_11[0 + 5]);
                                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(_tmem_load_11[0 + 6], _tmem_load_11[0 + 7]);
                                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(_tmem_load_11[0 + 8], _tmem_load_11[0 + 9]);
                                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(_tmem_load_11[0 + 10], _tmem_load_11[0 + 11]);
                                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(_tmem_load_11[0 + 12], _tmem_load_11[0 + 13]);
                                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(_tmem_load_11[0 + 14], _tmem_load_11[0 + 15]);
                                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                asm volatile(
                                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                    :: "l"((void*)(&((__nv_bfloat16*)(latent + (row_out_l_1 + (unsigned long long)col_1)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                            }
                                        }
                                    }
                                }
                            }
                            if (cls_1 == 2) {
                                int col_base_s_1 = (bid_n_1 - R_TILES - L_TILES) * B_HALF_N + col_half * (B_HALF_N / 2);
                                unsigned long long row_out_s_1 = (unsigned long long)global_row * (unsigned long long)I_LOCAL + (unsigned long long)col_base_s_1;
                                #pragma unroll 1
                                for (int n_chunk_5 = 0; n_chunk_5 < B_HALF_N / 32; n_chunk_5++) {
                                    int col_2 = n_chunk_5 * 16;
                                    float _tmem_load_12[16];
                                    tmem_ld_x16(&_tmem_load_12[0], lane_addr + col_half * (B_HALF_N / 2) + col_2);
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    float _tmem_load_13[16];
                                    tmem_ld_x16(&_tmem_load_13[0], lane_addr + B_HALF_N + col_half * (B_HALF_N / 2) + col_2);
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    float out_vals_1[16];
                                    #pragma unroll
                                    for (int j_5 = 0; j_5 < 16; j_5++) {
                                        __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(_tmem_load_12[j_5]);
                                        float _cvt_f32_4 = __bfloat162float(_cvt_bf16_4);
                                        float g_1 = _cvt_f32_4;
                                        __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(_tmem_load_13[j_5]);
                                        float _cvt_f32_5 = __bfloat162float(_cvt_bf16_5);
                                        float u_1 = _cvt_f32_5;
                                        float _exp2_2 = approx_exp2((-g_1) * 1.4426950408889634f);
                                        float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                                        float sig_1 = _rcp_2;
                                        float _tanh_4 = tanhf(g_1 * 0.25f);
                                        float a_1 = 4.0f * _tanh_4 * sig_1;
                                        float _tanh_5 = tanhf(u_1 * 0.04f);
                                        float b_1 = 25.0f * _tanh_5;
                                        out_vals_1[j_5] = a_1 * b_1;
                                    }
                                    if (row_ok == 1) {
                                        {
                                            {
                                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(out_vals_1[0 + 0], out_vals_1[0 + 1]);
                                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(out_vals_1[0 + 2], out_vals_1[0 + 3]);
                                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(out_vals_1[0 + 4], out_vals_1[0 + 5]);
                                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(out_vals_1[0 + 6], out_vals_1[0 + 7]);
                                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(out_vals_1[0 + 8], out_vals_1[0 + 9]);
                                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(out_vals_1[0 + 10], out_vals_1[0 + 11]);
                                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(out_vals_1[0 + 12], out_vals_1[0 + 13]);
                                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(out_vals_1[0 + 14], out_vals_1[0 + 15]);
                                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                asm volatile(
                                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                    :: "l"((void*)(&((__nv_bfloat16*)(shared_act + (row_out_s_1 + (unsigned long long)col_2)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
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
