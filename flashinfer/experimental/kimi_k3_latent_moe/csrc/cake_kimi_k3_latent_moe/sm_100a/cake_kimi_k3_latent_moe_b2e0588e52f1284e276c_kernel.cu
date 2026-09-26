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
#define TMEM_ACC_OFFSET 0
#define NUM_TMA_PIPE_STAGES 5
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_LAND_OFF 2048
#define SMEM_LAND_STAGE_BYTES 65536
#define SMEM_LAND_STRIDE 65536
#define SMEM_SMEM_A_OFF 67584
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 149504
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_BN_OFF 231424
#define SMEM_SMEM_BN_STAGE_BYTES 16384
#define SMEM_SMEM_BN_STRIDE 16384
#define SMEM_SMEM_AG_OFF 67584
#define SMEM_SMEM_AG_STAGE_BYTES 8192
#define SMEM_SMEM_AG_STRIDE 16384
#define SMEM_SMEM_AU_OFF 75776
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
kernel_cake_kimi_k3_latent_moe_b2e0588e52f1284e276c(const __grid_constant__ CUtensorMap A_R, const __grid_constant__ CUtensorMap A_L, const __grid_constant__ CUtensorMap A_S, const __grid_constant__ CUtensorMap A_2, const __grid_constant__ CUtensorMap B_1, const __grid_constant__ CUtensorMap B_2, float* __restrict__ out_r, __nv_bfloat16* __restrict__ out_l, __nv_bfloat16* __restrict__ out_s, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out, unsigned long long* __restrict__ tl, int num_tokens, int k1_off, int num_partials, float eps)
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
    #define a_empty_addr (mbar_base + 40)
    #define acc_full_addr (mbar_base + 80)
    #define acc_empty_addr (mbar_base + 88)
    #define red_bar_addr (mbar_base + 96)
    #define issue_bar_addr (mbar_base + 104)
    #define rows_bar_addr (mbar_base + 112)
    #define bn_bar_addr (mbar_base + 120)

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
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 67584);
    const int smem_a_addr = smem + 67584;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 149504);
    const int smem_b_addr = smem + 149504;
    __nv_bfloat16* smem_bn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 231424);
    const int smem_bn_addr = smem + 231424;
    __nv_bfloat16* smem_ag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 67584);
    const int smem_ag_addr = smem + 67584;
    __nv_bfloat16* smem_au = reinterpret_cast<__nv_bfloat16*>(smem_raw + 75776);
    const int smem_au_addr = smem + 75776;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // a_full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // a_empty: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'epi_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // acc_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // red_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // issue_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // rows_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // bn_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 96, 65536);
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
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
            float red[128];
            int fin = 1;
            unsigned int _phase_acc_full = 0;
            mbarrier_wait(acc_full_addr + (epi_stage) * 8, _phase_acc_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (cls == 2) {
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
                    : "r"(taddr + 128));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                    : "r"(taddr + 256));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                red[0] = _tmem_load_0[0];
                red[64] = _tmem_load_1[0];
                red[1] = _tmem_load_0[1];
                red[65] = _tmem_load_1[1];
                red[2] = _tmem_load_0[2];
                red[66] = _tmem_load_1[2];
                red[3] = _tmem_load_0[3];
                red[67] = _tmem_load_1[3];
                red[4] = _tmem_load_0[4];
                red[68] = _tmem_load_1[4];
                red[5] = _tmem_load_0[5];
                red[69] = _tmem_load_1[5];
                red[6] = _tmem_load_0[6];
                red[70] = _tmem_load_1[6];
                red[7] = _tmem_load_0[7];
                red[71] = _tmem_load_1[7];
                red[8] = _tmem_load_0[8];
                red[72] = _tmem_load_1[8];
                red[9] = _tmem_load_0[9];
                red[73] = _tmem_load_1[9];
                red[10] = _tmem_load_0[10];
                red[74] = _tmem_load_1[10];
                red[11] = _tmem_load_0[11];
                red[75] = _tmem_load_1[11];
                red[12] = _tmem_load_0[12];
                red[76] = _tmem_load_1[12];
                red[13] = _tmem_load_0[13];
                red[77] = _tmem_load_1[13];
                red[14] = _tmem_load_0[14];
                red[78] = _tmem_load_1[14];
                red[15] = _tmem_load_0[15];
                red[79] = _tmem_load_1[15];
                red[16] = _tmem_load_0[16];
                red[80] = _tmem_load_1[16];
                red[17] = _tmem_load_0[17];
                red[81] = _tmem_load_1[17];
                red[18] = _tmem_load_0[18];
                red[82] = _tmem_load_1[18];
                red[19] = _tmem_load_0[19];
                red[83] = _tmem_load_1[19];
                red[20] = _tmem_load_0[20];
                red[84] = _tmem_load_1[20];
                red[21] = _tmem_load_0[21];
                red[85] = _tmem_load_1[21];
                red[22] = _tmem_load_0[22];
                red[86] = _tmem_load_1[22];
                red[23] = _tmem_load_0[23];
                red[87] = _tmem_load_1[23];
                red[24] = _tmem_load_0[24];
                red[88] = _tmem_load_1[24];
                red[25] = _tmem_load_0[25];
                red[89] = _tmem_load_1[25];
                red[26] = _tmem_load_0[26];
                red[90] = _tmem_load_1[26];
                red[27] = _tmem_load_0[27];
                red[91] = _tmem_load_1[27];
                red[28] = _tmem_load_0[28];
                red[92] = _tmem_load_1[28];
                red[29] = _tmem_load_0[29];
                red[93] = _tmem_load_1[29];
                red[30] = _tmem_load_0[30];
                red[94] = _tmem_load_1[30];
                red[31] = _tmem_load_0[31];
                red[95] = _tmem_load_1[31];
                red[32] = _tmem_load_0[32];
                red[96] = _tmem_load_1[32];
                red[33] = _tmem_load_0[33];
                red[97] = _tmem_load_1[33];
                red[34] = _tmem_load_0[34];
                red[98] = _tmem_load_1[34];
                red[35] = _tmem_load_0[35];
                red[99] = _tmem_load_1[35];
                red[36] = _tmem_load_0[36];
                red[100] = _tmem_load_1[36];
                red[37] = _tmem_load_0[37];
                red[101] = _tmem_load_1[37];
                red[38] = _tmem_load_0[38];
                red[102] = _tmem_load_1[38];
                red[39] = _tmem_load_0[39];
                red[103] = _tmem_load_1[39];
                red[40] = _tmem_load_0[40];
                red[104] = _tmem_load_1[40];
                red[41] = _tmem_load_0[41];
                red[105] = _tmem_load_1[41];
                red[42] = _tmem_load_0[42];
                red[106] = _tmem_load_1[42];
                red[43] = _tmem_load_0[43];
                red[107] = _tmem_load_1[43];
                red[44] = _tmem_load_0[44];
                red[108] = _tmem_load_1[44];
                red[45] = _tmem_load_0[45];
                red[109] = _tmem_load_1[45];
                red[46] = _tmem_load_0[46];
                red[110] = _tmem_load_1[46];
                red[47] = _tmem_load_0[47];
                red[111] = _tmem_load_1[47];
                red[48] = _tmem_load_0[48];
                red[112] = _tmem_load_1[48];
                red[49] = _tmem_load_0[49];
                red[113] = _tmem_load_1[49];
                red[50] = _tmem_load_0[50];
                red[114] = _tmem_load_1[50];
                red[51] = _tmem_load_0[51];
                red[115] = _tmem_load_1[51];
                red[52] = _tmem_load_0[52];
                red[116] = _tmem_load_1[52];
                red[53] = _tmem_load_0[53];
                red[117] = _tmem_load_1[53];
                red[54] = _tmem_load_0[54];
                red[118] = _tmem_load_1[54];
                red[55] = _tmem_load_0[55];
                red[119] = _tmem_load_1[55];
                red[56] = _tmem_load_0[56];
                red[120] = _tmem_load_1[56];
                red[57] = _tmem_load_0[57];
                red[121] = _tmem_load_1[57];
                red[58] = _tmem_load_0[58];
                red[122] = _tmem_load_1[58];
                red[59] = _tmem_load_0[59];
                red[123] = _tmem_load_1[59];
                red[60] = _tmem_load_0[60];
                red[124] = _tmem_load_1[60];
                red[61] = _tmem_load_0[61];
                red[125] = _tmem_load_1[61];
                red[62] = _tmem_load_0[62];
                red[126] = _tmem_load_1[62];
                red[63] = _tmem_load_0[63];
                red[127] = _tmem_load_1[63];
            } else {
                float _tmem_load_2[128];
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
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[64]), "=f"(_tmem_load_2[65]), "=f"(_tmem_load_2[66]), "=f"(_tmem_load_2[67]), "=f"(_tmem_load_2[68]), "=f"(_tmem_load_2[69]), "=f"(_tmem_load_2[70]), "=f"(_tmem_load_2[71]), "=f"(_tmem_load_2[72]), "=f"(_tmem_load_2[73]), "=f"(_tmem_load_2[74]), "=f"(_tmem_load_2[75]), "=f"(_tmem_load_2[76]), "=f"(_tmem_load_2[77]), "=f"(_tmem_load_2[78]), "=f"(_tmem_load_2[79]), "=f"(_tmem_load_2[80]), "=f"(_tmem_load_2[81]), "=f"(_tmem_load_2[82]), "=f"(_tmem_load_2[83]), "=f"(_tmem_load_2[84]), "=f"(_tmem_load_2[85]), "=f"(_tmem_load_2[86]), "=f"(_tmem_load_2[87]), "=f"(_tmem_load_2[88]), "=f"(_tmem_load_2[89]), "=f"(_tmem_load_2[90]), "=f"(_tmem_load_2[91]), "=f"(_tmem_load_2[92]), "=f"(_tmem_load_2[93]), "=f"(_tmem_load_2[94]), "=f"(_tmem_load_2[95])
                    : "r"(lane_addr + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[96]), "=f"(_tmem_load_2[97]), "=f"(_tmem_load_2[98]), "=f"(_tmem_load_2[99]), "=f"(_tmem_load_2[100]), "=f"(_tmem_load_2[101]), "=f"(_tmem_load_2[102]), "=f"(_tmem_load_2[103]), "=f"(_tmem_load_2[104]), "=f"(_tmem_load_2[105]), "=f"(_tmem_load_2[106]), "=f"(_tmem_load_2[107]), "=f"(_tmem_load_2[108]), "=f"(_tmem_load_2[109]), "=f"(_tmem_load_2[110]), "=f"(_tmem_load_2[111]), "=f"(_tmem_load_2[112]), "=f"(_tmem_load_2[113]), "=f"(_tmem_load_2[114]), "=f"(_tmem_load_2[115]), "=f"(_tmem_load_2[116]), "=f"(_tmem_load_2[117]), "=f"(_tmem_load_2[118]), "=f"(_tmem_load_2[119]), "=f"(_tmem_load_2[120]), "=f"(_tmem_load_2[121]), "=f"(_tmem_load_2[122]), "=f"(_tmem_load_2[123]), "=f"(_tmem_load_2[124]), "=f"(_tmem_load_2[125]), "=f"(_tmem_load_2[126]), "=f"(_tmem_load_2[127])
                    : "r"(lane_addr + 96));
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
                red[64] = _tmem_load_2[64];
                red[65] = _tmem_load_2[65];
                red[66] = _tmem_load_2[66];
                red[67] = _tmem_load_2[67];
                red[68] = _tmem_load_2[68];
                red[69] = _tmem_load_2[69];
                red[70] = _tmem_load_2[70];
                red[71] = _tmem_load_2[71];
                red[72] = _tmem_load_2[72];
                red[73] = _tmem_load_2[73];
                red[74] = _tmem_load_2[74];
                red[75] = _tmem_load_2[75];
                red[76] = _tmem_load_2[76];
                red[77] = _tmem_load_2[77];
                red[78] = _tmem_load_2[78];
                red[79] = _tmem_load_2[79];
                red[80] = _tmem_load_2[80];
                red[81] = _tmem_load_2[81];
                red[82] = _tmem_load_2[82];
                red[83] = _tmem_load_2[83];
                red[84] = _tmem_load_2[84];
                red[85] = _tmem_load_2[85];
                red[86] = _tmem_load_2[86];
                red[87] = _tmem_load_2[87];
                red[88] = _tmem_load_2[88];
                red[89] = _tmem_load_2[89];
                red[90] = _tmem_load_2[90];
                red[91] = _tmem_load_2[91];
                red[92] = _tmem_load_2[92];
                red[93] = _tmem_load_2[93];
                red[94] = _tmem_load_2[94];
                red[95] = _tmem_load_2[95];
                red[96] = _tmem_load_2[96];
                red[97] = _tmem_load_2[97];
                red[98] = _tmem_load_2[98];
                red[99] = _tmem_load_2[99];
                red[100] = _tmem_load_2[100];
                red[101] = _tmem_load_2[101];
                red[102] = _tmem_load_2[102];
                red[103] = _tmem_load_2[103];
                red[104] = _tmem_load_2[104];
                red[105] = _tmem_load_2[105];
                red[106] = _tmem_load_2[106];
                red[107] = _tmem_load_2[107];
                red[108] = _tmem_load_2[108];
                red[109] = _tmem_load_2[109];
                red[110] = _tmem_load_2[110];
                red[111] = _tmem_load_2[111];
                red[112] = _tmem_load_2[112];
                red[113] = _tmem_load_2[113];
                red[114] = _tmem_load_2[114];
                red[115] = _tmem_load_2[115];
                red[116] = _tmem_load_2[116];
                red[117] = _tmem_load_2[117];
                red[118] = _tmem_load_2[118];
                red[119] = _tmem_load_2[119];
                red[120] = _tmem_load_2[120];
                red[121] = _tmem_load_2[121];
                red[122] = _tmem_load_2[122];
                red[123] = _tmem_load_2[123];
                red[124] = _tmem_load_2[124];
                red[125] = _tmem_load_2[125];
                red[126] = _tmem_load_2[126];
                red[127] = _tmem_load_2[127];
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
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 32768), "r"(__float_as_uint(red[64])), "r"(__float_as_uint(red[65])), "r"(__float_as_uint(red[66])), "r"(__float_as_uint(red[67])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 34816), "r"(__float_as_uint(red[68])), "r"(__float_as_uint(red[69])), "r"(__float_as_uint(red[70])), "r"(__float_as_uint(red[71])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 36864), "r"(__float_as_uint(red[72])), "r"(__float_as_uint(red[73])), "r"(__float_as_uint(red[74])), "r"(__float_as_uint(red[75])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 38912), "r"(__float_as_uint(red[76])), "r"(__float_as_uint(red[77])), "r"(__float_as_uint(red[78])), "r"(__float_as_uint(red[79])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 40960), "r"(__float_as_uint(red[80])), "r"(__float_as_uint(red[81])), "r"(__float_as_uint(red[82])), "r"(__float_as_uint(red[83])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 43008), "r"(__float_as_uint(red[84])), "r"(__float_as_uint(red[85])), "r"(__float_as_uint(red[86])), "r"(__float_as_uint(red[87])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 45056), "r"(__float_as_uint(red[88])), "r"(__float_as_uint(red[89])), "r"(__float_as_uint(red[90])), "r"(__float_as_uint(red[91])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 47104), "r"(__float_as_uint(red[92])), "r"(__float_as_uint(red[93])), "r"(__float_as_uint(red[94])), "r"(__float_as_uint(red[95])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 49152), "r"(__float_as_uint(red[96])), "r"(__float_as_uint(red[97])), "r"(__float_as_uint(red[98])), "r"(__float_as_uint(red[99])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 51200), "r"(__float_as_uint(red[100])), "r"(__float_as_uint(red[101])), "r"(__float_as_uint(red[102])), "r"(__float_as_uint(red[103])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 53248), "r"(__float_as_uint(red[104])), "r"(__float_as_uint(red[105])), "r"(__float_as_uint(red[106])), "r"(__float_as_uint(red[107])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 55296), "r"(__float_as_uint(red[108])), "r"(__float_as_uint(red[109])), "r"(__float_as_uint(red[110])), "r"(__float_as_uint(red[111])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 57344), "r"(__float_as_uint(red[112])), "r"(__float_as_uint(red[113])), "r"(__float_as_uint(red[114])), "r"(__float_as_uint(red[115])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 59392), "r"(__float_as_uint(red[116])), "r"(__float_as_uint(red[117])), "r"(__float_as_uint(red[118])), "r"(__float_as_uint(red[119])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 61440), "r"(__float_as_uint(red[120])), "r"(__float_as_uint(red[121])), "r"(__float_as_uint(red[122])), "r"(__float_as_uint(red[123])), "r"(_mapa_0) : "memory");
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_1 + 63488), "r"(__float_as_uint(red[124])), "r"(__float_as_uint(red[125])), "r"(__float_as_uint(red[126])), "r"(__float_as_uint(red[127])), "r"(_mapa_0) : "memory");
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
                red[64] = red[64] + land[(2048 + tid_1) * 4];
                red[65] = red[65] + land[(2048 + tid_1) * 4 + 1];
                red[66] = red[66] + land[(2048 + tid_1) * 4 + 2];
                red[67] = red[67] + land[(2048 + tid_1) * 4 + 3];
                red[68] = red[68] + land[(2176 + tid_1) * 4];
                red[69] = red[69] + land[(2176 + tid_1) * 4 + 1];
                red[70] = red[70] + land[(2176 + tid_1) * 4 + 2];
                red[71] = red[71] + land[(2176 + tid_1) * 4 + 3];
                red[72] = red[72] + land[(2304 + tid_1) * 4];
                red[73] = red[73] + land[(2304 + tid_1) * 4 + 1];
                red[74] = red[74] + land[(2304 + tid_1) * 4 + 2];
                red[75] = red[75] + land[(2304 + tid_1) * 4 + 3];
                red[76] = red[76] + land[(2432 + tid_1) * 4];
                red[77] = red[77] + land[(2432 + tid_1) * 4 + 1];
                red[78] = red[78] + land[(2432 + tid_1) * 4 + 2];
                red[79] = red[79] + land[(2432 + tid_1) * 4 + 3];
                red[80] = red[80] + land[(2560 + tid_1) * 4];
                red[81] = red[81] + land[(2560 + tid_1) * 4 + 1];
                red[82] = red[82] + land[(2560 + tid_1) * 4 + 2];
                red[83] = red[83] + land[(2560 + tid_1) * 4 + 3];
                red[84] = red[84] + land[(2688 + tid_1) * 4];
                red[85] = red[85] + land[(2688 + tid_1) * 4 + 1];
                red[86] = red[86] + land[(2688 + tid_1) * 4 + 2];
                red[87] = red[87] + land[(2688 + tid_1) * 4 + 3];
                red[88] = red[88] + land[(2816 + tid_1) * 4];
                red[89] = red[89] + land[(2816 + tid_1) * 4 + 1];
                red[90] = red[90] + land[(2816 + tid_1) * 4 + 2];
                red[91] = red[91] + land[(2816 + tid_1) * 4 + 3];
                red[92] = red[92] + land[(2944 + tid_1) * 4];
                red[93] = red[93] + land[(2944 + tid_1) * 4 + 1];
                red[94] = red[94] + land[(2944 + tid_1) * 4 + 2];
                red[95] = red[95] + land[(2944 + tid_1) * 4 + 3];
                red[96] = red[96] + land[(3072 + tid_1) * 4];
                red[97] = red[97] + land[(3072 + tid_1) * 4 + 1];
                red[98] = red[98] + land[(3072 + tid_1) * 4 + 2];
                red[99] = red[99] + land[(3072 + tid_1) * 4 + 3];
                red[100] = red[100] + land[(3200 + tid_1) * 4];
                red[101] = red[101] + land[(3200 + tid_1) * 4 + 1];
                red[102] = red[102] + land[(3200 + tid_1) * 4 + 2];
                red[103] = red[103] + land[(3200 + tid_1) * 4 + 3];
                red[104] = red[104] + land[(3328 + tid_1) * 4];
                red[105] = red[105] + land[(3328 + tid_1) * 4 + 1];
                red[106] = red[106] + land[(3328 + tid_1) * 4 + 2];
                red[107] = red[107] + land[(3328 + tid_1) * 4 + 3];
                red[108] = red[108] + land[(3456 + tid_1) * 4];
                red[109] = red[109] + land[(3456 + tid_1) * 4 + 1];
                red[110] = red[110] + land[(3456 + tid_1) * 4 + 2];
                red[111] = red[111] + land[(3456 + tid_1) * 4 + 3];
                red[112] = red[112] + land[(3584 + tid_1) * 4];
                red[113] = red[113] + land[(3584 + tid_1) * 4 + 1];
                red[114] = red[114] + land[(3584 + tid_1) * 4 + 2];
                red[115] = red[115] + land[(3584 + tid_1) * 4 + 3];
                red[116] = red[116] + land[(3712 + tid_1) * 4];
                red[117] = red[117] + land[(3712 + tid_1) * 4 + 1];
                red[118] = red[118] + land[(3712 + tid_1) * 4 + 2];
                red[119] = red[119] + land[(3712 + tid_1) * 4 + 3];
                red[120] = red[120] + land[(3840 + tid_1) * 4];
                red[121] = red[121] + land[(3840 + tid_1) * 4 + 1];
                red[122] = red[122] + land[(3840 + tid_1) * 4 + 2];
                red[123] = red[123] + land[(3840 + tid_1) * 4 + 3];
                red[124] = red[124] + land[(3968 + tid_1) * 4];
                red[125] = red[125] + land[(3968 + tid_1) * 4 + 1];
                red[126] = red[126] + land[(3968 + tid_1) * 4 + 2];
                red[127] = red[127] + land[(3968 + tid_1) * 4 + 3];
            }
            if (fin == 1) {
                if (cls == 2) {
                    int tok = lane_pair * 2;
                    int frow = row_base + ((0) ? 8 : 0);
                    if (tok < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_132 = __float2bfloat16(red[0]);
                        float _cvt_f32_132 = __bfloat162float(_cvt_bf16_132);
                        float gk = _cvt_f32_132;
                        __nv_bfloat16 _cvt_bf16_133 = __float2bfloat16(red[64]);
                        float _cvt_f32_133 = __bfloat162float(_cvt_bf16_133);
                        float uk = _cvt_f32_133;
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
                        __nv_bfloat16 _cvt_bf16_134 = __float2bfloat16(red[1]);
                        float _cvt_f32_134 = __bfloat162float(_cvt_bf16_134);
                        float gk_1 = _cvt_f32_134;
                        __nv_bfloat16 _cvt_bf16_135 = __float2bfloat16(red[65]);
                        float _cvt_f32_135 = __bfloat162float(_cvt_bf16_135);
                        float uk_1 = _cvt_f32_135;
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
                        __nv_bfloat16 _cvt_bf16_136 = __float2bfloat16(red[2]);
                        float _cvt_f32_136 = __bfloat162float(_cvt_bf16_136);
                        float gk_2 = _cvt_f32_136;
                        __nv_bfloat16 _cvt_bf16_137 = __float2bfloat16(red[66]);
                        float _cvt_f32_137 = __bfloat162float(_cvt_bf16_137);
                        float uk_2 = _cvt_f32_137;
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
                        __nv_bfloat16 _cvt_bf16_138 = __float2bfloat16(red[3]);
                        float _cvt_f32_138 = __bfloat162float(_cvt_bf16_138);
                        float gk_3 = _cvt_f32_138;
                        __nv_bfloat16 _cvt_bf16_139 = __float2bfloat16(red[67]);
                        float _cvt_f32_139 = __bfloat162float(_cvt_bf16_139);
                        float uk_3 = _cvt_f32_139;
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
                        __nv_bfloat16 _cvt_bf16_140 = __float2bfloat16(red[4]);
                        float _cvt_f32_140 = __bfloat162float(_cvt_bf16_140);
                        float gk_4 = _cvt_f32_140;
                        __nv_bfloat16 _cvt_bf16_141 = __float2bfloat16(red[68]);
                        float _cvt_f32_141 = __bfloat162float(_cvt_bf16_141);
                        float uk_4 = _cvt_f32_141;
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
                        __nv_bfloat16 _cvt_bf16_142 = __float2bfloat16(red[5]);
                        float _cvt_f32_142 = __bfloat162float(_cvt_bf16_142);
                        float gk_5 = _cvt_f32_142;
                        __nv_bfloat16 _cvt_bf16_143 = __float2bfloat16(red[69]);
                        float _cvt_f32_143 = __bfloat162float(_cvt_bf16_143);
                        float uk_5 = _cvt_f32_143;
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
                        __nv_bfloat16 _cvt_bf16_144 = __float2bfloat16(red[6]);
                        float _cvt_f32_144 = __bfloat162float(_cvt_bf16_144);
                        float gk_6 = _cvt_f32_144;
                        __nv_bfloat16 _cvt_bf16_145 = __float2bfloat16(red[70]);
                        float _cvt_f32_145 = __bfloat162float(_cvt_bf16_145);
                        float uk_6 = _cvt_f32_145;
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
                        __nv_bfloat16 _cvt_bf16_146 = __float2bfloat16(red[7]);
                        float _cvt_f32_146 = __bfloat162float(_cvt_bf16_146);
                        float gk_7 = _cvt_f32_146;
                        __nv_bfloat16 _cvt_bf16_147 = __float2bfloat16(red[71]);
                        float _cvt_f32_147 = __bfloat162float(_cvt_bf16_147);
                        float uk_7 = _cvt_f32_147;
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
                        __nv_bfloat16 _cvt_bf16_148 = __float2bfloat16(red[8]);
                        float _cvt_f32_148 = __bfloat162float(_cvt_bf16_148);
                        float gk_8 = _cvt_f32_148;
                        __nv_bfloat16 _cvt_bf16_149 = __float2bfloat16(red[72]);
                        float _cvt_f32_149 = __bfloat162float(_cvt_bf16_149);
                        float uk_8 = _cvt_f32_149;
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
                        __nv_bfloat16 _cvt_bf16_150 = __float2bfloat16(red[9]);
                        float _cvt_f32_150 = __bfloat162float(_cvt_bf16_150);
                        float gk_9 = _cvt_f32_150;
                        __nv_bfloat16 _cvt_bf16_151 = __float2bfloat16(red[73]);
                        float _cvt_f32_151 = __bfloat162float(_cvt_bf16_151);
                        float uk_9 = _cvt_f32_151;
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
                        __nv_bfloat16 _cvt_bf16_152 = __float2bfloat16(red[10]);
                        float _cvt_f32_152 = __bfloat162float(_cvt_bf16_152);
                        float gk_10 = _cvt_f32_152;
                        __nv_bfloat16 _cvt_bf16_153 = __float2bfloat16(red[74]);
                        float _cvt_f32_153 = __bfloat162float(_cvt_bf16_153);
                        float uk_10 = _cvt_f32_153;
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
                        __nv_bfloat16 _cvt_bf16_154 = __float2bfloat16(red[11]);
                        float _cvt_f32_154 = __bfloat162float(_cvt_bf16_154);
                        float gk_11 = _cvt_f32_154;
                        __nv_bfloat16 _cvt_bf16_155 = __float2bfloat16(red[75]);
                        float _cvt_f32_155 = __bfloat162float(_cvt_bf16_155);
                        float uk_11 = _cvt_f32_155;
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
                        __nv_bfloat16 _cvt_bf16_156 = __float2bfloat16(red[12]);
                        float _cvt_f32_156 = __bfloat162float(_cvt_bf16_156);
                        float gk_12 = _cvt_f32_156;
                        __nv_bfloat16 _cvt_bf16_157 = __float2bfloat16(red[76]);
                        float _cvt_f32_157 = __bfloat162float(_cvt_bf16_157);
                        float uk_12 = _cvt_f32_157;
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
                        __nv_bfloat16 _cvt_bf16_158 = __float2bfloat16(red[13]);
                        float _cvt_f32_158 = __bfloat162float(_cvt_bf16_158);
                        float gk_13 = _cvt_f32_158;
                        __nv_bfloat16 _cvt_bf16_159 = __float2bfloat16(red[77]);
                        float _cvt_f32_159 = __bfloat162float(_cvt_bf16_159);
                        float uk_13 = _cvt_f32_159;
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
                        __nv_bfloat16 _cvt_bf16_160 = __float2bfloat16(red[14]);
                        float _cvt_f32_160 = __bfloat162float(_cvt_bf16_160);
                        float gk_14 = _cvt_f32_160;
                        __nv_bfloat16 _cvt_bf16_161 = __float2bfloat16(red[78]);
                        float _cvt_f32_161 = __bfloat162float(_cvt_bf16_161);
                        float uk_14 = _cvt_f32_161;
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
                        __nv_bfloat16 _cvt_bf16_162 = __float2bfloat16(red[15]);
                        float _cvt_f32_162 = __bfloat162float(_cvt_bf16_162);
                        float gk_15 = _cvt_f32_162;
                        __nv_bfloat16 _cvt_bf16_163 = __float2bfloat16(red[79]);
                        float _cvt_f32_163 = __bfloat162float(_cvt_bf16_163);
                        float uk_15 = _cvt_f32_163;
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
                        __nv_bfloat16 _cvt_bf16_164 = __float2bfloat16(red[16]);
                        float _cvt_f32_164 = __bfloat162float(_cvt_bf16_164);
                        float gk_16 = _cvt_f32_164;
                        __nv_bfloat16 _cvt_bf16_165 = __float2bfloat16(red[80]);
                        float _cvt_f32_165 = __bfloat162float(_cvt_bf16_165);
                        float uk_16 = _cvt_f32_165;
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
                        __nv_bfloat16 _cvt_bf16_166 = __float2bfloat16(red[17]);
                        float _cvt_f32_166 = __bfloat162float(_cvt_bf16_166);
                        float gk_17 = _cvt_f32_166;
                        __nv_bfloat16 _cvt_bf16_167 = __float2bfloat16(red[81]);
                        float _cvt_f32_167 = __bfloat162float(_cvt_bf16_167);
                        float uk_17 = _cvt_f32_167;
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
                        __nv_bfloat16 _cvt_bf16_168 = __float2bfloat16(red[18]);
                        float _cvt_f32_168 = __bfloat162float(_cvt_bf16_168);
                        float gk_18 = _cvt_f32_168;
                        __nv_bfloat16 _cvt_bf16_169 = __float2bfloat16(red[82]);
                        float _cvt_f32_169 = __bfloat162float(_cvt_bf16_169);
                        float uk_18 = _cvt_f32_169;
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
                        __nv_bfloat16 _cvt_bf16_170 = __float2bfloat16(red[19]);
                        float _cvt_f32_170 = __bfloat162float(_cvt_bf16_170);
                        float gk_19 = _cvt_f32_170;
                        __nv_bfloat16 _cvt_bf16_171 = __float2bfloat16(red[83]);
                        float _cvt_f32_171 = __bfloat162float(_cvt_bf16_171);
                        float uk_19 = _cvt_f32_171;
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
                        __nv_bfloat16 _cvt_bf16_172 = __float2bfloat16(red[20]);
                        float _cvt_f32_172 = __bfloat162float(_cvt_bf16_172);
                        float gk_20 = _cvt_f32_172;
                        __nv_bfloat16 _cvt_bf16_173 = __float2bfloat16(red[84]);
                        float _cvt_f32_173 = __bfloat162float(_cvt_bf16_173);
                        float uk_20 = _cvt_f32_173;
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
                        __nv_bfloat16 _cvt_bf16_174 = __float2bfloat16(red[21]);
                        float _cvt_f32_174 = __bfloat162float(_cvt_bf16_174);
                        float gk_21 = _cvt_f32_174;
                        __nv_bfloat16 _cvt_bf16_175 = __float2bfloat16(red[85]);
                        float _cvt_f32_175 = __bfloat162float(_cvt_bf16_175);
                        float uk_21 = _cvt_f32_175;
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
                        __nv_bfloat16 _cvt_bf16_176 = __float2bfloat16(red[22]);
                        float _cvt_f32_176 = __bfloat162float(_cvt_bf16_176);
                        float gk_22 = _cvt_f32_176;
                        __nv_bfloat16 _cvt_bf16_177 = __float2bfloat16(red[86]);
                        float _cvt_f32_177 = __bfloat162float(_cvt_bf16_177);
                        float uk_22 = _cvt_f32_177;
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
                        __nv_bfloat16 _cvt_bf16_178 = __float2bfloat16(red[23]);
                        float _cvt_f32_178 = __bfloat162float(_cvt_bf16_178);
                        float gk_23 = _cvt_f32_178;
                        __nv_bfloat16 _cvt_bf16_179 = __float2bfloat16(red[87]);
                        float _cvt_f32_179 = __bfloat162float(_cvt_bf16_179);
                        float uk_23 = _cvt_f32_179;
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
                        __nv_bfloat16 _cvt_bf16_180 = __float2bfloat16(red[24]);
                        float _cvt_f32_180 = __bfloat162float(_cvt_bf16_180);
                        float gk_24 = _cvt_f32_180;
                        __nv_bfloat16 _cvt_bf16_181 = __float2bfloat16(red[88]);
                        float _cvt_f32_181 = __bfloat162float(_cvt_bf16_181);
                        float uk_24 = _cvt_f32_181;
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
                        __nv_bfloat16 _cvt_bf16_182 = __float2bfloat16(red[25]);
                        float _cvt_f32_182 = __bfloat162float(_cvt_bf16_182);
                        float gk_25 = _cvt_f32_182;
                        __nv_bfloat16 _cvt_bf16_183 = __float2bfloat16(red[89]);
                        float _cvt_f32_183 = __bfloat162float(_cvt_bf16_183);
                        float uk_25 = _cvt_f32_183;
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
                        __nv_bfloat16 _cvt_bf16_184 = __float2bfloat16(red[26]);
                        float _cvt_f32_184 = __bfloat162float(_cvt_bf16_184);
                        float gk_26 = _cvt_f32_184;
                        __nv_bfloat16 _cvt_bf16_185 = __float2bfloat16(red[90]);
                        float _cvt_f32_185 = __bfloat162float(_cvt_bf16_185);
                        float uk_26 = _cvt_f32_185;
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
                        __nv_bfloat16 _cvt_bf16_186 = __float2bfloat16(red[27]);
                        float _cvt_f32_186 = __bfloat162float(_cvt_bf16_186);
                        float gk_27 = _cvt_f32_186;
                        __nv_bfloat16 _cvt_bf16_187 = __float2bfloat16(red[91]);
                        float _cvt_f32_187 = __bfloat162float(_cvt_bf16_187);
                        float uk_27 = _cvt_f32_187;
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
                        __nv_bfloat16 _cvt_bf16_188 = __float2bfloat16(red[28]);
                        float _cvt_f32_188 = __bfloat162float(_cvt_bf16_188);
                        float gk_28 = _cvt_f32_188;
                        __nv_bfloat16 _cvt_bf16_189 = __float2bfloat16(red[92]);
                        float _cvt_f32_189 = __bfloat162float(_cvt_bf16_189);
                        float uk_28 = _cvt_f32_189;
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
                        __nv_bfloat16 _cvt_bf16_190 = __float2bfloat16(red[29]);
                        float _cvt_f32_190 = __bfloat162float(_cvt_bf16_190);
                        float gk_29 = _cvt_f32_190;
                        __nv_bfloat16 _cvt_bf16_191 = __float2bfloat16(red[93]);
                        float _cvt_f32_191 = __bfloat162float(_cvt_bf16_191);
                        float uk_29 = _cvt_f32_191;
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
                        __nv_bfloat16 _cvt_bf16_192 = __float2bfloat16(red[30]);
                        float _cvt_f32_192 = __bfloat162float(_cvt_bf16_192);
                        float gk_30 = _cvt_f32_192;
                        __nv_bfloat16 _cvt_bf16_193 = __float2bfloat16(red[94]);
                        float _cvt_f32_193 = __bfloat162float(_cvt_bf16_193);
                        float uk_30 = _cvt_f32_193;
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
                        __nv_bfloat16 _cvt_bf16_194 = __float2bfloat16(red[31]);
                        float _cvt_f32_194 = __bfloat162float(_cvt_bf16_194);
                        float gk_31 = _cvt_f32_194;
                        __nv_bfloat16 _cvt_bf16_195 = __float2bfloat16(red[95]);
                        float _cvt_f32_195 = __bfloat162float(_cvt_bf16_195);
                        float uk_31 = _cvt_f32_195;
                        float _exp2_31 = approx_exp2((-gk_31) * 1.4426950408889634f);
                        float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                        float sig_31 = _rcp_31;
                        float _tanh_62 = tanhf(gk_31 * 0.25f);
                        float aa_31 = 4.0f * _tanh_62 * sig_31;
                        float _tanh_63 = tanhf(uk_31 * 0.04f);
                        float bb_31 = 25.0f * _tanh_63;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_60 * 768 + row_s + frow_61)) + (0)) = __float2bfloat16_rn(aa_31 * bb_31);
                    }
                    int tok_62 = 64 + lane_pair * 2;
                    int frow_63 = row_base + ((0) ? 8 : 0);
                    if (tok_62 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_196 = __float2bfloat16(red[32]);
                        float _cvt_f32_196 = __bfloat162float(_cvt_bf16_196);
                        float gk_32 = _cvt_f32_196;
                        __nv_bfloat16 _cvt_bf16_197 = __float2bfloat16(red[96]);
                        float _cvt_f32_197 = __bfloat162float(_cvt_bf16_197);
                        float uk_32 = _cvt_f32_197;
                        float _exp2_32 = approx_exp2((-gk_32) * 1.4426950408889634f);
                        float _rcp_32 = approx_rcp(1.0f + _exp2_32);
                        float sig_32 = _rcp_32;
                        float _tanh_64 = tanhf(gk_32 * 0.25f);
                        float aa_32 = 4.0f * _tanh_64 * sig_32;
                        float _tanh_65 = tanhf(uk_32 * 0.04f);
                        float bb_32 = 25.0f * _tanh_65;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_62 * 768 + row_s + frow_63)) + (0)) = __float2bfloat16_rn(aa_32 * bb_32);
                    }
                    int tok_64 = 64 + lane_pair * 2 + 1;
                    int frow_65 = row_base + ((0) ? 8 : 0);
                    if (tok_64 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_198 = __float2bfloat16(red[33]);
                        float _cvt_f32_198 = __bfloat162float(_cvt_bf16_198);
                        float gk_33 = _cvt_f32_198;
                        __nv_bfloat16 _cvt_bf16_199 = __float2bfloat16(red[97]);
                        float _cvt_f32_199 = __bfloat162float(_cvt_bf16_199);
                        float uk_33 = _cvt_f32_199;
                        float _exp2_33 = approx_exp2((-gk_33) * 1.4426950408889634f);
                        float _rcp_33 = approx_rcp(1.0f + _exp2_33);
                        float sig_33 = _rcp_33;
                        float _tanh_66 = tanhf(gk_33 * 0.25f);
                        float aa_33 = 4.0f * _tanh_66 * sig_33;
                        float _tanh_67 = tanhf(uk_33 * 0.04f);
                        float bb_33 = 25.0f * _tanh_67;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_64 * 768 + row_s + frow_65)) + (0)) = __float2bfloat16_rn(aa_33 * bb_33);
                    }
                    int tok_66 = 64 + lane_pair * 2;
                    int frow_67 = row_base + ((1) ? 8 : 0);
                    if (tok_66 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_200 = __float2bfloat16(red[34]);
                        float _cvt_f32_200 = __bfloat162float(_cvt_bf16_200);
                        float gk_34 = _cvt_f32_200;
                        __nv_bfloat16 _cvt_bf16_201 = __float2bfloat16(red[98]);
                        float _cvt_f32_201 = __bfloat162float(_cvt_bf16_201);
                        float uk_34 = _cvt_f32_201;
                        float _exp2_34 = approx_exp2((-gk_34) * 1.4426950408889634f);
                        float _rcp_34 = approx_rcp(1.0f + _exp2_34);
                        float sig_34 = _rcp_34;
                        float _tanh_68 = tanhf(gk_34 * 0.25f);
                        float aa_34 = 4.0f * _tanh_68 * sig_34;
                        float _tanh_69 = tanhf(uk_34 * 0.04f);
                        float bb_34 = 25.0f * _tanh_69;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_66 * 768 + row_s + frow_67)) + (0)) = __float2bfloat16_rn(aa_34 * bb_34);
                    }
                    int tok_68 = 64 + lane_pair * 2 + 1;
                    int frow_69 = row_base + ((1) ? 8 : 0);
                    if (tok_68 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_202 = __float2bfloat16(red[35]);
                        float _cvt_f32_202 = __bfloat162float(_cvt_bf16_202);
                        float gk_35 = _cvt_f32_202;
                        __nv_bfloat16 _cvt_bf16_203 = __float2bfloat16(red[99]);
                        float _cvt_f32_203 = __bfloat162float(_cvt_bf16_203);
                        float uk_35 = _cvt_f32_203;
                        float _exp2_35 = approx_exp2((-gk_35) * 1.4426950408889634f);
                        float _rcp_35 = approx_rcp(1.0f + _exp2_35);
                        float sig_35 = _rcp_35;
                        float _tanh_70 = tanhf(gk_35 * 0.25f);
                        float aa_35 = 4.0f * _tanh_70 * sig_35;
                        float _tanh_71 = tanhf(uk_35 * 0.04f);
                        float bb_35 = 25.0f * _tanh_71;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_68 * 768 + row_s + frow_69)) + (0)) = __float2bfloat16_rn(aa_35 * bb_35);
                    }
                    int tok_70 = 72 + lane_pair * 2;
                    int frow_71 = row_base + ((0) ? 8 : 0);
                    if (tok_70 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_204 = __float2bfloat16(red[36]);
                        float _cvt_f32_204 = __bfloat162float(_cvt_bf16_204);
                        float gk_36 = _cvt_f32_204;
                        __nv_bfloat16 _cvt_bf16_205 = __float2bfloat16(red[100]);
                        float _cvt_f32_205 = __bfloat162float(_cvt_bf16_205);
                        float uk_36 = _cvt_f32_205;
                        float _exp2_36 = approx_exp2((-gk_36) * 1.4426950408889634f);
                        float _rcp_36 = approx_rcp(1.0f + _exp2_36);
                        float sig_36 = _rcp_36;
                        float _tanh_72 = tanhf(gk_36 * 0.25f);
                        float aa_36 = 4.0f * _tanh_72 * sig_36;
                        float _tanh_73 = tanhf(uk_36 * 0.04f);
                        float bb_36 = 25.0f * _tanh_73;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_70 * 768 + row_s + frow_71)) + (0)) = __float2bfloat16_rn(aa_36 * bb_36);
                    }
                    int tok_72 = 72 + lane_pair * 2 + 1;
                    int frow_73 = row_base + ((0) ? 8 : 0);
                    if (tok_72 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_206 = __float2bfloat16(red[37]);
                        float _cvt_f32_206 = __bfloat162float(_cvt_bf16_206);
                        float gk_37 = _cvt_f32_206;
                        __nv_bfloat16 _cvt_bf16_207 = __float2bfloat16(red[101]);
                        float _cvt_f32_207 = __bfloat162float(_cvt_bf16_207);
                        float uk_37 = _cvt_f32_207;
                        float _exp2_37 = approx_exp2((-gk_37) * 1.4426950408889634f);
                        float _rcp_37 = approx_rcp(1.0f + _exp2_37);
                        float sig_37 = _rcp_37;
                        float _tanh_74 = tanhf(gk_37 * 0.25f);
                        float aa_37 = 4.0f * _tanh_74 * sig_37;
                        float _tanh_75 = tanhf(uk_37 * 0.04f);
                        float bb_37 = 25.0f * _tanh_75;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_72 * 768 + row_s + frow_73)) + (0)) = __float2bfloat16_rn(aa_37 * bb_37);
                    }
                    int tok_74 = 72 + lane_pair * 2;
                    int frow_75 = row_base + ((1) ? 8 : 0);
                    if (tok_74 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_208 = __float2bfloat16(red[38]);
                        float _cvt_f32_208 = __bfloat162float(_cvt_bf16_208);
                        float gk_38 = _cvt_f32_208;
                        __nv_bfloat16 _cvt_bf16_209 = __float2bfloat16(red[102]);
                        float _cvt_f32_209 = __bfloat162float(_cvt_bf16_209);
                        float uk_38 = _cvt_f32_209;
                        float _exp2_38 = approx_exp2((-gk_38) * 1.4426950408889634f);
                        float _rcp_38 = approx_rcp(1.0f + _exp2_38);
                        float sig_38 = _rcp_38;
                        float _tanh_76 = tanhf(gk_38 * 0.25f);
                        float aa_38 = 4.0f * _tanh_76 * sig_38;
                        float _tanh_77 = tanhf(uk_38 * 0.04f);
                        float bb_38 = 25.0f * _tanh_77;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_74 * 768 + row_s + frow_75)) + (0)) = __float2bfloat16_rn(aa_38 * bb_38);
                    }
                    int tok_76 = 72 + lane_pair * 2 + 1;
                    int frow_77 = row_base + ((1) ? 8 : 0);
                    if (tok_76 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_210 = __float2bfloat16(red[39]);
                        float _cvt_f32_210 = __bfloat162float(_cvt_bf16_210);
                        float gk_39 = _cvt_f32_210;
                        __nv_bfloat16 _cvt_bf16_211 = __float2bfloat16(red[103]);
                        float _cvt_f32_211 = __bfloat162float(_cvt_bf16_211);
                        float uk_39 = _cvt_f32_211;
                        float _exp2_39 = approx_exp2((-gk_39) * 1.4426950408889634f);
                        float _rcp_39 = approx_rcp(1.0f + _exp2_39);
                        float sig_39 = _rcp_39;
                        float _tanh_78 = tanhf(gk_39 * 0.25f);
                        float aa_39 = 4.0f * _tanh_78 * sig_39;
                        float _tanh_79 = tanhf(uk_39 * 0.04f);
                        float bb_39 = 25.0f * _tanh_79;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_76 * 768 + row_s + frow_77)) + (0)) = __float2bfloat16_rn(aa_39 * bb_39);
                    }
                    int tok_78 = 80 + lane_pair * 2;
                    int frow_79 = row_base + ((0) ? 8 : 0);
                    if (tok_78 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_212 = __float2bfloat16(red[40]);
                        float _cvt_f32_212 = __bfloat162float(_cvt_bf16_212);
                        float gk_40 = _cvt_f32_212;
                        __nv_bfloat16 _cvt_bf16_213 = __float2bfloat16(red[104]);
                        float _cvt_f32_213 = __bfloat162float(_cvt_bf16_213);
                        float uk_40 = _cvt_f32_213;
                        float _exp2_40 = approx_exp2((-gk_40) * 1.4426950408889634f);
                        float _rcp_40 = approx_rcp(1.0f + _exp2_40);
                        float sig_40 = _rcp_40;
                        float _tanh_80 = tanhf(gk_40 * 0.25f);
                        float aa_40 = 4.0f * _tanh_80 * sig_40;
                        float _tanh_81 = tanhf(uk_40 * 0.04f);
                        float bb_40 = 25.0f * _tanh_81;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_78 * 768 + row_s + frow_79)) + (0)) = __float2bfloat16_rn(aa_40 * bb_40);
                    }
                    int tok_80 = 80 + lane_pair * 2 + 1;
                    int frow_81 = row_base + ((0) ? 8 : 0);
                    if (tok_80 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_214 = __float2bfloat16(red[41]);
                        float _cvt_f32_214 = __bfloat162float(_cvt_bf16_214);
                        float gk_41 = _cvt_f32_214;
                        __nv_bfloat16 _cvt_bf16_215 = __float2bfloat16(red[105]);
                        float _cvt_f32_215 = __bfloat162float(_cvt_bf16_215);
                        float uk_41 = _cvt_f32_215;
                        float _exp2_41 = approx_exp2((-gk_41) * 1.4426950408889634f);
                        float _rcp_41 = approx_rcp(1.0f + _exp2_41);
                        float sig_41 = _rcp_41;
                        float _tanh_82 = tanhf(gk_41 * 0.25f);
                        float aa_41 = 4.0f * _tanh_82 * sig_41;
                        float _tanh_83 = tanhf(uk_41 * 0.04f);
                        float bb_41 = 25.0f * _tanh_83;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_80 * 768 + row_s + frow_81)) + (0)) = __float2bfloat16_rn(aa_41 * bb_41);
                    }
                    int tok_82 = 80 + lane_pair * 2;
                    int frow_83 = row_base + ((1) ? 8 : 0);
                    if (tok_82 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_216 = __float2bfloat16(red[42]);
                        float _cvt_f32_216 = __bfloat162float(_cvt_bf16_216);
                        float gk_42 = _cvt_f32_216;
                        __nv_bfloat16 _cvt_bf16_217 = __float2bfloat16(red[106]);
                        float _cvt_f32_217 = __bfloat162float(_cvt_bf16_217);
                        float uk_42 = _cvt_f32_217;
                        float _exp2_42 = approx_exp2((-gk_42) * 1.4426950408889634f);
                        float _rcp_42 = approx_rcp(1.0f + _exp2_42);
                        float sig_42 = _rcp_42;
                        float _tanh_84 = tanhf(gk_42 * 0.25f);
                        float aa_42 = 4.0f * _tanh_84 * sig_42;
                        float _tanh_85 = tanhf(uk_42 * 0.04f);
                        float bb_42 = 25.0f * _tanh_85;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_82 * 768 + row_s + frow_83)) + (0)) = __float2bfloat16_rn(aa_42 * bb_42);
                    }
                    int tok_84 = 80 + lane_pair * 2 + 1;
                    int frow_85 = row_base + ((1) ? 8 : 0);
                    if (tok_84 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_218 = __float2bfloat16(red[43]);
                        float _cvt_f32_218 = __bfloat162float(_cvt_bf16_218);
                        float gk_43 = _cvt_f32_218;
                        __nv_bfloat16 _cvt_bf16_219 = __float2bfloat16(red[107]);
                        float _cvt_f32_219 = __bfloat162float(_cvt_bf16_219);
                        float uk_43 = _cvt_f32_219;
                        float _exp2_43 = approx_exp2((-gk_43) * 1.4426950408889634f);
                        float _rcp_43 = approx_rcp(1.0f + _exp2_43);
                        float sig_43 = _rcp_43;
                        float _tanh_86 = tanhf(gk_43 * 0.25f);
                        float aa_43 = 4.0f * _tanh_86 * sig_43;
                        float _tanh_87 = tanhf(uk_43 * 0.04f);
                        float bb_43 = 25.0f * _tanh_87;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_84 * 768 + row_s + frow_85)) + (0)) = __float2bfloat16_rn(aa_43 * bb_43);
                    }
                    int tok_86 = 88 + lane_pair * 2;
                    int frow_87 = row_base + ((0) ? 8 : 0);
                    if (tok_86 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_220 = __float2bfloat16(red[44]);
                        float _cvt_f32_220 = __bfloat162float(_cvt_bf16_220);
                        float gk_44 = _cvt_f32_220;
                        __nv_bfloat16 _cvt_bf16_221 = __float2bfloat16(red[108]);
                        float _cvt_f32_221 = __bfloat162float(_cvt_bf16_221);
                        float uk_44 = _cvt_f32_221;
                        float _exp2_44 = approx_exp2((-gk_44) * 1.4426950408889634f);
                        float _rcp_44 = approx_rcp(1.0f + _exp2_44);
                        float sig_44 = _rcp_44;
                        float _tanh_88 = tanhf(gk_44 * 0.25f);
                        float aa_44 = 4.0f * _tanh_88 * sig_44;
                        float _tanh_89 = tanhf(uk_44 * 0.04f);
                        float bb_44 = 25.0f * _tanh_89;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_86 * 768 + row_s + frow_87)) + (0)) = __float2bfloat16_rn(aa_44 * bb_44);
                    }
                    int tok_88 = 88 + lane_pair * 2 + 1;
                    int frow_89 = row_base + ((0) ? 8 : 0);
                    if (tok_88 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_222 = __float2bfloat16(red[45]);
                        float _cvt_f32_222 = __bfloat162float(_cvt_bf16_222);
                        float gk_45 = _cvt_f32_222;
                        __nv_bfloat16 _cvt_bf16_223 = __float2bfloat16(red[109]);
                        float _cvt_f32_223 = __bfloat162float(_cvt_bf16_223);
                        float uk_45 = _cvt_f32_223;
                        float _exp2_45 = approx_exp2((-gk_45) * 1.4426950408889634f);
                        float _rcp_45 = approx_rcp(1.0f + _exp2_45);
                        float sig_45 = _rcp_45;
                        float _tanh_90 = tanhf(gk_45 * 0.25f);
                        float aa_45 = 4.0f * _tanh_90 * sig_45;
                        float _tanh_91 = tanhf(uk_45 * 0.04f);
                        float bb_45 = 25.0f * _tanh_91;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_88 * 768 + row_s + frow_89)) + (0)) = __float2bfloat16_rn(aa_45 * bb_45);
                    }
                    int tok_90 = 88 + lane_pair * 2;
                    int frow_91 = row_base + ((1) ? 8 : 0);
                    if (tok_90 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_224 = __float2bfloat16(red[46]);
                        float _cvt_f32_224 = __bfloat162float(_cvt_bf16_224);
                        float gk_46 = _cvt_f32_224;
                        __nv_bfloat16 _cvt_bf16_225 = __float2bfloat16(red[110]);
                        float _cvt_f32_225 = __bfloat162float(_cvt_bf16_225);
                        float uk_46 = _cvt_f32_225;
                        float _exp2_46 = approx_exp2((-gk_46) * 1.4426950408889634f);
                        float _rcp_46 = approx_rcp(1.0f + _exp2_46);
                        float sig_46 = _rcp_46;
                        float _tanh_92 = tanhf(gk_46 * 0.25f);
                        float aa_46 = 4.0f * _tanh_92 * sig_46;
                        float _tanh_93 = tanhf(uk_46 * 0.04f);
                        float bb_46 = 25.0f * _tanh_93;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_90 * 768 + row_s + frow_91)) + (0)) = __float2bfloat16_rn(aa_46 * bb_46);
                    }
                    int tok_92 = 88 + lane_pair * 2 + 1;
                    int frow_93 = row_base + ((1) ? 8 : 0);
                    if (tok_92 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_226 = __float2bfloat16(red[47]);
                        float _cvt_f32_226 = __bfloat162float(_cvt_bf16_226);
                        float gk_47 = _cvt_f32_226;
                        __nv_bfloat16 _cvt_bf16_227 = __float2bfloat16(red[111]);
                        float _cvt_f32_227 = __bfloat162float(_cvt_bf16_227);
                        float uk_47 = _cvt_f32_227;
                        float _exp2_47 = approx_exp2((-gk_47) * 1.4426950408889634f);
                        float _rcp_47 = approx_rcp(1.0f + _exp2_47);
                        float sig_47 = _rcp_47;
                        float _tanh_94 = tanhf(gk_47 * 0.25f);
                        float aa_47 = 4.0f * _tanh_94 * sig_47;
                        float _tanh_95 = tanhf(uk_47 * 0.04f);
                        float bb_47 = 25.0f * _tanh_95;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_92 * 768 + row_s + frow_93)) + (0)) = __float2bfloat16_rn(aa_47 * bb_47);
                    }
                    int tok_94 = 96 + lane_pair * 2;
                    int frow_95 = row_base + ((0) ? 8 : 0);
                    if (tok_94 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_228 = __float2bfloat16(red[48]);
                        float _cvt_f32_228 = __bfloat162float(_cvt_bf16_228);
                        float gk_48 = _cvt_f32_228;
                        __nv_bfloat16 _cvt_bf16_229 = __float2bfloat16(red[112]);
                        float _cvt_f32_229 = __bfloat162float(_cvt_bf16_229);
                        float uk_48 = _cvt_f32_229;
                        float _exp2_48 = approx_exp2((-gk_48) * 1.4426950408889634f);
                        float _rcp_48 = approx_rcp(1.0f + _exp2_48);
                        float sig_48 = _rcp_48;
                        float _tanh_96 = tanhf(gk_48 * 0.25f);
                        float aa_48 = 4.0f * _tanh_96 * sig_48;
                        float _tanh_97 = tanhf(uk_48 * 0.04f);
                        float bb_48 = 25.0f * _tanh_97;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_94 * 768 + row_s + frow_95)) + (0)) = __float2bfloat16_rn(aa_48 * bb_48);
                    }
                    int tok_96 = 96 + lane_pair * 2 + 1;
                    int frow_97 = row_base + ((0) ? 8 : 0);
                    if (tok_96 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_230 = __float2bfloat16(red[49]);
                        float _cvt_f32_230 = __bfloat162float(_cvt_bf16_230);
                        float gk_49 = _cvt_f32_230;
                        __nv_bfloat16 _cvt_bf16_231 = __float2bfloat16(red[113]);
                        float _cvt_f32_231 = __bfloat162float(_cvt_bf16_231);
                        float uk_49 = _cvt_f32_231;
                        float _exp2_49 = approx_exp2((-gk_49) * 1.4426950408889634f);
                        float _rcp_49 = approx_rcp(1.0f + _exp2_49);
                        float sig_49 = _rcp_49;
                        float _tanh_98 = tanhf(gk_49 * 0.25f);
                        float aa_49 = 4.0f * _tanh_98 * sig_49;
                        float _tanh_99 = tanhf(uk_49 * 0.04f);
                        float bb_49 = 25.0f * _tanh_99;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_96 * 768 + row_s + frow_97)) + (0)) = __float2bfloat16_rn(aa_49 * bb_49);
                    }
                    int tok_98 = 96 + lane_pair * 2;
                    int frow_99 = row_base + ((1) ? 8 : 0);
                    if (tok_98 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_232 = __float2bfloat16(red[50]);
                        float _cvt_f32_232 = __bfloat162float(_cvt_bf16_232);
                        float gk_50 = _cvt_f32_232;
                        __nv_bfloat16 _cvt_bf16_233 = __float2bfloat16(red[114]);
                        float _cvt_f32_233 = __bfloat162float(_cvt_bf16_233);
                        float uk_50 = _cvt_f32_233;
                        float _exp2_50 = approx_exp2((-gk_50) * 1.4426950408889634f);
                        float _rcp_50 = approx_rcp(1.0f + _exp2_50);
                        float sig_50 = _rcp_50;
                        float _tanh_100 = tanhf(gk_50 * 0.25f);
                        float aa_50 = 4.0f * _tanh_100 * sig_50;
                        float _tanh_101 = tanhf(uk_50 * 0.04f);
                        float bb_50 = 25.0f * _tanh_101;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_98 * 768 + row_s + frow_99)) + (0)) = __float2bfloat16_rn(aa_50 * bb_50);
                    }
                    int tok_100 = 96 + lane_pair * 2 + 1;
                    int frow_101 = row_base + ((1) ? 8 : 0);
                    if (tok_100 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_234 = __float2bfloat16(red[51]);
                        float _cvt_f32_234 = __bfloat162float(_cvt_bf16_234);
                        float gk_51 = _cvt_f32_234;
                        __nv_bfloat16 _cvt_bf16_235 = __float2bfloat16(red[115]);
                        float _cvt_f32_235 = __bfloat162float(_cvt_bf16_235);
                        float uk_51 = _cvt_f32_235;
                        float _exp2_51 = approx_exp2((-gk_51) * 1.4426950408889634f);
                        float _rcp_51 = approx_rcp(1.0f + _exp2_51);
                        float sig_51 = _rcp_51;
                        float _tanh_102 = tanhf(gk_51 * 0.25f);
                        float aa_51 = 4.0f * _tanh_102 * sig_51;
                        float _tanh_103 = tanhf(uk_51 * 0.04f);
                        float bb_51 = 25.0f * _tanh_103;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_100 * 768 + row_s + frow_101)) + (0)) = __float2bfloat16_rn(aa_51 * bb_51);
                    }
                    int tok_102 = 104 + lane_pair * 2;
                    int frow_103 = row_base + ((0) ? 8 : 0);
                    if (tok_102 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_236 = __float2bfloat16(red[52]);
                        float _cvt_f32_236 = __bfloat162float(_cvt_bf16_236);
                        float gk_52 = _cvt_f32_236;
                        __nv_bfloat16 _cvt_bf16_237 = __float2bfloat16(red[116]);
                        float _cvt_f32_237 = __bfloat162float(_cvt_bf16_237);
                        float uk_52 = _cvt_f32_237;
                        float _exp2_52 = approx_exp2((-gk_52) * 1.4426950408889634f);
                        float _rcp_52 = approx_rcp(1.0f + _exp2_52);
                        float sig_52 = _rcp_52;
                        float _tanh_104 = tanhf(gk_52 * 0.25f);
                        float aa_52 = 4.0f * _tanh_104 * sig_52;
                        float _tanh_105 = tanhf(uk_52 * 0.04f);
                        float bb_52 = 25.0f * _tanh_105;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_102 * 768 + row_s + frow_103)) + (0)) = __float2bfloat16_rn(aa_52 * bb_52);
                    }
                    int tok_104 = 104 + lane_pair * 2 + 1;
                    int frow_105 = row_base + ((0) ? 8 : 0);
                    if (tok_104 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_238 = __float2bfloat16(red[53]);
                        float _cvt_f32_238 = __bfloat162float(_cvt_bf16_238);
                        float gk_53 = _cvt_f32_238;
                        __nv_bfloat16 _cvt_bf16_239 = __float2bfloat16(red[117]);
                        float _cvt_f32_239 = __bfloat162float(_cvt_bf16_239);
                        float uk_53 = _cvt_f32_239;
                        float _exp2_53 = approx_exp2((-gk_53) * 1.4426950408889634f);
                        float _rcp_53 = approx_rcp(1.0f + _exp2_53);
                        float sig_53 = _rcp_53;
                        float _tanh_106 = tanhf(gk_53 * 0.25f);
                        float aa_53 = 4.0f * _tanh_106 * sig_53;
                        float _tanh_107 = tanhf(uk_53 * 0.04f);
                        float bb_53 = 25.0f * _tanh_107;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_104 * 768 + row_s + frow_105)) + (0)) = __float2bfloat16_rn(aa_53 * bb_53);
                    }
                    int tok_106 = 104 + lane_pair * 2;
                    int frow_107 = row_base + ((1) ? 8 : 0);
                    if (tok_106 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_240 = __float2bfloat16(red[54]);
                        float _cvt_f32_240 = __bfloat162float(_cvt_bf16_240);
                        float gk_54 = _cvt_f32_240;
                        __nv_bfloat16 _cvt_bf16_241 = __float2bfloat16(red[118]);
                        float _cvt_f32_241 = __bfloat162float(_cvt_bf16_241);
                        float uk_54 = _cvt_f32_241;
                        float _exp2_54 = approx_exp2((-gk_54) * 1.4426950408889634f);
                        float _rcp_54 = approx_rcp(1.0f + _exp2_54);
                        float sig_54 = _rcp_54;
                        float _tanh_108 = tanhf(gk_54 * 0.25f);
                        float aa_54 = 4.0f * _tanh_108 * sig_54;
                        float _tanh_109 = tanhf(uk_54 * 0.04f);
                        float bb_54 = 25.0f * _tanh_109;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_106 * 768 + row_s + frow_107)) + (0)) = __float2bfloat16_rn(aa_54 * bb_54);
                    }
                    int tok_108 = 104 + lane_pair * 2 + 1;
                    int frow_109 = row_base + ((1) ? 8 : 0);
                    if (tok_108 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_242 = __float2bfloat16(red[55]);
                        float _cvt_f32_242 = __bfloat162float(_cvt_bf16_242);
                        float gk_55 = _cvt_f32_242;
                        __nv_bfloat16 _cvt_bf16_243 = __float2bfloat16(red[119]);
                        float _cvt_f32_243 = __bfloat162float(_cvt_bf16_243);
                        float uk_55 = _cvt_f32_243;
                        float _exp2_55 = approx_exp2((-gk_55) * 1.4426950408889634f);
                        float _rcp_55 = approx_rcp(1.0f + _exp2_55);
                        float sig_55 = _rcp_55;
                        float _tanh_110 = tanhf(gk_55 * 0.25f);
                        float aa_55 = 4.0f * _tanh_110 * sig_55;
                        float _tanh_111 = tanhf(uk_55 * 0.04f);
                        float bb_55 = 25.0f * _tanh_111;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_108 * 768 + row_s + frow_109)) + (0)) = __float2bfloat16_rn(aa_55 * bb_55);
                    }
                    int tok_110 = 112 + lane_pair * 2;
                    int frow_111 = row_base + ((0) ? 8 : 0);
                    if (tok_110 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_244 = __float2bfloat16(red[56]);
                        float _cvt_f32_244 = __bfloat162float(_cvt_bf16_244);
                        float gk_56 = _cvt_f32_244;
                        __nv_bfloat16 _cvt_bf16_245 = __float2bfloat16(red[120]);
                        float _cvt_f32_245 = __bfloat162float(_cvt_bf16_245);
                        float uk_56 = _cvt_f32_245;
                        float _exp2_56 = approx_exp2((-gk_56) * 1.4426950408889634f);
                        float _rcp_56 = approx_rcp(1.0f + _exp2_56);
                        float sig_56 = _rcp_56;
                        float _tanh_112 = tanhf(gk_56 * 0.25f);
                        float aa_56 = 4.0f * _tanh_112 * sig_56;
                        float _tanh_113 = tanhf(uk_56 * 0.04f);
                        float bb_56 = 25.0f * _tanh_113;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_110 * 768 + row_s + frow_111)) + (0)) = __float2bfloat16_rn(aa_56 * bb_56);
                    }
                    int tok_112 = 112 + lane_pair * 2 + 1;
                    int frow_113 = row_base + ((0) ? 8 : 0);
                    if (tok_112 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_246 = __float2bfloat16(red[57]);
                        float _cvt_f32_246 = __bfloat162float(_cvt_bf16_246);
                        float gk_57 = _cvt_f32_246;
                        __nv_bfloat16 _cvt_bf16_247 = __float2bfloat16(red[121]);
                        float _cvt_f32_247 = __bfloat162float(_cvt_bf16_247);
                        float uk_57 = _cvt_f32_247;
                        float _exp2_57 = approx_exp2((-gk_57) * 1.4426950408889634f);
                        float _rcp_57 = approx_rcp(1.0f + _exp2_57);
                        float sig_57 = _rcp_57;
                        float _tanh_114 = tanhf(gk_57 * 0.25f);
                        float aa_57 = 4.0f * _tanh_114 * sig_57;
                        float _tanh_115 = tanhf(uk_57 * 0.04f);
                        float bb_57 = 25.0f * _tanh_115;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_112 * 768 + row_s + frow_113)) + (0)) = __float2bfloat16_rn(aa_57 * bb_57);
                    }
                    int tok_114 = 112 + lane_pair * 2;
                    int frow_115 = row_base + ((1) ? 8 : 0);
                    if (tok_114 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_248 = __float2bfloat16(red[58]);
                        float _cvt_f32_248 = __bfloat162float(_cvt_bf16_248);
                        float gk_58 = _cvt_f32_248;
                        __nv_bfloat16 _cvt_bf16_249 = __float2bfloat16(red[122]);
                        float _cvt_f32_249 = __bfloat162float(_cvt_bf16_249);
                        float uk_58 = _cvt_f32_249;
                        float _exp2_58 = approx_exp2((-gk_58) * 1.4426950408889634f);
                        float _rcp_58 = approx_rcp(1.0f + _exp2_58);
                        float sig_58 = _rcp_58;
                        float _tanh_116 = tanhf(gk_58 * 0.25f);
                        float aa_58 = 4.0f * _tanh_116 * sig_58;
                        float _tanh_117 = tanhf(uk_58 * 0.04f);
                        float bb_58 = 25.0f * _tanh_117;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_114 * 768 + row_s + frow_115)) + (0)) = __float2bfloat16_rn(aa_58 * bb_58);
                    }
                    int tok_116 = 112 + lane_pair * 2 + 1;
                    int frow_117 = row_base + ((1) ? 8 : 0);
                    if (tok_116 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_250 = __float2bfloat16(red[59]);
                        float _cvt_f32_250 = __bfloat162float(_cvt_bf16_250);
                        float gk_59 = _cvt_f32_250;
                        __nv_bfloat16 _cvt_bf16_251 = __float2bfloat16(red[123]);
                        float _cvt_f32_251 = __bfloat162float(_cvt_bf16_251);
                        float uk_59 = _cvt_f32_251;
                        float _exp2_59 = approx_exp2((-gk_59) * 1.4426950408889634f);
                        float _rcp_59 = approx_rcp(1.0f + _exp2_59);
                        float sig_59 = _rcp_59;
                        float _tanh_118 = tanhf(gk_59 * 0.25f);
                        float aa_59 = 4.0f * _tanh_118 * sig_59;
                        float _tanh_119 = tanhf(uk_59 * 0.04f);
                        float bb_59 = 25.0f * _tanh_119;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_116 * 768 + row_s + frow_117)) + (0)) = __float2bfloat16_rn(aa_59 * bb_59);
                    }
                    int tok_118 = 120 + lane_pair * 2;
                    int frow_119 = row_base + ((0) ? 8 : 0);
                    if (tok_118 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_252 = __float2bfloat16(red[60]);
                        float _cvt_f32_252 = __bfloat162float(_cvt_bf16_252);
                        float gk_60 = _cvt_f32_252;
                        __nv_bfloat16 _cvt_bf16_253 = __float2bfloat16(red[124]);
                        float _cvt_f32_253 = __bfloat162float(_cvt_bf16_253);
                        float uk_60 = _cvt_f32_253;
                        float _exp2_60 = approx_exp2((-gk_60) * 1.4426950408889634f);
                        float _rcp_60 = approx_rcp(1.0f + _exp2_60);
                        float sig_60 = _rcp_60;
                        float _tanh_120 = tanhf(gk_60 * 0.25f);
                        float aa_60 = 4.0f * _tanh_120 * sig_60;
                        float _tanh_121 = tanhf(uk_60 * 0.04f);
                        float bb_60 = 25.0f * _tanh_121;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_118 * 768 + row_s + frow_119)) + (0)) = __float2bfloat16_rn(aa_60 * bb_60);
                    }
                    int tok_120 = 120 + lane_pair * 2 + 1;
                    int frow_121 = row_base + ((0) ? 8 : 0);
                    if (tok_120 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_254 = __float2bfloat16(red[61]);
                        float _cvt_f32_254 = __bfloat162float(_cvt_bf16_254);
                        float gk_61 = _cvt_f32_254;
                        __nv_bfloat16 _cvt_bf16_255 = __float2bfloat16(red[125]);
                        float _cvt_f32_255 = __bfloat162float(_cvt_bf16_255);
                        float uk_61 = _cvt_f32_255;
                        float _exp2_61 = approx_exp2((-gk_61) * 1.4426950408889634f);
                        float _rcp_61 = approx_rcp(1.0f + _exp2_61);
                        float sig_61 = _rcp_61;
                        float _tanh_122 = tanhf(gk_61 * 0.25f);
                        float aa_61 = 4.0f * _tanh_122 * sig_61;
                        float _tanh_123 = tanhf(uk_61 * 0.04f);
                        float bb_61 = 25.0f * _tanh_123;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_120 * 768 + row_s + frow_121)) + (0)) = __float2bfloat16_rn(aa_61 * bb_61);
                    }
                    int tok_122 = 120 + lane_pair * 2;
                    int frow_123 = row_base + ((1) ? 8 : 0);
                    if (tok_122 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_256 = __float2bfloat16(red[62]);
                        float _cvt_f32_256 = __bfloat162float(_cvt_bf16_256);
                        float gk_62 = _cvt_f32_256;
                        __nv_bfloat16 _cvt_bf16_257 = __float2bfloat16(red[126]);
                        float _cvt_f32_257 = __bfloat162float(_cvt_bf16_257);
                        float uk_62 = _cvt_f32_257;
                        float _exp2_62 = approx_exp2((-gk_62) * 1.4426950408889634f);
                        float _rcp_62 = approx_rcp(1.0f + _exp2_62);
                        float sig_62 = _rcp_62;
                        float _tanh_124 = tanhf(gk_62 * 0.25f);
                        float aa_62 = 4.0f * _tanh_124 * sig_62;
                        float _tanh_125 = tanhf(uk_62 * 0.04f);
                        float bb_62 = 25.0f * _tanh_125;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_122 * 768 + row_s + frow_123)) + (0)) = __float2bfloat16_rn(aa_62 * bb_62);
                    }
                    int tok_124 = 120 + lane_pair * 2 + 1;
                    int frow_125 = row_base + ((1) ? 8 : 0);
                    if (tok_124 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_258 = __float2bfloat16(red[63]);
                        float _cvt_f32_258 = __bfloat162float(_cvt_bf16_258);
                        float gk_63 = _cvt_f32_258;
                        __nv_bfloat16 _cvt_bf16_259 = __float2bfloat16(red[127]);
                        float _cvt_f32_259 = __bfloat162float(_cvt_bf16_259);
                        float uk_63 = _cvt_f32_259;
                        float _exp2_63 = approx_exp2((-gk_63) * 1.4426950408889634f);
                        float _rcp_63 = approx_rcp(1.0f + _exp2_63);
                        float sig_63 = _rcp_63;
                        float _tanh_126 = tanhf(gk_63 * 0.25f);
                        float aa_63 = 4.0f * _tanh_126 * sig_63;
                        float _tanh_127 = tanhf(uk_63 * 0.04f);
                        float bb_63 = 25.0f * _tanh_127;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_124 * 768 + row_s + frow_125)) + (0)) = __float2bfloat16_rn(aa_63 * bb_63);
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
                    if (num_tokens > 64) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (57344 + row_r + tid_1)) + (0)) = red[64];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (229376 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[64]);
                        }
                    }
                    if (num_tokens > 65) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (58240 + row_r + tid_1)) + (0)) = red[65];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (232960 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[65]);
                        }
                    }
                    if (num_tokens > 66) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (59136 + row_r + tid_1)) + (0)) = red[66];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (236544 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[66]);
                        }
                    }
                    if (num_tokens > 67) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (60032 + row_r + tid_1)) + (0)) = red[67];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (240128 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[67]);
                        }
                    }
                    if (num_tokens > 68) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (60928 + row_r + tid_1)) + (0)) = red[68];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (243712 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[68]);
                        }
                    }
                    if (num_tokens > 69) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (61824 + row_r + tid_1)) + (0)) = red[69];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (247296 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[69]);
                        }
                    }
                    if (num_tokens > 70) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (62720 + row_r + tid_1)) + (0)) = red[70];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (250880 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[70]);
                        }
                    }
                    if (num_tokens > 71) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (63616 + row_r + tid_1)) + (0)) = red[71];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (254464 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[71]);
                        }
                    }
                    if (num_tokens > 72) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (64512 + row_r + tid_1)) + (0)) = red[72];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (258048 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[72]);
                        }
                    }
                    if (num_tokens > 73) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (65408 + row_r + tid_1)) + (0)) = red[73];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (261632 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[73]);
                        }
                    }
                    if (num_tokens > 74) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (66304 + row_r + tid_1)) + (0)) = red[74];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (265216 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[74]);
                        }
                    }
                    if (num_tokens > 75) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (67200 + row_r + tid_1)) + (0)) = red[75];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (268800 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[75]);
                        }
                    }
                    if (num_tokens > 76) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (68096 + row_r + tid_1)) + (0)) = red[76];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (272384 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[76]);
                        }
                    }
                    if (num_tokens > 77) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (68992 + row_r + tid_1)) + (0)) = red[77];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (275968 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[77]);
                        }
                    }
                    if (num_tokens > 78) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (69888 + row_r + tid_1)) + (0)) = red[78];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (279552 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[78]);
                        }
                    }
                    if (num_tokens > 79) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (70784 + row_r + tid_1)) + (0)) = red[79];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (283136 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[79]);
                        }
                    }
                    if (num_tokens > 80) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (71680 + row_r + tid_1)) + (0)) = red[80];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (286720 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[80]);
                        }
                    }
                    if (num_tokens > 81) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (72576 + row_r + tid_1)) + (0)) = red[81];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (290304 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[81]);
                        }
                    }
                    if (num_tokens > 82) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (73472 + row_r + tid_1)) + (0)) = red[82];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (293888 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[82]);
                        }
                    }
                    if (num_tokens > 83) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (74368 + row_r + tid_1)) + (0)) = red[83];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (297472 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[83]);
                        }
                    }
                    if (num_tokens > 84) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (75264 + row_r + tid_1)) + (0)) = red[84];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (301056 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[84]);
                        }
                    }
                    if (num_tokens > 85) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (76160 + row_r + tid_1)) + (0)) = red[85];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (304640 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[85]);
                        }
                    }
                    if (num_tokens > 86) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (77056 + row_r + tid_1)) + (0)) = red[86];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (308224 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[86]);
                        }
                    }
                    if (num_tokens > 87) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (77952 + row_r + tid_1)) + (0)) = red[87];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (311808 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[87]);
                        }
                    }
                    if (num_tokens > 88) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (78848 + row_r + tid_1)) + (0)) = red[88];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (315392 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[88]);
                        }
                    }
                    if (num_tokens > 89) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (79744 + row_r + tid_1)) + (0)) = red[89];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (318976 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[89]);
                        }
                    }
                    if (num_tokens > 90) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (80640 + row_r + tid_1)) + (0)) = red[90];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (322560 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[90]);
                        }
                    }
                    if (num_tokens > 91) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (81536 + row_r + tid_1)) + (0)) = red[91];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (326144 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[91]);
                        }
                    }
                    if (num_tokens > 92) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (82432 + row_r + tid_1)) + (0)) = red[92];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (329728 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[92]);
                        }
                    }
                    if (num_tokens > 93) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (83328 + row_r + tid_1)) + (0)) = red[93];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (333312 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[93]);
                        }
                    }
                    if (num_tokens > 94) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (84224 + row_r + tid_1)) + (0)) = red[94];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (336896 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[94]);
                        }
                    }
                    if (num_tokens > 95) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (85120 + row_r + tid_1)) + (0)) = red[95];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (340480 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[95]);
                        }
                    }
                    if (num_tokens > 96) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (86016 + row_r + tid_1)) + (0)) = red[96];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (344064 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[96]);
                        }
                    }
                    if (num_tokens > 97) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (86912 + row_r + tid_1)) + (0)) = red[97];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (347648 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[97]);
                        }
                    }
                    if (num_tokens > 98) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (87808 + row_r + tid_1)) + (0)) = red[98];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (351232 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[98]);
                        }
                    }
                    if (num_tokens > 99) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (88704 + row_r + tid_1)) + (0)) = red[99];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (354816 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[99]);
                        }
                    }
                    if (num_tokens > 100) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (89600 + row_r + tid_1)) + (0)) = red[100];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (358400 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[100]);
                        }
                    }
                    if (num_tokens > 101) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (90496 + row_r + tid_1)) + (0)) = red[101];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (361984 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[101]);
                        }
                    }
                    if (num_tokens > 102) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (91392 + row_r + tid_1)) + (0)) = red[102];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (365568 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[102]);
                        }
                    }
                    if (num_tokens > 103) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (92288 + row_r + tid_1)) + (0)) = red[103];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (369152 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[103]);
                        }
                    }
                    if (num_tokens > 104) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (93184 + row_r + tid_1)) + (0)) = red[104];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (372736 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[104]);
                        }
                    }
                    if (num_tokens > 105) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (94080 + row_r + tid_1)) + (0)) = red[105];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (376320 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[105]);
                        }
                    }
                    if (num_tokens > 106) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (94976 + row_r + tid_1)) + (0)) = red[106];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (379904 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[106]);
                        }
                    }
                    if (num_tokens > 107) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (95872 + row_r + tid_1)) + (0)) = red[107];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (383488 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[107]);
                        }
                    }
                    if (num_tokens > 108) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (96768 + row_r + tid_1)) + (0)) = red[108];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (387072 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[108]);
                        }
                    }
                    if (num_tokens > 109) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (97664 + row_r + tid_1)) + (0)) = red[109];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (390656 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[109]);
                        }
                    }
                    if (num_tokens > 110) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (98560 + row_r + tid_1)) + (0)) = red[110];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (394240 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[110]);
                        }
                    }
                    if (num_tokens > 111) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (99456 + row_r + tid_1)) + (0)) = red[111];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (397824 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[111]);
                        }
                    }
                    if (num_tokens > 112) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (100352 + row_r + tid_1)) + (0)) = red[112];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (401408 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[112]);
                        }
                    }
                    if (num_tokens > 113) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (101248 + row_r + tid_1)) + (0)) = red[113];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (404992 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[113]);
                        }
                    }
                    if (num_tokens > 114) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (102144 + row_r + tid_1)) + (0)) = red[114];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (408576 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[114]);
                        }
                    }
                    if (num_tokens > 115) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (103040 + row_r + tid_1)) + (0)) = red[115];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (412160 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[115]);
                        }
                    }
                    if (num_tokens > 116) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (103936 + row_r + tid_1)) + (0)) = red[116];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (415744 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[116]);
                        }
                    }
                    if (num_tokens > 117) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (104832 + row_r + tid_1)) + (0)) = red[117];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (419328 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[117]);
                        }
                    }
                    if (num_tokens > 118) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (105728 + row_r + tid_1)) + (0)) = red[118];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (422912 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[118]);
                        }
                    }
                    if (num_tokens > 119) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (106624 + row_r + tid_1)) + (0)) = red[119];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (426496 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[119]);
                        }
                    }
                    if (num_tokens > 120) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (107520 + row_r + tid_1)) + (0)) = red[120];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (430080 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[120]);
                        }
                    }
                    if (num_tokens > 121) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (108416 + row_r + tid_1)) + (0)) = red[121];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (433664 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[121]);
                        }
                    }
                    if (num_tokens > 122) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (109312 + row_r + tid_1)) + (0)) = red[122];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (437248 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[122]);
                        }
                    }
                    if (num_tokens > 123) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (110208 + row_r + tid_1)) + (0)) = red[123];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (440832 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[123]);
                        }
                    }
                    if (num_tokens > 124) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (111104 + row_r + tid_1)) + (0)) = red[124];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (444416 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[124]);
                        }
                    }
                    if (num_tokens > 125) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (112000 + row_r + tid_1)) + (0)) = red[125];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (448000 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[125]);
                        }
                    }
                    if (num_tokens > 126) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (112896 + row_r + tid_1)) + (0)) = red[126];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (451584 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[126]);
                        }
                    }
                    if (num_tokens > 127) {
                        if (cls == 0) {
                            *(reinterpret_cast<float*>(out_r + (113792 + row_r + tid_1)) + (0)) = red[127];
                        } else {
                            *(reinterpret_cast<__nv_bfloat16*>(out_l + (455168 + row_l + tid_1)) + (0)) = __float2bfloat16_rn(red[127]);
                        }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
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
            int _min_0 = ((5) < (u_count_l) ? (5) : (u_count_l));
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
                        tma_3d_gmem2smem(smem_b_addr + stage * 16384, (&B_1), 0, 0, kb1, a_full_addr + (stage) * 8);
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
                        mbarrier_arrive_expect_tx(a_full_addr + (stage) * 8, 32768);
                    }
                    stage += 1;
                    if (stage == 5) { stage = 0; _phase_a_empty ^= 1; }
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
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
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
                    "mov.b32 id, 69207184;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_acc + (128))), "r"(((init_sub) ? 0 : 1)));
                            int _mma_a_lo_1 = (((smem_au_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
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
                    "mov.b32 id, 69207184;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_acc + (256))), "r"(((init_sub) ? 0 : 1)));
                        } else {
                            int _mma_a_lo_2 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
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
                    "mov.b32 id, 136316048;\n\t"
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
                    if (stage_1 == 5) { stage_1 = 0; _phase_a_full ^= 1; }
                }
                tcgen05_commit(acc_full_addr + (epi_stage_1) * 8);
            }
        }
    }

    // Cleanup
}

} // extern "C"
