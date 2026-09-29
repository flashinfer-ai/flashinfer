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
#define NUM_TMA_PIPE_STAGES 5
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_LAND_OFF 2048
#define SMEM_LAND_STAGE_BYTES 8192
#define SMEM_LAND_STRIDE 8192
#define SMEM_FREE_WORD_OFF 1536
#define SMEM_FREE_WORD_STAGE_BYTES 8
#define SMEM_FREE_WORD_STRIDE 8
#define SMEM_SMEM_A_OFF 10240
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 92160
#define SMEM_SMEM_B_STAGE_BYTES 2048
#define SMEM_SMEM_B_STRIDE 2048
#define SMEM_SMEM_BN_OFF 102400
#define SMEM_SMEM_BN_STAGE_BYTES 2048
#define SMEM_SMEM_BN_STRIDE 2048
#define SMEM_SMEM_V5_OFF 110592
#define SMEM_SMEM_V5_STAGE_BYTES 114688
#define SMEM_SMEM_V5_STRIDE 114688
#define SMEM_SMEM_V6_OFF 110592
#define SMEM_SMEM_V6_STAGE_BYTES 114688
#define SMEM_SMEM_V6_STRIDE 114688
#define SMEM_SMEM_V7_OFF 225280
#define SMEM_SMEM_V7_STAGE_BYTES 7168
#define SMEM_SMEM_V7_STRIDE 7168
#define SMEM_SMEM_V8_OFF 225280
#define SMEM_SMEM_V8_STAGE_BYTES 7168
#define SMEM_SMEM_V8_STRIDE 7168
#define SMEM_SMEM_AG_OFF 10240
#define SMEM_SMEM_AG_STAGE_BYTES 8192
#define SMEM_SMEM_AG_STRIDE 16384
#define SMEM_SMEM_AU_OFF 18432
#define SMEM_SMEM_AU_STAGE_BYTES 8192
#define SMEM_SMEM_AU_STRIDE 16384
#define SMEM_TOTAL 232448

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


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
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


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_latent_moe_e8097277b8df0d31c513(const __grid_constant__ CUtensorMap A_R, const __grid_constant__ CUtensorMap A_L, const __grid_constant__ CUtensorMap A_S, const __grid_constant__ CUtensorMap A_2, const __grid_constant__ CUtensorMap B_1, const __grid_constant__ CUtensorMap B_2, float* __restrict__ out_r, __nv_bfloat16* __restrict__ out_l, __nv_bfloat16* __restrict__ out_s, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out, unsigned long long* __restrict__ tl, int num_tokens, int k1_off, int num_partials, float eps)
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
    #define rows_full_addr (mbar_base + 128)

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
    unsigned int* free_word = reinterpret_cast<unsigned int*>(smem_raw + 1536);
    const int free_word_addr = smem + 1536;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 10240);
    const int smem_a_addr = smem + 10240;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 92160);
    const int smem_b_addr = smem + 92160;
    __nv_bfloat16* smem_bn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 102400);
    const int smem_bn_addr = smem + 102400;
    __nv_bfloat16* smem_v5 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 110592);
    const int smem_v5_addr = smem + 110592;
    unsigned int* smem_v6 = reinterpret_cast<unsigned int*>(smem_raw + 110592);
    const int smem_v6_addr = smem + 110592;
    __nv_bfloat16* smem_v7 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 225280);
    const int smem_v7_addr = smem + 225280;
    unsigned int* smem_v8 = reinterpret_cast<unsigned int*>(smem_raw + 225280);
    const int smem_v8_addr = smem + 225280;
    __nv_bfloat16* smem_ag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 10240);
    const int smem_ag_addr = smem + 10240;
    __nv_bfloat16* smem_au = reinterpret_cast<__nv_bfloat16*>(smem_raw + 18432);
    const int smem_au_addr = smem + 18432;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 17 barriers)
    // Mbarriers at smem_raw[0..136)

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
            // rows_full: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 96, 8192);
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 136);
    if (warp == 0) {
        int _tmem_hold = smem + 136;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    // barrier.cluster.wait deferred (WarpConfig.cluster_init_wait_warps): role entries, or the schedule's ClusterSyncWait when the tuple is empty

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        // Deferred post-initialization cluster wait (WarpConfig.cluster_init_wait_warps)
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        { // epilogue_main
            int g_e = bid;
            int epi_warp = warp % 4;
            int tid_1 = epi_warp * 32 + lane;
            int tl_e = g_e * 16;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                {
                    int rank_e = 0;
                    {
                        rank_e = g_e % 2;
                    }
                    int c_lo = k1_off + ((rank_e == 0) ? 0 : 4);
                    int c_cnt = ((rank_e == 0) ? 4 : 3);
                    int col_b = lane % 8 * 16;
                    unsigned long long partial_stride = (unsigned long long)num_tokens * 3584;
                    unsigned int zero_w = 0;
                    unsigned int nwp[56];
                    mbarrier_wait(rows_full_addr, 0);
                    unsigned int xp[112];
                    {
                        #pragma unroll
                        for (int i = 0; i < 14; i++) {
                            unsigned int _smem_v8_reg_0[4];
                            {
                                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_v8);
                                #pragma unroll
                                for (int _lr = 0; _lr < 4; _lr++)
                                    _smem_v8_reg_0[_lr] = _smem_ptr[((lane + i * 32) * 4) + _lr];
                            }
                            #pragma unroll
                            for (int k = 0; k < 4; k++) {
                                nwp[i * 4 + k] = _smem_v8_reg_0[k];
                            }
                        }
                    }
                    int r_ld = epi_warp;
                    if (r_ld < num_tokens) {
                        unsigned long long nrow_ld = (unsigned long long)r_ld * 3584;
                        {
                            #pragma unroll
                            for (int i_1 = 0; i_1 < 14; i_1++) {
                                unsigned int _smem_v6_reg_0[4];
                                {
                                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_v6);
                                    #pragma unroll
                                    for (int _lr = 0; _lr < 4; _lr++)
                                        _smem_v6_reg_0[_lr] = _smem_ptr[(r_ld * 1792 + (lane + i_1 * 32) * 4) + _lr];
                                }
                                #pragma unroll
                                for (int k_1 = 0; k_1 < 4; k_1++) {
                                    xp[i_1 * 4 + k_1] = _smem_v6_reg_0[k_1];
                                }
                            }
                        }
                    }
                    int r_ld_0 = epi_warp + 4;
                    if (r_ld_0 < num_tokens) {
                        unsigned long long nrow_ld_1 = (unsigned long long)r_ld_0 * 3584;
                        {
                            #pragma unroll
                            for (int i_2 = 0; i_2 < 14; i_2++) {
                                unsigned int _smem_v6_reg_1[4];
                                {
                                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_v6);
                                    #pragma unroll
                                    for (int _lr = 0; _lr < 4; _lr++)
                                        _smem_v6_reg_1[_lr] = _smem_ptr[(r_ld_0 * 1792 + (lane + i_2 * 32) * 4) + _lr];
                                }
                                #pragma unroll
                                for (int k_2 = 0; k_2 < 4; k_2++) {
                                    xp[56 + i_2 * 4 + k_2] = _smem_v6_reg_1[k_2];
                                }
                            }
                        }
                    }
                    if (warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(issue_bar_addr);
                        }
                    }
                    int r = epi_warp;
                    if (r < num_tokens) {
                        unsigned long long nrow_base = (unsigned long long)r * 3584;
                        float sum_sq = 0.0f;
                        #pragma unroll
                        for (int i_3 = 0; i_3 < 56; i_3++) {
                            float _bf16x2_add_f32_0[2];
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                "}\n"
                                : "=&f"(_bf16x2_add_f32_0[0]), "=&f"(_bf16x2_add_f32_0[1]) : "r"(xp[i_3]), "f"(0.0f), "f"(0.0f));
                            sum_sq += _bf16x2_add_f32_0[0] * _bf16x2_add_f32_0[0];
                            sum_sq += _bf16x2_add_f32_0[1] * _bf16x2_add_f32_0[1];
                        }
                        float _warp_reduce_0 = sum_sq;
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                        float total = _warp_reduce_0;
                        float _rsqrt_0 = rsqrtf(total / 3584.0f + eps);
                        float rstd = _rsqrt_0;
                        int own = (r - g_e) % 112;
                        #pragma unroll
                        for (int i_4 = 0; i_4 < 14; i_4++) {
                            int kk2 = (lane + i_4 * 32) * 8;
                            float y8[8];
                            #pragma unroll
                            for (int k_3 = 0; k_3 < 4; k_3++) {
                                float _bf16x2_add_f32_1[2];
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b16 lo, hi;\n\t"
                                    "mov.b32 {lo, hi}, %2;\n\t"
                                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                    "}\n"
                                    : "=&f"(_bf16x2_add_f32_1[0]), "=&f"(_bf16x2_add_f32_1[1]) : "r"(xp[i_4 * 4 + k_3]), "f"(0.0f), "f"(0.0f));
                                y8[2 * k_3] = _bf16x2_add_f32_1[0] * rstd;
                                y8[2 * k_3 + 1] = _bf16x2_add_f32_1[1] * rstd;
                            }
                            uint32_t y8_bf16[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y8[_lp*2 + 0], y8[_lp*2+1 + 0]));
                                y8_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int ow[4];
                            #pragma unroll
                            for (int k_4 = 0; k_4 < 4; k_4++) {
                                uint32_t _bf16x2_mul_0;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(nwp[i_4 * 4 + k_4]), "r"(y8_bf16[k_4]));
                                ow[k_4] = _bf16x2_mul_0;
                            }
                            if (own == 0) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(y_out + (nrow_base + (unsigned long long)kk2) + (0)), "r"(ow[(0) + 0]), "r"(ow[(0) + 1]), "r"(ow[(0) + 2]), "r"(ow[(0) + 3]) : "memory");
                            }
                            int c_rel = i_4 * 4 + lane / 8 - c_lo;
                            if (c_rel >= 0) {
                                if (c_rel < c_cnt) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(c_rel * 2048) + (unsigned int)(r * 128 + col_b ^ (r * 128 + col_b >> 7 & 7) << 4))), "r"(ow[0]), "r"(ow[1]), "r"(ow[2]), "r"(ow[3]) : "memory");
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int i_5 = 0; i_5 < 14; i_5++) {
                            int c_relz = i_5 * 4 + lane / 8 - c_lo;
                            if (c_relz >= 0) {
                                if (c_relz < c_cnt) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(c_relz * 2048) + (unsigned int)(r * 128 + col_b ^ (r * 128 + col_b >> 7 & 7) << 4))), "r"(zero_w), "r"(zero_w), "r"(zero_w), "r"(zero_w) : "memory");
                                }
                            }
                        }
                    }
                    int r_1 = epi_warp + 4;
                    if (r_1 < num_tokens) {
                        unsigned long long nrow_base_1 = (unsigned long long)r_1 * 3584;
                        float sum_sq_1 = 0.0f;
                        #pragma unroll
                        for (int i_6 = 0; i_6 < 56; i_6++) {
                            float _bf16x2_add_f32_2[2];
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                "}\n"
                                : "=&f"(_bf16x2_add_f32_2[0]), "=&f"(_bf16x2_add_f32_2[1]) : "r"(xp[56 + i_6]), "f"(0.0f), "f"(0.0f));
                            sum_sq_1 += _bf16x2_add_f32_2[0] * _bf16x2_add_f32_2[0];
                            sum_sq_1 += _bf16x2_add_f32_2[1] * _bf16x2_add_f32_2[1];
                        }
                        float _warp_reduce_1 = sum_sq_1;
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                        float total_1 = _warp_reduce_1;
                        float _rsqrt_1 = rsqrtf(total_1 / 3584.0f + eps);
                        float rstd_1 = _rsqrt_1;
                        int own_1 = (r_1 - g_e) % 112;
                        #pragma unroll
                        for (int i_7 = 0; i_7 < 14; i_7++) {
                            int kk2_1 = (lane + i_7 * 32) * 8;
                            float y8_1[8];
                            #pragma unroll
                            for (int k_5 = 0; k_5 < 4; k_5++) {
                                float _bf16x2_add_f32_3[2];
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b16 lo, hi;\n\t"
                                    "mov.b32 {lo, hi}, %2;\n\t"
                                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                    "}\n"
                                    : "=&f"(_bf16x2_add_f32_3[0]), "=&f"(_bf16x2_add_f32_3[1]) : "r"(xp[56 + i_7 * 4 + k_5]), "f"(0.0f), "f"(0.0f));
                                y8_1[2 * k_5] = _bf16x2_add_f32_3[0] * rstd_1;
                                y8_1[2 * k_5 + 1] = _bf16x2_add_f32_3[1] * rstd_1;
                            }
                            uint32_t y8_bf16_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y8_1[_lp*2 + 0], y8_1[_lp*2+1 + 0]));
                                y8_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int ow_1[4];
                            #pragma unroll
                            for (int k_6 = 0; k_6 < 4; k_6++) {
                                uint32_t _bf16x2_mul_1;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(nwp[i_7 * 4 + k_6]), "r"(y8_bf16_1[k_6]));
                                ow_1[k_6] = _bf16x2_mul_1;
                            }
                            if (own_1 == 0) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(y_out + (nrow_base_1 + (unsigned long long)kk2_1) + (0)), "r"(ow_1[(0) + 0]), "r"(ow_1[(0) + 1]), "r"(ow_1[(0) + 2]), "r"(ow_1[(0) + 3]) : "memory");
                            }
                            int c_rel_1 = i_7 * 4 + lane / 8 - c_lo;
                            if (c_rel_1 >= 0) {
                                if (c_rel_1 < c_cnt) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(c_rel_1 * 2048) + (unsigned int)(r_1 * 128 + col_b ^ (r_1 * 128 + col_b >> 7 & 7) << 4))), "r"(ow_1[0]), "r"(ow_1[1]), "r"(ow_1[2]), "r"(ow_1[3]) : "memory");
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int i_8 = 0; i_8 < 14; i_8++) {
                            int c_relz_1 = i_8 * 4 + lane / 8 - c_lo;
                            if (c_relz_1 >= 0) {
                                if (c_relz_1 < c_cnt) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(c_relz_1 * 2048) + (unsigned int)(r_1 * 128 + col_b ^ (r_1 * 128 + col_b >> 7 & 7) << 4))), "r"(zero_w), "r"(zero_w), "r"(zero_w), "r"(zero_w) : "memory");
                                }
                            }
                        }
                    }
                    unsigned int xp_2[112];
                    int r_ld_3 = epi_warp + 8;
                    if (r_ld_3 < num_tokens) {
                        unsigned long long nrow_ld_2 = (unsigned long long)r_ld_3 * 3584;
                        {
                            #pragma unroll
                            for (int i_9 = 0; i_9 < 14; i_9++) {
                                unsigned int _smem_v6_reg_2[4];
                                {
                                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_v6);
                                    #pragma unroll
                                    for (int _lr = 0; _lr < 4; _lr++)
                                        _smem_v6_reg_2[_lr] = _smem_ptr[(r_ld_3 * 1792 + (lane + i_9 * 32) * 4) + _lr];
                                }
                                #pragma unroll
                                for (int k_7 = 0; k_7 < 4; k_7++) {
                                    xp_2[i_9 * 4 + k_7] = _smem_v6_reg_2[k_7];
                                }
                            }
                        }
                    }
                    int r_ld_4 = epi_warp + 12;
                    if (r_ld_4 < num_tokens) {
                        unsigned long long nrow_ld_3 = (unsigned long long)r_ld_4 * 3584;
                        {
                            #pragma unroll
                            for (int i_10 = 0; i_10 < 14; i_10++) {
                                unsigned int _smem_v6_reg_3[4];
                                {
                                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_v6);
                                    #pragma unroll
                                    for (int _lr = 0; _lr < 4; _lr++)
                                        _smem_v6_reg_3[_lr] = _smem_ptr[(r_ld_4 * 1792 + (lane + i_10 * 32) * 4) + _lr];
                                }
                                #pragma unroll
                                for (int k_8 = 0; k_8 < 4; k_8++) {
                                    xp_2[56 + i_10 * 4 + k_8] = _smem_v6_reg_3[k_8];
                                }
                            }
                        }
                    }
                    int r_5 = epi_warp + 8;
                    if (r_5 < num_tokens) {
                        unsigned long long nrow_base_2 = (unsigned long long)r_5 * 3584;
                        float sum_sq_2 = 0.0f;
                        #pragma unroll
                        for (int i_11 = 0; i_11 < 56; i_11++) {
                            float _bf16x2_add_f32_4[2];
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                "}\n"
                                : "=&f"(_bf16x2_add_f32_4[0]), "=&f"(_bf16x2_add_f32_4[1]) : "r"(xp_2[i_11]), "f"(0.0f), "f"(0.0f));
                            sum_sq_2 += _bf16x2_add_f32_4[0] * _bf16x2_add_f32_4[0];
                            sum_sq_2 += _bf16x2_add_f32_4[1] * _bf16x2_add_f32_4[1];
                        }
                        float _warp_reduce_2 = sum_sq_2;
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
                        float total_2 = _warp_reduce_2;
                        float _rsqrt_2 = rsqrtf(total_2 / 3584.0f + eps);
                        float rstd_2 = _rsqrt_2;
                        int own_2 = (r_5 - g_e) % 112;
                        #pragma unroll
                        for (int i_12 = 0; i_12 < 14; i_12++) {
                            int kk2_2 = (lane + i_12 * 32) * 8;
                            float y8_2[8];
                            #pragma unroll
                            for (int k_9 = 0; k_9 < 4; k_9++) {
                                float _bf16x2_add_f32_5[2];
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b16 lo, hi;\n\t"
                                    "mov.b32 {lo, hi}, %2;\n\t"
                                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                    "}\n"
                                    : "=&f"(_bf16x2_add_f32_5[0]), "=&f"(_bf16x2_add_f32_5[1]) : "r"(xp_2[i_12 * 4 + k_9]), "f"(0.0f), "f"(0.0f));
                                y8_2[2 * k_9] = _bf16x2_add_f32_5[0] * rstd_2;
                                y8_2[2 * k_9 + 1] = _bf16x2_add_f32_5[1] * rstd_2;
                            }
                            uint32_t y8_bf16_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y8_2[_lp*2 + 0], y8_2[_lp*2+1 + 0]));
                                y8_bf16_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int ow_2[4];
                            #pragma unroll
                            for (int k_10 = 0; k_10 < 4; k_10++) {
                                uint32_t _bf16x2_mul_2;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(nwp[i_12 * 4 + k_10]), "r"(y8_bf16_2[k_10]));
                                ow_2[k_10] = _bf16x2_mul_2;
                            }
                            if (own_2 == 0) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(y_out + (nrow_base_2 + (unsigned long long)kk2_2) + (0)), "r"(ow_2[(0) + 0]), "r"(ow_2[(0) + 1]), "r"(ow_2[(0) + 2]), "r"(ow_2[(0) + 3]) : "memory");
                            }
                            int c_rel_2 = i_12 * 4 + lane / 8 - c_lo;
                            if (c_rel_2 >= 0) {
                                if (c_rel_2 < c_cnt) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(c_rel_2 * 2048) + (unsigned int)(r_5 * 128 + col_b ^ (r_5 * 128 + col_b >> 7 & 7) << 4))), "r"(ow_2[0]), "r"(ow_2[1]), "r"(ow_2[2]), "r"(ow_2[3]) : "memory");
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int i_13 = 0; i_13 < 14; i_13++) {
                            int c_relz_2 = i_13 * 4 + lane / 8 - c_lo;
                            if (c_relz_2 >= 0) {
                                if (c_relz_2 < c_cnt) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(c_relz_2 * 2048) + (unsigned int)(r_5 * 128 + col_b ^ (r_5 * 128 + col_b >> 7 & 7) << 4))), "r"(zero_w), "r"(zero_w), "r"(zero_w), "r"(zero_w) : "memory");
                                }
                            }
                        }
                    }
                    int r_6 = epi_warp + 12;
                    if (r_6 < num_tokens) {
                        unsigned long long nrow_base_3 = (unsigned long long)r_6 * 3584;
                        float sum_sq_3 = 0.0f;
                        #pragma unroll
                        for (int i_14 = 0; i_14 < 56; i_14++) {
                            float _bf16x2_add_f32_6[2];
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                "}\n"
                                : "=&f"(_bf16x2_add_f32_6[0]), "=&f"(_bf16x2_add_f32_6[1]) : "r"(xp_2[56 + i_14]), "f"(0.0f), "f"(0.0f));
                            sum_sq_3 += _bf16x2_add_f32_6[0] * _bf16x2_add_f32_6[0];
                            sum_sq_3 += _bf16x2_add_f32_6[1] * _bf16x2_add_f32_6[1];
                        }
                        float _warp_reduce_3 = sum_sq_3;
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset >>= 1)
                            _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
                        float total_3 = _warp_reduce_3;
                        float _rsqrt_3 = rsqrtf(total_3 / 3584.0f + eps);
                        float rstd_3 = _rsqrt_3;
                        int own_3 = (r_6 - g_e) % 112;
                        #pragma unroll
                        for (int i_15 = 0; i_15 < 14; i_15++) {
                            int kk2_3 = (lane + i_15 * 32) * 8;
                            float y8_3[8];
                            #pragma unroll
                            for (int k_11 = 0; k_11 < 4; k_11++) {
                                float _bf16x2_add_f32_7[2];
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b16 lo, hi;\n\t"
                                    "mov.b32 {lo, hi}, %2;\n\t"
                                    "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                    "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                    "}\n"
                                    : "=&f"(_bf16x2_add_f32_7[0]), "=&f"(_bf16x2_add_f32_7[1]) : "r"(xp_2[56 + i_15 * 4 + k_11]), "f"(0.0f), "f"(0.0f));
                                y8_3[2 * k_11] = _bf16x2_add_f32_7[0] * rstd_3;
                                y8_3[2 * k_11 + 1] = _bf16x2_add_f32_7[1] * rstd_3;
                            }
                            uint32_t y8_bf16_3[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y8_3[_lp*2 + 0], y8_3[_lp*2+1 + 0]));
                                y8_bf16_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int ow_3[4];
                            #pragma unroll
                            for (int k_12 = 0; k_12 < 4; k_12++) {
                                uint32_t _bf16x2_mul_3;
                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(nwp[i_15 * 4 + k_12]), "r"(y8_bf16_3[k_12]));
                                ow_3[k_12] = _bf16x2_mul_3;
                            }
                            if (own_3 == 0) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(y_out + (nrow_base_3 + (unsigned long long)kk2_3) + (0)), "r"(ow_3[(0) + 0]), "r"(ow_3[(0) + 1]), "r"(ow_3[(0) + 2]), "r"(ow_3[(0) + 3]) : "memory");
                            }
                            int c_rel_3 = i_15 * 4 + lane / 8 - c_lo;
                            if (c_rel_3 >= 0) {
                                if (c_rel_3 < c_cnt) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(c_rel_3 * 2048) + (unsigned int)(r_6 * 128 + col_b ^ (r_6 * 128 + col_b >> 7 & 7) << 4))), "r"(ow_3[0]), "r"(ow_3[1]), "r"(ow_3[2]), "r"(ow_3[3]) : "memory");
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int i_16 = 0; i_16 < 14; i_16++) {
                            int c_relz_3 = i_16 * 4 + lane / 8 - c_lo;
                            if (c_relz_3 >= 0) {
                                if (c_relz_3 < c_cnt) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_bn_addr + (unsigned int)(c_relz_3 * 2048) + (unsigned int)(r_6 * 128 + col_b ^ (r_6 * 128 + col_b >> 7 & 7) << 4))), "r"(zero_w), "r"(zero_w), "r"(zero_w), "r"(zero_w) : "memory");
                                }
                            }
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(bn_bar_addr);
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
            float red[16];
            int fin = 1;
            unsigned int _phase_acc_full = 0;
            mbarrier_wait(acc_full_addr + (epi_stage) * 8, _phase_acc_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            {
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], lane_addr);
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
            int d_lo_l = ((rank_l == 0) ? 0 : 6);
            int d_cnt_l = ((rank_l == 0) ? 6 : 6);
            int u_lo_l = ((rank_l == 0) ? 0 : 4);
            int u_cnt_l = ((rank_l == 0) ? 4 : 3);
            int u_count_l = d_cnt_l + u_cnt_l;
            unsigned int stage = 0;
            int _min_0 = ((5) < (u_count_l) ? (5) : (u_count_l));
            int _min_1 = ((3) < (u_count_l) ? (3) : (u_count_l));
            int pro = ((0) ? _min_0 : ((0) ? _min_1 : 0));
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(rows_full_addr, (num_tokens + 1) * 7168);
                cp_async_bulk_gmem2smem(smem_v7_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(norm_w) + ((unsigned long long)0 * (unsigned long long)2)), 7168, rows_full_addr);
                #pragma unroll 1
                for (int rr = 0; rr < num_tokens; rr++) {
                    cp_async_bulk_gmem2smem(smem_v5_addr + (unsigned int)(rr * 3584 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(routed) + ((unsigned long long)((unsigned long long)rr * 3584) * (unsigned long long)2)), 7168, rows_full_addr);
                }
            }
            unsigned int _phase_a_empty = 1;
            if (elect_sync()) {
                int norm_ok = ((0) ? 1 : 0);
                #pragma unroll 1
                for (int s = 0; s < u_count_l; s++) {
                    mbarrier_wait(a_empty_addr + (stage) * 8, _phase_a_empty);
                    int tile_1 = tile_l;
                    int m = ((d_cnt_l > s) ? d_lo_l + s : 12 + u_lo_l + (s - d_cnt_l));
                    int cls_1 = ((tile_1 < 0) ? 0 : ((tile_1 < 56) ? 1 : 2));
                    int row_r_1 = tile_1 * 128;
                    int row_l_1 = tile_1 * 128;
                    int row_s_1 = (tile_1 - 56) * 64;
                    int need_a = ((pro > s) ? 0 : 1);
                    if (m < 12) {
                        int kb2 = m;
                        tma_3d_gmem2smem(smem_b_addr + stage * 2048, (&B_2), 0, 0, kb2, a_full_addr + (stage) * 8);
                    } else if (!1) {
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
                        int kb = k1_off + (m - 12);
                        tma_3d_gmem2smem(smem_b_addr + stage * 2048, (&B_1), 0, 0, kb, a_full_addr + (stage) * 8);
                    }
                    if (need_a == 1) {
                        if (m < 12) {
                            int kc2 = m;
                            tma_3d_gmem2smem(smem_a_addr + stage * 16384, (&A_2), 0, row_l_1, kc2, a_full_addr + (stage) * 8);
                        } else {
                            int kc = k1_off + (m - 12);
                            tma_3d_gmem2smem(smem_a_addr + stage * 16384, (&A_L), 0, row_l_1, kc, a_full_addr + (stage) * 8);
                        }
                    }
                    {
                        int txb = ((m < 12) ? 18432 : 16384);
                        mbarrier_arrive_expect_tx(a_full_addr + (stage) * 8, txb);
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
            int u_count_m = ((rank_m == 0) ? 10 : 9);
            int d_cnt_m = ((rank_m == 0) ? 6 : 6);
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
                    if (m_1 == d_cnt_m) {
                        mbarrier_wait(bn_bar_addr, 0);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((m_1 == 0) ? 1 : 0);
                    {
                        int init_sub = ((1) ? init_flag : 0);
                        if (d_cnt_m > m_1) {
                            int _mma_a_lo_3 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_3 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 128;
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
                    "mov.b32 id, 134481040;\n\t"
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_acc), "r"(((init_sub) ? 0 : 1)));
                        } else {
                            unsigned int ub = (unsigned int)(m_1 - d_cnt_m);
                            int _mma_a_lo_4 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_4 = (((smem_bn_addr) >> 4) & 0x3FFF) + (ub) * 128;
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
                    "mov.b32 id, 134481040;\n\t"
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_acc), "r"(((init_sub) ? 0 : 1)));
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
