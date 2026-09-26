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
#define NUM_TMA_PIPE_STAGES 11
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_LAND_OFF 2048
#define SMEM_LAND_STAGE_BYTES 1024
#define SMEM_LAND_STRIDE 1024
#define SMEM_SMEM_A_OFF 3072
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 183296
#define SMEM_SMEM_B_STAGE_BYTES 4096
#define SMEM_SMEM_B_STRIDE 4096
#define SMEM_SMEM_BN_OFF 228352
#define SMEM_SMEM_BN_STAGE_BYTES 4096
#define SMEM_SMEM_BN_STRIDE 4096
#define SMEM_SMEM_AG_OFF 3072
#define SMEM_SMEM_AG_STAGE_BYTES 8192
#define SMEM_SMEM_AG_STRIDE 16384
#define SMEM_SMEM_AU_OFF 11264
#define SMEM_SMEM_AU_STAGE_BYTES 8192
#define SMEM_SMEM_AU_STRIDE 16384
#define SMEM_TOTAL 228352

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

__global__ __launch_bounds__(192) void
kernel_cake_kimi_k3_latent_moe_42d584e96e3f05e3f095(const __grid_constant__ CUtensorMap A_R, const __grid_constant__ CUtensorMap A_L, const __grid_constant__ CUtensorMap A_S, const __grid_constant__ CUtensorMap A_2, const __grid_constant__ CUtensorMap B_1, const __grid_constant__ CUtensorMap B_2, float* __restrict__ out_r, __nv_bfloat16* __restrict__ out_l, __nv_bfloat16* __restrict__ out_s, unsigned int* __restrict__ counters, __nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out, unsigned long long* __restrict__ tl, int num_tokens, int k1_off, int num_partials, float eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define a_full_addr (mbar_base + 0)
    #define a_empty_addr (mbar_base + 88)
    #define acc_full_addr (mbar_base + 176)
    #define acc_empty_addr (mbar_base + 184)
    #define red_bar_addr (mbar_base + 192)
    #define issue_bar_addr (mbar_base + 200)
    #define rows_bar_addr (mbar_base + 208)
    #define bn_bar_addr (mbar_base + 216)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* land = reinterpret_cast<float*>(smem_raw + 2048);
    const int land_addr = smem + 2048;
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_a_addr = smem + 3072;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 183296);
    const int smem_b_addr = smem + 183296;
    __nv_bfloat16* smem_bn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 228352);
    const int smem_bn_addr = smem + 228352;
    __nv_bfloat16* smem_ag = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_ag_addr = smem + 3072;
    __nv_bfloat16* smem_au = reinterpret_cast<__nv_bfloat16*>(smem_raw + 11264);
    const int smem_au_addr = smem + 11264;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 28 barriers)
    // Mbarriers at smem_raw[0..224)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // a_full: 11 barriers, init_count=1
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
            // a_empty: 11 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // --- pipeline 'epi_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // acc_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            // red_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // issue_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            // rows_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            // bn_bar: 1 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 0) {
        int _tmem_hold = smem + 224;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
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
            int cls = ((tile < 7) ? 0 : ((tile < 35) ? 1 : 2));
            int row_r = tile * 128;
            int row_l = (tile - 7) * 128;
            int row_s = (tile - 7 - 28) * 64;
            int lane_pair = lane % 4;
            int row_base = epi_warp * 16 + lane / 4;
            int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            unsigned int epi_stage = 0;
            float red[32];
            int fin = 1;
            unsigned int _phase_acc_full = 0;
            mbarrier_wait(acc_full_addr + (epi_stage) * 8, _phase_acc_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (cls == 2) {
                float _tmem_load_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                    : "r"(taddr + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                    : "r"(taddr + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                red[0] = _tmem_load_0[0];
                red[16] = _tmem_load_1[0];
                red[1] = _tmem_load_0[1];
                red[17] = _tmem_load_1[1];
                red[2] = _tmem_load_0[2];
                red[18] = _tmem_load_1[2];
                red[3] = _tmem_load_0[3];
                red[19] = _tmem_load_1[3];
                red[4] = _tmem_load_0[4];
                red[20] = _tmem_load_1[4];
                red[5] = _tmem_load_0[5];
                red[21] = _tmem_load_1[5];
                red[6] = _tmem_load_0[6];
                red[22] = _tmem_load_1[6];
                red[7] = _tmem_load_0[7];
                red[23] = _tmem_load_1[7];
                red[8] = _tmem_load_0[8];
                red[24] = _tmem_load_1[8];
                red[9] = _tmem_load_0[9];
                red[25] = _tmem_load_1[9];
                red[10] = _tmem_load_0[10];
                red[26] = _tmem_load_1[10];
                red[11] = _tmem_load_0[11];
                red[27] = _tmem_load_1[11];
                red[12] = _tmem_load_0[12];
                red[28] = _tmem_load_1[12];
                red[13] = _tmem_load_0[13];
                red[29] = _tmem_load_1[13];
                red[14] = _tmem_load_0[14];
                red[30] = _tmem_load_1[14];
                red[15] = _tmem_load_0[15];
                red[31] = _tmem_load_1[15];
            } else {
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(lane_addr));
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
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(acc_empty_addr + (epi_stage) * 8);
                }
            }
            if (fin == 1) {
                if (cls == 2) {
                    int tok = lane_pair * 2;
                    int frow = row_base + ((0) ? 8 : 0);
                    if (tok < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_36 = __float2bfloat16(red[0]);
                        float _cvt_f32_36 = __bfloat162float(_cvt_bf16_36);
                        float gk = _cvt_f32_36;
                        __nv_bfloat16 _cvt_bf16_37 = __float2bfloat16(red[16]);
                        float _cvt_f32_37 = __bfloat162float(_cvt_bf16_37);
                        float uk = _cvt_f32_37;
                        float _exp2_0 = approx_exp2((-gk) * 1.4426950408889634f);
                        float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                        float sig = _rcp_0;
                        float _tanh_0 = tanhf(gk * 0.25f);
                        float aa = 4.0f * _tanh_0 * sig;
                        float _tanh_1 = tanhf(uk * 0.04f);
                        float bb = 25.0f * _tanh_1;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok * 6144 + row_s + frow)) + (0)) = __float2bfloat16_rn(aa * bb);
                    }
                    int tok_0 = lane_pair * 2 + 1;
                    int frow_1 = row_base + ((0) ? 8 : 0);
                    if (tok_0 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_38 = __float2bfloat16(red[1]);
                        float _cvt_f32_38 = __bfloat162float(_cvt_bf16_38);
                        float gk_1 = _cvt_f32_38;
                        __nv_bfloat16 _cvt_bf16_39 = __float2bfloat16(red[17]);
                        float _cvt_f32_39 = __bfloat162float(_cvt_bf16_39);
                        float uk_1 = _cvt_f32_39;
                        float _exp2_1 = approx_exp2((-gk_1) * 1.4426950408889634f);
                        float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                        float sig_1 = _rcp_1;
                        float _tanh_2 = tanhf(gk_1 * 0.25f);
                        float aa_1 = 4.0f * _tanh_2 * sig_1;
                        float _tanh_3 = tanhf(uk_1 * 0.04f);
                        float bb_1 = 25.0f * _tanh_3;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_0 * 6144 + row_s + frow_1)) + (0)) = __float2bfloat16_rn(aa_1 * bb_1);
                    }
                    int tok_2 = lane_pair * 2;
                    int frow_3 = row_base + ((1) ? 8 : 0);
                    if (tok_2 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_40 = __float2bfloat16(red[2]);
                        float _cvt_f32_40 = __bfloat162float(_cvt_bf16_40);
                        float gk_2 = _cvt_f32_40;
                        __nv_bfloat16 _cvt_bf16_41 = __float2bfloat16(red[18]);
                        float _cvt_f32_41 = __bfloat162float(_cvt_bf16_41);
                        float uk_2 = _cvt_f32_41;
                        float _exp2_2 = approx_exp2((-gk_2) * 1.4426950408889634f);
                        float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                        float sig_2 = _rcp_2;
                        float _tanh_4 = tanhf(gk_2 * 0.25f);
                        float aa_2 = 4.0f * _tanh_4 * sig_2;
                        float _tanh_5 = tanhf(uk_2 * 0.04f);
                        float bb_2 = 25.0f * _tanh_5;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_2 * 6144 + row_s + frow_3)) + (0)) = __float2bfloat16_rn(aa_2 * bb_2);
                    }
                    int tok_4 = lane_pair * 2 + 1;
                    int frow_5 = row_base + ((1) ? 8 : 0);
                    if (tok_4 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_42 = __float2bfloat16(red[3]);
                        float _cvt_f32_42 = __bfloat162float(_cvt_bf16_42);
                        float gk_3 = _cvt_f32_42;
                        __nv_bfloat16 _cvt_bf16_43 = __float2bfloat16(red[19]);
                        float _cvt_f32_43 = __bfloat162float(_cvt_bf16_43);
                        float uk_3 = _cvt_f32_43;
                        float _exp2_3 = approx_exp2((-gk_3) * 1.4426950408889634f);
                        float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                        float sig_3 = _rcp_3;
                        float _tanh_6 = tanhf(gk_3 * 0.25f);
                        float aa_3 = 4.0f * _tanh_6 * sig_3;
                        float _tanh_7 = tanhf(uk_3 * 0.04f);
                        float bb_3 = 25.0f * _tanh_7;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_4 * 6144 + row_s + frow_5)) + (0)) = __float2bfloat16_rn(aa_3 * bb_3);
                    }
                    int tok_6 = 8 + lane_pair * 2;
                    int frow_7 = row_base + ((0) ? 8 : 0);
                    if (tok_6 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_44 = __float2bfloat16(red[4]);
                        float _cvt_f32_44 = __bfloat162float(_cvt_bf16_44);
                        float gk_4 = _cvt_f32_44;
                        __nv_bfloat16 _cvt_bf16_45 = __float2bfloat16(red[20]);
                        float _cvt_f32_45 = __bfloat162float(_cvt_bf16_45);
                        float uk_4 = _cvt_f32_45;
                        float _exp2_4 = approx_exp2((-gk_4) * 1.4426950408889634f);
                        float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                        float sig_4 = _rcp_4;
                        float _tanh_8 = tanhf(gk_4 * 0.25f);
                        float aa_4 = 4.0f * _tanh_8 * sig_4;
                        float _tanh_9 = tanhf(uk_4 * 0.04f);
                        float bb_4 = 25.0f * _tanh_9;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_6 * 6144 + row_s + frow_7)) + (0)) = __float2bfloat16_rn(aa_4 * bb_4);
                    }
                    int tok_8 = 8 + lane_pair * 2 + 1;
                    int frow_9 = row_base + ((0) ? 8 : 0);
                    if (tok_8 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_46 = __float2bfloat16(red[5]);
                        float _cvt_f32_46 = __bfloat162float(_cvt_bf16_46);
                        float gk_5 = _cvt_f32_46;
                        __nv_bfloat16 _cvt_bf16_47 = __float2bfloat16(red[21]);
                        float _cvt_f32_47 = __bfloat162float(_cvt_bf16_47);
                        float uk_5 = _cvt_f32_47;
                        float _exp2_5 = approx_exp2((-gk_5) * 1.4426950408889634f);
                        float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                        float sig_5 = _rcp_5;
                        float _tanh_10 = tanhf(gk_5 * 0.25f);
                        float aa_5 = 4.0f * _tanh_10 * sig_5;
                        float _tanh_11 = tanhf(uk_5 * 0.04f);
                        float bb_5 = 25.0f * _tanh_11;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_8 * 6144 + row_s + frow_9)) + (0)) = __float2bfloat16_rn(aa_5 * bb_5);
                    }
                    int tok_10 = 8 + lane_pair * 2;
                    int frow_11 = row_base + ((1) ? 8 : 0);
                    if (tok_10 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_48 = __float2bfloat16(red[6]);
                        float _cvt_f32_48 = __bfloat162float(_cvt_bf16_48);
                        float gk_6 = _cvt_f32_48;
                        __nv_bfloat16 _cvt_bf16_49 = __float2bfloat16(red[22]);
                        float _cvt_f32_49 = __bfloat162float(_cvt_bf16_49);
                        float uk_6 = _cvt_f32_49;
                        float _exp2_6 = approx_exp2((-gk_6) * 1.4426950408889634f);
                        float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                        float sig_6 = _rcp_6;
                        float _tanh_12 = tanhf(gk_6 * 0.25f);
                        float aa_6 = 4.0f * _tanh_12 * sig_6;
                        float _tanh_13 = tanhf(uk_6 * 0.04f);
                        float bb_6 = 25.0f * _tanh_13;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_10 * 6144 + row_s + frow_11)) + (0)) = __float2bfloat16_rn(aa_6 * bb_6);
                    }
                    int tok_12 = 8 + lane_pair * 2 + 1;
                    int frow_13 = row_base + ((1) ? 8 : 0);
                    if (tok_12 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_50 = __float2bfloat16(red[7]);
                        float _cvt_f32_50 = __bfloat162float(_cvt_bf16_50);
                        float gk_7 = _cvt_f32_50;
                        __nv_bfloat16 _cvt_bf16_51 = __float2bfloat16(red[23]);
                        float _cvt_f32_51 = __bfloat162float(_cvt_bf16_51);
                        float uk_7 = _cvt_f32_51;
                        float _exp2_7 = approx_exp2((-gk_7) * 1.4426950408889634f);
                        float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                        float sig_7 = _rcp_7;
                        float _tanh_14 = tanhf(gk_7 * 0.25f);
                        float aa_7 = 4.0f * _tanh_14 * sig_7;
                        float _tanh_15 = tanhf(uk_7 * 0.04f);
                        float bb_7 = 25.0f * _tanh_15;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_12 * 6144 + row_s + frow_13)) + (0)) = __float2bfloat16_rn(aa_7 * bb_7);
                    }
                    int tok_14 = 16 + lane_pair * 2;
                    int frow_15 = row_base + ((0) ? 8 : 0);
                    if (tok_14 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_52 = __float2bfloat16(red[8]);
                        float _cvt_f32_52 = __bfloat162float(_cvt_bf16_52);
                        float gk_8 = _cvt_f32_52;
                        __nv_bfloat16 _cvt_bf16_53 = __float2bfloat16(red[24]);
                        float _cvt_f32_53 = __bfloat162float(_cvt_bf16_53);
                        float uk_8 = _cvt_f32_53;
                        float _exp2_8 = approx_exp2((-gk_8) * 1.4426950408889634f);
                        float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                        float sig_8 = _rcp_8;
                        float _tanh_16 = tanhf(gk_8 * 0.25f);
                        float aa_8 = 4.0f * _tanh_16 * sig_8;
                        float _tanh_17 = tanhf(uk_8 * 0.04f);
                        float bb_8 = 25.0f * _tanh_17;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_14 * 6144 + row_s + frow_15)) + (0)) = __float2bfloat16_rn(aa_8 * bb_8);
                    }
                    int tok_16 = 16 + lane_pair * 2 + 1;
                    int frow_17 = row_base + ((0) ? 8 : 0);
                    if (tok_16 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_54 = __float2bfloat16(red[9]);
                        float _cvt_f32_54 = __bfloat162float(_cvt_bf16_54);
                        float gk_9 = _cvt_f32_54;
                        __nv_bfloat16 _cvt_bf16_55 = __float2bfloat16(red[25]);
                        float _cvt_f32_55 = __bfloat162float(_cvt_bf16_55);
                        float uk_9 = _cvt_f32_55;
                        float _exp2_9 = approx_exp2((-gk_9) * 1.4426950408889634f);
                        float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                        float sig_9 = _rcp_9;
                        float _tanh_18 = tanhf(gk_9 * 0.25f);
                        float aa_9 = 4.0f * _tanh_18 * sig_9;
                        float _tanh_19 = tanhf(uk_9 * 0.04f);
                        float bb_9 = 25.0f * _tanh_19;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_16 * 6144 + row_s + frow_17)) + (0)) = __float2bfloat16_rn(aa_9 * bb_9);
                    }
                    int tok_18 = 16 + lane_pair * 2;
                    int frow_19 = row_base + ((1) ? 8 : 0);
                    if (tok_18 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_56 = __float2bfloat16(red[10]);
                        float _cvt_f32_56 = __bfloat162float(_cvt_bf16_56);
                        float gk_10 = _cvt_f32_56;
                        __nv_bfloat16 _cvt_bf16_57 = __float2bfloat16(red[26]);
                        float _cvt_f32_57 = __bfloat162float(_cvt_bf16_57);
                        float uk_10 = _cvt_f32_57;
                        float _exp2_10 = approx_exp2((-gk_10) * 1.4426950408889634f);
                        float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                        float sig_10 = _rcp_10;
                        float _tanh_20 = tanhf(gk_10 * 0.25f);
                        float aa_10 = 4.0f * _tanh_20 * sig_10;
                        float _tanh_21 = tanhf(uk_10 * 0.04f);
                        float bb_10 = 25.0f * _tanh_21;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_18 * 6144 + row_s + frow_19)) + (0)) = __float2bfloat16_rn(aa_10 * bb_10);
                    }
                    int tok_20 = 16 + lane_pair * 2 + 1;
                    int frow_21 = row_base + ((1) ? 8 : 0);
                    if (tok_20 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_58 = __float2bfloat16(red[11]);
                        float _cvt_f32_58 = __bfloat162float(_cvt_bf16_58);
                        float gk_11 = _cvt_f32_58;
                        __nv_bfloat16 _cvt_bf16_59 = __float2bfloat16(red[27]);
                        float _cvt_f32_59 = __bfloat162float(_cvt_bf16_59);
                        float uk_11 = _cvt_f32_59;
                        float _exp2_11 = approx_exp2((-gk_11) * 1.4426950408889634f);
                        float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                        float sig_11 = _rcp_11;
                        float _tanh_22 = tanhf(gk_11 * 0.25f);
                        float aa_11 = 4.0f * _tanh_22 * sig_11;
                        float _tanh_23 = tanhf(uk_11 * 0.04f);
                        float bb_11 = 25.0f * _tanh_23;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_20 * 6144 + row_s + frow_21)) + (0)) = __float2bfloat16_rn(aa_11 * bb_11);
                    }
                    int tok_22 = 24 + lane_pair * 2;
                    int frow_23 = row_base + ((0) ? 8 : 0);
                    if (tok_22 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_60 = __float2bfloat16(red[12]);
                        float _cvt_f32_60 = __bfloat162float(_cvt_bf16_60);
                        float gk_12 = _cvt_f32_60;
                        __nv_bfloat16 _cvt_bf16_61 = __float2bfloat16(red[28]);
                        float _cvt_f32_61 = __bfloat162float(_cvt_bf16_61);
                        float uk_12 = _cvt_f32_61;
                        float _exp2_12 = approx_exp2((-gk_12) * 1.4426950408889634f);
                        float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                        float sig_12 = _rcp_12;
                        float _tanh_24 = tanhf(gk_12 * 0.25f);
                        float aa_12 = 4.0f * _tanh_24 * sig_12;
                        float _tanh_25 = tanhf(uk_12 * 0.04f);
                        float bb_12 = 25.0f * _tanh_25;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_22 * 6144 + row_s + frow_23)) + (0)) = __float2bfloat16_rn(aa_12 * bb_12);
                    }
                    int tok_24 = 24 + lane_pair * 2 + 1;
                    int frow_25 = row_base + ((0) ? 8 : 0);
                    if (tok_24 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_62 = __float2bfloat16(red[13]);
                        float _cvt_f32_62 = __bfloat162float(_cvt_bf16_62);
                        float gk_13 = _cvt_f32_62;
                        __nv_bfloat16 _cvt_bf16_63 = __float2bfloat16(red[29]);
                        float _cvt_f32_63 = __bfloat162float(_cvt_bf16_63);
                        float uk_13 = _cvt_f32_63;
                        float _exp2_13 = approx_exp2((-gk_13) * 1.4426950408889634f);
                        float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                        float sig_13 = _rcp_13;
                        float _tanh_26 = tanhf(gk_13 * 0.25f);
                        float aa_13 = 4.0f * _tanh_26 * sig_13;
                        float _tanh_27 = tanhf(uk_13 * 0.04f);
                        float bb_13 = 25.0f * _tanh_27;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_24 * 6144 + row_s + frow_25)) + (0)) = __float2bfloat16_rn(aa_13 * bb_13);
                    }
                    int tok_26 = 24 + lane_pair * 2;
                    int frow_27 = row_base + ((1) ? 8 : 0);
                    if (tok_26 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_64 = __float2bfloat16(red[14]);
                        float _cvt_f32_64 = __bfloat162float(_cvt_bf16_64);
                        float gk_14 = _cvt_f32_64;
                        __nv_bfloat16 _cvt_bf16_65 = __float2bfloat16(red[30]);
                        float _cvt_f32_65 = __bfloat162float(_cvt_bf16_65);
                        float uk_14 = _cvt_f32_65;
                        float _exp2_14 = approx_exp2((-gk_14) * 1.4426950408889634f);
                        float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                        float sig_14 = _rcp_14;
                        float _tanh_28 = tanhf(gk_14 * 0.25f);
                        float aa_14 = 4.0f * _tanh_28 * sig_14;
                        float _tanh_29 = tanhf(uk_14 * 0.04f);
                        float bb_14 = 25.0f * _tanh_29;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_26 * 6144 + row_s + frow_27)) + (0)) = __float2bfloat16_rn(aa_14 * bb_14);
                    }
                    int tok_28 = 24 + lane_pair * 2 + 1;
                    int frow_29 = row_base + ((1) ? 8 : 0);
                    if (tok_28 < num_tokens) {
                        __nv_bfloat16 _cvt_bf16_66 = __float2bfloat16(red[15]);
                        float _cvt_f32_66 = __bfloat162float(_cvt_bf16_66);
                        float gk_15 = _cvt_f32_66;
                        __nv_bfloat16 _cvt_bf16_67 = __float2bfloat16(red[31]);
                        float _cvt_f32_67 = __bfloat162float(_cvt_bf16_67);
                        float uk_15 = _cvt_f32_67;
                        float _exp2_15 = approx_exp2((-gk_15) * 1.4426950408889634f);
                        float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                        float sig_15 = _rcp_15;
                        float _tanh_30 = tanhf(gk_15 * 0.25f);
                        float aa_15 = 4.0f * _tanh_30 * sig_15;
                        float _tanh_31 = tanhf(uk_15 * 0.04f);
                        float bb_15 = 25.0f * _tanh_31;
                        *(reinterpret_cast<__nv_bfloat16*>(out_s + (tok_28 * 6144 + row_s + frow_29)) + (0)) = __float2bfloat16_rn(aa_15 * bb_15);
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
            int d_lo_l = ((rank_l == 0) ? 0 : 0);
            int d_cnt_l = ((rank_l == 0) ? 0 : 0);
            int u_lo_l = ((rank_l == 0) ? 0 : 112);
            int u_cnt_l = ((rank_l == 0) ? 112 : 0);
            int u_count_l = d_cnt_l + u_cnt_l;
            unsigned int stage = 0;
            int _min_0 = ((11) < (u_count_l) ? (11) : (u_count_l));
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
                        tma_3d_gmem2smem(smem_b_addr + stage * 4096, (&B_1), 0, 0, kb1, a_full_addr + (stage) * 8);
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
                                tma_3d_gmem2smem(smem_a_addr + stage * 16384 + 8192, (&A_S), 0, 6144 + row_s_1, kc1, a_full_addr + (stage) * 8);
                            }
                        }
                    }
                    {
                        mbarrier_arrive_expect_tx(a_full_addr + (stage) * 8, 20480);
                    }
                    stage += 1;
                    if (stage == 11) { stage = 0; _phase_a_empty ^= 1; }
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
            int u_count_m = ((rank_m == 0) ? 112 : 0);
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
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 256;
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
                    "mov.b32 id, 67634320;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_acc + (32))), "r"(((init_sub) ? 0 : 1)));
                            int _mma_a_lo_1 = (((smem_au_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 256;
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
                    "mov.b32 id, 67634320;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_acc + (64))), "r"(((init_sub) ? 0 : 1)));
                        } else {
                            int _mma_a_lo_2 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 256;
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_acc), "r"(((init_sub) ? 0 : 1)));
                        }
                    }
                    tcgen05_commit(a_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 11) { stage_1 = 0; _phase_a_full ^= 1; }
                }
                tcgen05_commit(acc_full_addr + (epi_stage_1) * 8);
            }
        }
    }

    // Cleanup
}

} // extern "C"
