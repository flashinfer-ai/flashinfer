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
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 336
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 16
#define TMEM_SFB_OFFSET 176
#define NUM_K_PIPE_STAGES 5
#define NUM_MMA_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 164864
#define SMEM_SMEM_B_STAGE_BYTES 2048
#define SMEM_SMEM_B_STRIDE 2048
#define SMEM_SMEM_A_MMA0_OFF 1024
#define SMEM_SMEM_A_MMA0_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA0_STRIDE 32768
#define SMEM_SMEM_A_MMA1_OFF 1056
#define SMEM_SMEM_A_MMA1_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA1_STRIDE 32768
#define SMEM_SMEM_A_MMA2_OFF 1088
#define SMEM_SMEM_A_MMA2_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA2_STRIDE 32768
#define SMEM_SMEM_A_MMA3_OFF 1120
#define SMEM_SMEM_A_MMA3_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA3_STRIDE 32768
#define SMEM_SMEM_A_MMA4_OFF 17408
#define SMEM_SMEM_A_MMA4_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA4_STRIDE 32768
#define SMEM_SMEM_A_MMA5_OFF 17440
#define SMEM_SMEM_A_MMA5_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA5_STRIDE 32768
#define SMEM_SMEM_A_MMA6_OFF 17472
#define SMEM_SMEM_A_MMA6_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA6_STRIDE 32768
#define SMEM_SMEM_A_MMA7_OFF 17504
#define SMEM_SMEM_A_MMA7_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA7_STRIDE 32768
#define SMEM_SMEM_B_MMA0_OFF 164864
#define SMEM_SMEM_B_MMA0_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA0_STRIDE 2048
#define SMEM_SMEM_B_MMA1_OFF 164896
#define SMEM_SMEM_B_MMA1_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA1_STRIDE 2048
#define SMEM_SMEM_B_MMA2_OFF 164928
#define SMEM_SMEM_B_MMA2_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA2_STRIDE 2048
#define SMEM_SMEM_B_MMA3_OFF 164960
#define SMEM_SMEM_B_MMA3_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA3_STRIDE 2048
#define SMEM_SMEM_B_MMA4_OFF 165888
#define SMEM_SMEM_B_MMA4_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA4_STRIDE 2048
#define SMEM_SMEM_B_MMA5_OFF 165920
#define SMEM_SMEM_B_MMA5_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA5_STRIDE 2048
#define SMEM_SMEM_B_MMA6_OFF 165952
#define SMEM_SMEM_B_MMA6_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA6_STRIDE 2048
#define SMEM_SMEM_B_MMA7_OFF 165984
#define SMEM_SMEM_B_MMA7_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA7_STRIDE 2048
#define SMEM_SMEM_A_MMA0_NEXT_OFF 1024
#define SMEM_SMEM_A_MMA0_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA0_NEXT_STRIDE 32768
#define SMEM_SMEM_B_MMA0_NEXT_OFF 164864
#define SMEM_SMEM_B_MMA0_NEXT_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA0_NEXT_STRIDE 2048
#define SMEM_SMEM_A_MMA2_NEXT_OFF 1024
#define SMEM_SMEM_A_MMA2_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA2_NEXT_STRIDE 32768
#define SMEM_SMEM_B_MMA2_NEXT_OFF 164864
#define SMEM_SMEM_B_MMA2_NEXT_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA2_NEXT_STRIDE 2048
#define SMEM_SMEM_A_MMA5_NEXT_OFF 1024
#define SMEM_SMEM_A_MMA5_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA5_NEXT_STRIDE 32768
#define SMEM_SMEM_B_MMA5_NEXT_OFF 164864
#define SMEM_SMEM_B_MMA5_NEXT_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA5_NEXT_STRIDE 2048
#define SMEM_SMEM_SFA_OFF 175104
#define SMEM_SMEM_SFA_STAGE_BYTES 4096
#define SMEM_SMEM_SFA_STRIDE 4096
#define SMEM_SMEM_SFB_OFF 195584
#define SMEM_SMEM_SFB_STAGE_BYTES 256
#define SMEM_SMEM_SFB_STRIDE 4096
#define SMEM_EPI_STAGING_OFF 216064
#define SMEM_EPI_STAGING_STAGE_BYTES 2048
#define SMEM_EPI_STAGING_STRIDE 2048
#define SMEM_EPI_STAGING_U64_OFF 216064
#define SMEM_EPI_STAGING_U64_STAGE_BYTES 2048
#define SMEM_EPI_STAGING_U64_STRIDE 2048
#define SMEM_WORK_RESPONSE_OFF 218112
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 218240
#define THREADS 384
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_cake_warp_decode_838c83072f3f6100a5f8(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C_tma, __nv_bfloat16* __restrict__ C, float* __restrict__ scale_c, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, int* __restrict__ work_counter, int M, int K, int grid_m, int grid_n, int K_tiles)
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
    #define sfa_smem_free_addr (mbar_base + 160)
    #define sfb_smem_free_addr (mbar_base + 200)
    #define tmem_sfa_full_addr (mbar_base + 240)
    #define tmem_sfb_full_addr (mbar_base + 280)
    #define k_done_addr (mbar_base + 320)
    #define mma_full_addr (mbar_base + 360)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 376);

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int smem_b_addr = smem + 164864;
    uint8_t* smem_a_mma0 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma0_addr = smem + 1024;
    uint8_t* smem_a_mma1 = reinterpret_cast<uint8_t*>(smem_raw + 1056);
    const int smem_a_mma1_addr = smem + 1056;
    uint8_t* smem_a_mma2 = reinterpret_cast<uint8_t*>(smem_raw + 1088);
    const int smem_a_mma2_addr = smem + 1088;
    uint8_t* smem_a_mma3 = reinterpret_cast<uint8_t*>(smem_raw + 1120);
    const int smem_a_mma3_addr = smem + 1120;
    uint8_t* smem_a_mma4 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_a_mma4_addr = smem + 17408;
    uint8_t* smem_a_mma5 = reinterpret_cast<uint8_t*>(smem_raw + 17440);
    const int smem_a_mma5_addr = smem + 17440;
    uint8_t* smem_a_mma6 = reinterpret_cast<uint8_t*>(smem_raw + 17472);
    const int smem_a_mma6_addr = smem + 17472;
    uint8_t* smem_a_mma7 = reinterpret_cast<uint8_t*>(smem_raw + 17504);
    const int smem_a_mma7_addr = smem + 17504;
    uint8_t* smem_b_mma0 = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int smem_b_mma0_addr = smem + 164864;
    uint8_t* smem_b_mma1 = reinterpret_cast<uint8_t*>(smem_raw + 164896);
    const int smem_b_mma1_addr = smem + 164896;
    uint8_t* smem_b_mma2 = reinterpret_cast<uint8_t*>(smem_raw + 164928);
    const int smem_b_mma2_addr = smem + 164928;
    uint8_t* smem_b_mma3 = reinterpret_cast<uint8_t*>(smem_raw + 164960);
    const int smem_b_mma3_addr = smem + 164960;
    uint8_t* smem_b_mma4 = reinterpret_cast<uint8_t*>(smem_raw + 165888);
    const int smem_b_mma4_addr = smem + 165888;
    uint8_t* smem_b_mma5 = reinterpret_cast<uint8_t*>(smem_raw + 165920);
    const int smem_b_mma5_addr = smem + 165920;
    uint8_t* smem_b_mma6 = reinterpret_cast<uint8_t*>(smem_raw + 165952);
    const int smem_b_mma6_addr = smem + 165952;
    uint8_t* smem_b_mma7 = reinterpret_cast<uint8_t*>(smem_raw + 165984);
    const int smem_b_mma7_addr = smem + 165984;
    uint8_t* smem_a_mma0_next = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma0_next_addr = smem + 1024;
    uint8_t* smem_b_mma0_next = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int smem_b_mma0_next_addr = smem + 164864;
    uint8_t* smem_a_mma2_next = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma2_next_addr = smem + 1024;
    uint8_t* smem_b_mma2_next = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int smem_b_mma2_next_addr = smem + 164864;
    uint8_t* smem_a_mma5_next = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma5_next_addr = smem + 1024;
    uint8_t* smem_b_mma5_next = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int smem_b_mma5_next_addr = smem + 164864;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 175104);
    const int smem_sfa_addr = smem + 175104;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 195584);
    const int smem_sfb_addr = smem + 195584;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 216064);
    const int epi_staging_addr = smem + 216064;
    unsigned long long* epi_staging_u64 = reinterpret_cast<unsigned long long*>(smem_raw + 216064);
    const int epi_staging_u64_addr = smem + 216064;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 218112);
    const int work_response_addr = smem + 218112;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 47 barriers)
    // Mbarriers at smem_raw[0..376)

    if (warp == 6) {
        // --- pipeline 'k_pipe' ---
        // a_full: 5 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 0 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 4) {
        // --- pipeline 'k_pipe' ---
        // b_full: 5 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 40 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 7) {
        // --- pipeline 'k_pipe' ---
        // sfa_full: 5 barriers, init_count=1
        // sfa_smem_free: 5 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 80 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 160 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 5) {
        // --- pipeline 'k_pipe' ---
        // sfb_full: 5 barriers, init_count=1
        // sfb_smem_free: 5 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 120 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 200 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 8) {
        // --- pipeline 'k_pipe' ---
        // tmem_sfa_full: 5 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 240 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 9) {
        // --- pipeline 'k_pipe' ---
        // tmem_sfb_full: 5 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 280 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        // --- pipeline 'k_pipe' ---
        // k_done: 5 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 5; _bar += 32) {
            mbarrier_init(smem + 320 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 1) {
        // --- pipeline 'mma_pipe' ---
        // mma_full: 2 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 2; _bar += 32) {
            mbarrier_init(smem + 360 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 336 used)
    if (warp == 0) {
        int _tmem_hold = smem + 376;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sfa = taddr + 16;
    const int tmem_sfb = taddr + 176;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // epilogue_main
            unsigned int acc_stage = 0;
            unsigned int work_stage = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            const int epi_warp = warp;
            int row = epi_warp * 32;
            int feature = (unsigned int)row + lane % 8 * 4 + lane / 8;
            int wide_feature = (unsigned int)row + lane / 4 * 4;
            int wide_token = lane % 4 * 2;
            __nv_bfloat16 bf = 0.0f;
            float wide_values[4];
            unsigned int wide_packed[2];
            unsigned long long wide_word = 0;
            unsigned int _phase_mma_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < ((1) ? 1 : grid_m * grid_n); _tile_iter++) {
                if (n_tile >= 4) {
                    break;
                }
                int token_base = n_tile * (unsigned int)BLOCK_N;
                int off_m = m_tile * (unsigned int)BLOCK_M;
                int expert = tile_expert[n_tile];
                float output_scale = scale_c[expert];
                int _mn_limit = n_tile * 8 + 1;
                mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                {
                    float _tmem_load_0[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                        : "r"(taddr + acc_stage * 8));
                    float _tmem_load_1[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                        : "r"(taddr + 1048576 + acc_stage * 8));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    int token = wide_token;
                    wide_values[0] = _tmem_load_0[0] * output_scale;
                    wide_values[1] = _tmem_load_0[2] * output_scale;
                    wide_values[2] = _tmem_load_1[0] * output_scale;
                    wide_values[3] = _tmem_load_1[2] * output_scale;
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                        wide_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                    if (token < _mn_limit - token_base) {
                        *(reinterpret_cast<unsigned long long*>(C + ((token_base + token) * M + off_m + wide_feature)) + (0)) = wide_word;
                    }
                    int token_0 = wide_token + 1;
                    wide_values[0] = _tmem_load_0[1] * output_scale;
                    wide_values[1] = _tmem_load_0[3] * output_scale;
                    wide_values[2] = _tmem_load_1[1] * output_scale;
                    wide_values[3] = _tmem_load_1[3] * output_scale;
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                        wide_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                    if (token_0 < _mn_limit - token_base) {
                        *(reinterpret_cast<unsigned long long*>(C + ((token_base + token_0) * M + off_m + wide_feature)) + (0)) = wide_word;
                    }
                }
                unsigned int static_valid = 0;
                unsigned int static_x = 0;
                unsigned int static_y = 0;
                unsigned int valid = static_valid;
                m_tile = static_x;
                n_tile = static_y;
                if (valid == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 4) {
        { // load_b_main
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_k_done = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < ((1) ? 1 : grid_m * grid_n); _tile_iter_1++) {
                if (n_tile_1 >= 4) {
                    break;
                }
                int route_k_extent = 6;
                int _min_2 = ((route_k_extent) < (5) ? (route_k_extent) : (5));
                #pragma unroll 1
                for (int route_k = ((1) ? 0 : 5); route_k < ((1) ? _min_2 : route_k_extent); route_k++) {
                    int iter_k = route_k;
                    int route_tile = n_tile_1;
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_b_addr + stage * 2048, (&B), 0, 0, iter_k * 2, route_tile, b_full_addr + (stage) * 8);
                        mbarrier_arrive_expect_tx(b_full_addr + (stage) * 8, 2048);
                    }
                    stage += 1;
                    if (stage == 5) { stage = 0; _phase_k_done ^= 1; }
                }
                int _min_3 = ((route_k_extent) < (5) ? (route_k_extent) : (5));
                #pragma unroll 1
                for (int route_k_1 = ((!1) ? 0 : 5); route_k_1 < ((0) ? _min_3 : route_k_extent); route_k_1++) {
                    int iter_k_1 = route_k_1;
                    int route_tile_1 = n_tile_1;
                    {
                        mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    }
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_b_addr + stage * 2048, (&B), 0, 0, iter_k_1 * 2, route_tile_1, b_full_addr + (stage) * 8);
                        mbarrier_arrive_expect_tx(b_full_addr + (stage) * 8, 2048);
                    }
                    stage += 1;
                    if (stage == 5) { stage = 0; _phase_k_done ^= 1; }
                }
                unsigned int static_valid_1 = 0;
                unsigned int static_x_1 = 0;
                unsigned int static_y_1 = 0;
                unsigned int valid_1 = static_valid_1;
                m_tile_1 = static_x_1;
                n_tile_1 = static_y_1;
                if (valid_1 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 5) {
        { // load_sfb_main
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_sfb_smem_free = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < ((1) ? 1 : grid_m * grid_n); _tile_iter_2++) {
                if (n_tile_2 >= 4) {
                    break;
                }
                int route_k_extent_1 = 6;
                int _min_6 = ((route_k_extent_1) < (5) ? (route_k_extent_1) : (5));
                #pragma unroll 1
                for (int route_k_2 = ((1) ? 0 : 5); route_k_2 < ((1) ? _min_6 : route_k_extent_1); route_k_2++) {
                    int iter_k_2 = route_k_2;
                    int route_tile_2 = n_tile_2;
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_sfb_addr + stage_1 * 4096, (&SFB), 0, 0, iter_k_2 * 8, route_tile_2, sfb_full_addr + (stage_1) * 8);
                        mbarrier_arrive_expect_tx(sfb_full_addr + (stage_1) * 8, 4096);
                    }
                    stage_1 += 1;
                    if (stage_1 == 5) { stage_1 = 0; _phase_sfb_smem_free ^= 1; }
                }
                int _min_7 = ((route_k_extent_1) < (5) ? (route_k_extent_1) : (5));
                #pragma unroll 1
                for (int route_k_3 = ((!1) ? 0 : 5); route_k_3 < ((0) ? _min_7 : route_k_extent_1); route_k_3++) {
                    int iter_k_3 = route_k_3;
                    int route_tile_3 = n_tile_2;
                    {
                        mbarrier_wait(sfb_smem_free_addr + (stage_1) * 8, _phase_sfb_smem_free);
                    }
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_sfb_addr + stage_1 * 4096, (&SFB), 0, 0, iter_k_3 * 8, route_tile_3, sfb_full_addr + (stage_1) * 8);
                        mbarrier_arrive_expect_tx(sfb_full_addr + (stage_1) * 8, 4096);
                    }
                    stage_1 += 1;
                    if (stage_1 == 5) { stage_1 = 0; _phase_sfb_smem_free ^= 1; }
                }
                unsigned int static_valid_2 = 0;
                unsigned int static_x_2 = 0;
                unsigned int static_y_2 = 0;
                unsigned int valid_2 = static_valid_2;
                m_tile_2 = static_x_2;
                n_tile_2 = static_y_2;
                if (valid_2 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 6) {
        { // load_a_main
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            unsigned int _phase_k_done_1 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < ((1) ? 1 : grid_m * grid_n); _tile_iter_3++) {
                if (n_tile_3 >= 4) {
                    break;
                }
                int route_k_extent_2 = 6;
                int prefetch_k = 5;
                int prefetch_expert = tile_expert[n_tile_3];
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&A))), "r"((int)(0)), "r"((int)(m_tile_3 * (unsigned int)BLOCK_M)), "r"((int)(prefetch_k * 2)), "r"((int)(prefetch_expert)) : "memory");
                }
                int _min_0 = ((route_k_extent_2) < (5) ? (route_k_extent_2) : (5));
                #pragma unroll 1
                for (int route_k_4 = ((1) ? 0 : 5); route_k_4 < ((1) ? _min_0 : route_k_extent_2); route_k_4++) {
                    int iter_k_4 = route_k_4;
                    int expert_1 = tile_expert[n_tile_3];
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_a_addr + stage_2 * 32768, (&A), 0, m_tile_3 * (unsigned int)BLOCK_M, iter_k_4 * 2, expert_1, a_full_addr + (stage_2) * 8);
                        mbarrier_arrive_expect_tx(a_full_addr + (stage_2) * 8, 32768);
                    }
                    stage_2 += 1;
                    if (stage_2 == 5) { stage_2 = 0; _phase_k_done_1 ^= 1; }
                }
                int _min_1 = ((route_k_extent_2) < (5) ? (route_k_extent_2) : (5));
                #pragma unroll 1
                for (int route_k_5 = ((!1) ? 0 : 5); route_k_5 < ((0) ? _min_1 : route_k_extent_2); route_k_5++) {
                    int iter_k_5 = route_k_5;
                    int expert_2 = tile_expert[n_tile_3];
                    {
                        mbarrier_wait(k_done_addr + (stage_2) * 8, _phase_k_done_1);
                    }
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_a_addr + stage_2 * 32768, (&A), 0, m_tile_3 * (unsigned int)BLOCK_M, iter_k_5 * 2, expert_2, a_full_addr + (stage_2) * 8);
                        mbarrier_arrive_expect_tx(a_full_addr + (stage_2) * 8, 32768);
                    }
                    stage_2 += 1;
                    if (stage_2 == 5) { stage_2 = 0; _phase_k_done_1 ^= 1; }
                }
                unsigned int static_valid_3 = 0;
                unsigned int static_x_3 = 0;
                unsigned int static_y_3 = 0;
                unsigned int valid_3 = static_valid_3;
                m_tile_3 = static_x_3;
                n_tile_3 = static_y_3;
                if (valid_3 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 7) {
        { // load_sfa_main
            unsigned int stage_3 = 0;
            unsigned int work_stage_4 = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int _phase_sfa_smem_free = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < ((1) ? 1 : grid_m * grid_n); _tile_iter_4++) {
                if (n_tile_4 >= 4) {
                    break;
                }
                int route_k_extent_3 = 6;
                int _min_4 = ((route_k_extent_3) < (5) ? (route_k_extent_3) : (5));
                #pragma unroll 1
                for (int route_k_6 = ((1) ? 0 : 5); route_k_6 < ((1) ? _min_4 : route_k_extent_3); route_k_6++) {
                    int iter_k_6 = route_k_6;
                    int expert_3 = tile_expert[n_tile_4];
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_sfa_addr + stage_3 * 4096, (&SFA), 0, 0, iter_k_6 * 8, (unsigned int)(expert_3 * grid_m) + m_tile_4, sfa_full_addr + (stage_3) * 8);
                        mbarrier_arrive_expect_tx(sfa_full_addr + (stage_3) * 8, 4096);
                    }
                    stage_3 += 1;
                    if (stage_3 == 5) { stage_3 = 0; _phase_sfa_smem_free ^= 1; }
                }
                int _min_5 = ((route_k_extent_3) < (5) ? (route_k_extent_3) : (5));
                #pragma unroll 1
                for (int route_k_7 = ((!1) ? 0 : 5); route_k_7 < ((0) ? _min_5 : route_k_extent_3); route_k_7++) {
                    int iter_k_7 = route_k_7;
                    int expert_4 = tile_expert[n_tile_4];
                    {
                        mbarrier_wait(sfa_smem_free_addr + (stage_3) * 8, _phase_sfa_smem_free);
                    }
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_sfa_addr + stage_3 * 4096, (&SFA), 0, 0, iter_k_7 * 8, (unsigned int)(expert_4 * grid_m) + m_tile_4, sfa_full_addr + (stage_3) * 8);
                        mbarrier_arrive_expect_tx(sfa_full_addr + (stage_3) * 8, 4096);
                    }
                    stage_3 += 1;
                    if (stage_3 == 5) { stage_3 = 0; _phase_sfa_smem_free ^= 1; }
                }
                unsigned int static_valid_4 = 0;
                unsigned int static_x_4 = 0;
                unsigned int static_y_4 = 0;
                unsigned int valid_4 = static_valid_4;
                m_tile_4 = static_x_4;
                n_tile_4 = static_y_4;
                if (valid_4 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: copy_sfa ----
    if (warp == 8) {
        { // copy_sfa_main
            unsigned int stage_4 = 0;
            unsigned int work_stage_5 = 0;
            unsigned int m_tile_5 = blockIdx.x;
            unsigned int n_tile_5 = blockIdx.y;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_k_done_2 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < ((1) ? 1 : grid_m * grid_n); _tile_iter_5++) {
                if (n_tile_5 >= 4) {
                    break;
                }
                int route_k_extent_4 = 6;
                int _min_8 = ((route_k_extent_4) < (5) ? (route_k_extent_4) : (5));
                #pragma unroll 1
                for (int _route_k = ((1) ? 0 : 5); _route_k < ((1) ? _min_8 : route_k_extent_4); _route_k++) {
                    mbarrier_wait(sfa_full_addr + (stage_4) * 8, _phase_sfa_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_4 * 32)), "l"(_tcgen05_cp_desc_0)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 4))), "l"(_tcgen05_cp_desc_1)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 8))), "l"(_tcgen05_cp_desc_2)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 12))), "l"(_tcgen05_cp_desc_3)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 16))), "l"(_tcgen05_cp_desc_4)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 2560)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 20))), "l"(_tcgen05_cp_desc_5)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 3072)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 24))), "l"(_tcgen05_cp_desc_6)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 3584)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 28))), "l"(_tcgen05_cp_desc_7)
                                : "memory");
                        }
                    }
                    if (route_k_extent_4 > _route_k + 5) {
                        elect_commit2(tmem_sfa_full_addr + (stage_4) * 8, sfa_smem_free_addr + (stage_4) * 8);
                    } else {
                        elect_commit(tmem_sfa_full_addr + (stage_4) * 8);
                    }
                    stage_4 += 1;
                    if (stage_4 == 5) { stage_4 = 0; _phase_sfa_full ^= 1; _phase_k_done_2 ^= 1; }
                }
                int _min_9 = ((route_k_extent_4) < (5) ? (route_k_extent_4) : (5));
                #pragma unroll 1
                for (int _route_k_1 = ((!1) ? 0 : 5); _route_k_1 < ((0) ? _min_9 : route_k_extent_4); _route_k_1++) {
                    mbarrier_wait(sfa_full_addr + (stage_4) * 8, _phase_sfa_full);
                    {
                        mbarrier_wait(k_done_addr + (stage_4) * 8, _phase_k_done_2);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_4 * 32)), "l"(_tcgen05_cp_desc_8)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 4))), "l"(_tcgen05_cp_desc_9)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 8))), "l"(_tcgen05_cp_desc_10)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 12))), "l"(_tcgen05_cp_desc_11)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 16))), "l"(_tcgen05_cp_desc_12)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 2560)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 20))), "l"(_tcgen05_cp_desc_13)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 3072)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 24))), "l"(_tcgen05_cp_desc_14)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 3584)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 28))), "l"(_tcgen05_cp_desc_15)
                                : "memory");
                        }
                    }
                    if (route_k_extent_4 > _route_k_1 + 5) {
                        elect_commit2(tmem_sfa_full_addr + (stage_4) * 8, sfa_smem_free_addr + (stage_4) * 8);
                    } else {
                        elect_commit(tmem_sfa_full_addr + (stage_4) * 8);
                    }
                    stage_4 += 1;
                    if (stage_4 == 5) { stage_4 = 0; _phase_sfa_full ^= 1; _phase_k_done_2 ^= 1; }
                }
                unsigned int static_valid_5 = 0;
                unsigned int static_x_5 = 0;
                unsigned int static_y_5 = 0;
                unsigned int valid_5 = static_valid_5;
                m_tile_5 = static_x_5;
                n_tile_5 = static_y_5;
                if (valid_5 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: copy_sfb_cp ----
    if (warp == 9) {
        { // copy_sfb_cp_main
            unsigned int stage_5 = 0;
            unsigned int work_stage_6 = 0;
            unsigned int m_tile_6 = blockIdx.x;
            unsigned int n_tile_6 = blockIdx.y;
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_k_done_3 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_6 = 0; _tile_iter_6 < ((1) ? 1 : grid_m * grid_n); _tile_iter_6++) {
                if (n_tile_6 >= 4) {
                    break;
                }
                int route_k_extent_5 = 6;
                int _min_10 = ((route_k_extent_5) < (5) ? (route_k_extent_5) : (5));
                #pragma unroll 1
                for (int _route_k_2 = ((1) ? 0 : 5); _route_k_2 < ((1) ? _min_10 : route_k_extent_5); _route_k_2++) {
                    mbarrier_wait(sfb_full_addr + (stage_5) * 8, _phase_sfb_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + stage_5 * 32)), "l"(_tcgen05_cp_desc_0)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 4))), "l"(_tcgen05_cp_desc_1)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 8))), "l"(_tcgen05_cp_desc_2)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 12))), "l"(_tcgen05_cp_desc_3)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 16))), "l"(_tcgen05_cp_desc_4)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 2560)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 20))), "l"(_tcgen05_cp_desc_5)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 3072)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 24))), "l"(_tcgen05_cp_desc_6)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 3584)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 28))), "l"(_tcgen05_cp_desc_7)
                                : "memory");
                        }
                    }
                    if (route_k_extent_5 > _route_k_2 + 5) {
                        elect_commit2(tmem_sfb_full_addr + (stage_5) * 8, sfb_smem_free_addr + (stage_5) * 8);
                    } else {
                        elect_commit(tmem_sfb_full_addr + (stage_5) * 8);
                    }
                    stage_5 += 1;
                    if (stage_5 == 5) { stage_5 = 0; _phase_sfb_full ^= 1; _phase_k_done_3 ^= 1; }
                }
                int _min_11 = ((route_k_extent_5) < (5) ? (route_k_extent_5) : (5));
                #pragma unroll 1
                for (int _route_k_3 = ((!1) ? 0 : 5); _route_k_3 < ((0) ? _min_11 : route_k_extent_5); _route_k_3++) {
                    mbarrier_wait(sfb_full_addr + (stage_5) * 8, _phase_sfb_full);
                    {
                        mbarrier_wait(k_done_addr + (stage_5) * 8, _phase_k_done_3);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + stage_5 * 32)), "l"(_tcgen05_cp_desc_8)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 4))), "l"(_tcgen05_cp_desc_9)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 8))), "l"(_tcgen05_cp_desc_10)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 12))), "l"(_tcgen05_cp_desc_11)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_12 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 16))), "l"(_tcgen05_cp_desc_12)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_13 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 2560)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 20))), "l"(_tcgen05_cp_desc_13)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_14 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 3072)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 24))), "l"(_tcgen05_cp_desc_14)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_15 = ((((uint64_t)(smem_sfb_addr + stage_5 * 4096 + 3584)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_5 * 32 + 28))), "l"(_tcgen05_cp_desc_15)
                                : "memory");
                        }
                    }
                    if (route_k_extent_5 > _route_k_3 + 5) {
                        elect_commit2(tmem_sfb_full_addr + (stage_5) * 8, sfb_smem_free_addr + (stage_5) * 8);
                    } else {
                        elect_commit(tmem_sfb_full_addr + (stage_5) * 8);
                    }
                    stage_5 += 1;
                    if (stage_5 == 5) { stage_5 = 0; _phase_sfb_full ^= 1; _phase_k_done_3 ^= 1; }
                }
                unsigned int static_valid_6 = 0;
                unsigned int static_x_6 = 0;
                unsigned int static_y_6 = 0;
                unsigned int valid_6 = static_valid_6;
                m_tile_6 = static_x_6;
                n_tile_6 = static_y_6;
                if (valid_6 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 10) {
        { // mma_main
            unsigned int acc_stage_1 = 0;
            unsigned int work_stage_7 = 0;
            unsigned int a_token = 0;
            unsigned int b_token = 0;
            unsigned int sfa_token = 0;
            unsigned int sfb_token = 0;
            unsigned int m_tile_7 = blockIdx.x;
            unsigned int n_tile_7 = blockIdx.y;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_tmem_sfa_full = 0;
            unsigned int _phase_a_full_1 = 0;
            unsigned int _phase_b_full_1 = 0;
            unsigned int _phase_tmem_sfa_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_7 = 0; _tile_iter_7 < ((1) ? 1 : grid_m * grid_n); _tile_iter_7++) {
                if (n_tile_7 >= 4) {
                    break;
                }
                int route_k_extent_6 = 6;
                #pragma unroll 1
                for (int route_k_8 = ((1) ? 0 : 5); route_k_8 < ((1) ? 5 : 6); route_k_8++) {
                    unsigned int _mma_stage = ((1) ? route_k_8 : route_k_8 - 5);
                    int route_slot = ((0) ? route_k_8 / K_tiles : 0);
                    {
                        uint32_t _mbar_token_0 = mbarrier_try_wait(a_full_addr + (_mma_stage) * 8, 0);
                        a_token = _mbar_token_0;
                        uint32_t _mbar_token_1 = mbarrier_try_wait(b_full_addr + (_mma_stage) * 8, 0);
                        b_token = _mbar_token_1;
                        uint32_t _mbar_token_2 = mbarrier_try_wait(tmem_sfa_full_addr + (_mma_stage) * 8, 0);
                        sfa_token = _mbar_token_2;
                        uint32_t _mbar_token_3 = mbarrier_try_wait(tmem_sfb_full_addr + (_mma_stage) * 8, 0);
                        sfb_token = _mbar_token_3;
                        mbarrier_wait_token(a_full_addr + (_mma_stage) * 8, 0, a_token);
                        mbarrier_wait_token(b_full_addr + (_mma_stage) * 8, 0, b_token);
                        mbarrier_wait_token(tmem_sfa_full_addr + (_mma_stage) * 8, 0, sfa_token);
                        mbarrier_wait_token(tmem_sfb_full_addr + (_mma_stage) * 8, 0, sfb_token);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int sfa_base_col = 0;
                    int sfb_base_col = 0;
                    int _mma_a_lo_0 = make_warp_uniform((((smem_a_mma0_addr) >> 4) & 0x3FFF) + (_mma_stage) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_b_mma0_addr) >> 4) & 0x3FFF) + (_mma_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage * 32 + (unsigned int)sfa_base_col) + 0, (unsigned int)tmem_sfb + (_mma_stage * 32 + (unsigned int)sfb_base_col) + 0, ((((1) ? ((0) ? ((route_k_8 % K_tiles == 0) ? 1 : 0) : ((route_k_8 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_0 = 4;
                    int sfb_base_col_1 = 4;
                    int _mma_a_lo_1 = make_warp_uniform((((smem_a_mma1_addr) >> 4) & 0x3FFF) + (_mma_stage) * 2048);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_b_mma1_addr) >> 4) & 0x3FFF) + (_mma_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage * 32 + (unsigned int)sfa_base_col_0) + 0, (unsigned int)tmem_sfb + (_mma_stage * 32 + (unsigned int)sfb_base_col_1) + 0, ((((0) ? ((0) ? ((route_k_8 % K_tiles == 0) ? 1 : 0) : ((route_k_8 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_2 = 8;
                    int sfb_base_col_3 = 8;
                    int _mma_a_lo_2 = make_warp_uniform((((smem_a_mma2_addr) >> 4) & 0x3FFF) + (_mma_stage) * 2048);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_b_mma2_addr) >> 4) & 0x3FFF) + (_mma_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage * 32 + (unsigned int)sfa_base_col_2) + 0, (unsigned int)tmem_sfb + (_mma_stage * 32 + (unsigned int)sfb_base_col_3) + 0, ((((0) ? ((0) ? ((route_k_8 % K_tiles == 0) ? 1 : 0) : ((route_k_8 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_4 = 12;
                    int sfb_base_col_5 = 12;
                    int _mma_a_lo_3 = make_warp_uniform((((smem_a_mma3_addr) >> 4) & 0x3FFF) + (_mma_stage) * 2048);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_b_mma3_addr) >> 4) & 0x3FFF) + (_mma_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage * 32 + (unsigned int)sfa_base_col_4) + 0, (unsigned int)tmem_sfb + (_mma_stage * 32 + (unsigned int)sfb_base_col_5) + 0, ((((0) ? ((0) ? ((route_k_8 % K_tiles == 0) ? 1 : 0) : ((route_k_8 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_6 = 16;
                    int sfb_base_col_7 = 16;
                    int _mma_a_lo_4 = make_warp_uniform((((smem_a_mma4_addr) >> 4) & 0x3FFF) + (_mma_stage) * 2048);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_b_mma4_addr) >> 4) & 0x3FFF) + (_mma_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage * 32 + (unsigned int)sfa_base_col_6) + 0, (unsigned int)tmem_sfb + (_mma_stage * 32 + (unsigned int)sfb_base_col_7) + 0, ((((0) ? ((0) ? ((route_k_8 % K_tiles == 0) ? 1 : 0) : ((route_k_8 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_8 = 20;
                    int sfb_base_col_9 = 20;
                    int _mma_a_lo_5 = make_warp_uniform((((smem_a_mma5_addr) >> 4) & 0x3FFF) + (_mma_stage) * 2048);
                    int _mma_b_lo_5 = make_warp_uniform((((smem_b_mma5_addr) >> 4) & 0x3FFF) + (_mma_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage * 32 + (unsigned int)sfa_base_col_8) + 0, (unsigned int)tmem_sfb + (_mma_stage * 32 + (unsigned int)sfb_base_col_9) + 0, ((((0) ? ((0) ? ((route_k_8 % K_tiles == 0) ? 1 : 0) : ((route_k_8 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_10 = 24;
                    int sfb_base_col_11 = 24;
                    int _mma_a_lo_6 = make_warp_uniform((((smem_a_mma6_addr) >> 4) & 0x3FFF) + (_mma_stage) * 2048);
                    int _mma_b_lo_6 = make_warp_uniform((((smem_b_mma6_addr) >> 4) & 0x3FFF) + (_mma_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage * 32 + (unsigned int)sfa_base_col_10) + 0, (unsigned int)tmem_sfb + (_mma_stage * 32 + (unsigned int)sfb_base_col_11) + 0, ((((0) ? ((0) ? ((route_k_8 % K_tiles == 0) ? 1 : 0) : ((route_k_8 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_12 = 28;
                    int sfb_base_col_13 = 28;
                    int _mma_a_lo_7 = make_warp_uniform((((smem_a_mma7_addr) >> 4) & 0x3FFF) + (_mma_stage) * 2048);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_b_mma7_addr) >> 4) & 0x3FFF) + (_mma_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage * 32 + (unsigned int)sfa_base_col_12) + 0, (unsigned int)tmem_sfb + (_mma_stage * 32 + (unsigned int)sfb_base_col_13) + 0, ((((0) ? ((0) ? ((route_k_8 % K_tiles == 0) ? 1 : 0) : ((route_k_8 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    if (route_k_extent_6 > route_k_8 + 5) {
                        elect_commit(k_done_addr + (_mma_stage) * 8);
                    }
                    if (route_k_8 + 1 == route_k_extent_6) {
                        elect_commit(mma_full_addr + (acc_stage_1) * 8);
                    }
                }
                #pragma unroll 1
                for (int route_k_9 = ((0) ? 0 : 5); route_k_9 < ((0) ? 5 : 6); route_k_9++) {
                    unsigned int _mma_stage_1 = ((0) ? route_k_9 : route_k_9 - 5);
                    int route_slot_1 = ((0) ? route_k_9 / K_tiles : 0);
                    {
                        uint32_t _mbar_token_4 = mbarrier_try_wait(a_full_addr + (_mma_stage_1) * 8, 1);
                        a_token = _mbar_token_4;
                        uint32_t _mbar_token_5 = mbarrier_try_wait(b_full_addr + (_mma_stage_1) * 8, 1);
                        b_token = _mbar_token_5;
                        uint32_t _mbar_token_6 = mbarrier_try_wait(tmem_sfa_full_addr + (_mma_stage_1) * 8, 1);
                        sfa_token = _mbar_token_6;
                        uint32_t _mbar_token_7 = mbarrier_try_wait(tmem_sfb_full_addr + (_mma_stage_1) * 8, 1);
                        sfb_token = _mbar_token_7;
                        mbarrier_wait_token(a_full_addr + (_mma_stage_1) * 8, 1, a_token);
                        mbarrier_wait_token(b_full_addr + (_mma_stage_1) * 8, 1, b_token);
                        mbarrier_wait_token(tmem_sfa_full_addr + (_mma_stage_1) * 8, 1, sfa_token);
                        mbarrier_wait_token(tmem_sfb_full_addr + (_mma_stage_1) * 8, 1, sfb_token);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int sfa_base_col_1 = 0;
                    int sfb_base_col_2 = 0;
                    int _mma_a_lo_8 = make_warp_uniform((((smem_a_mma0_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 2048);
                    int _mma_b_lo_8 = make_warp_uniform((((smem_b_mma0_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_8) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_8) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot_1) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage_1 * 32 + (unsigned int)sfa_base_col_1) + 0, (unsigned int)tmem_sfb + (_mma_stage_1 * 32 + (unsigned int)sfb_base_col_2) + 0, ((((1) ? ((0) ? ((route_k_9 % K_tiles == 0) ? 1 : 0) : ((route_k_9 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_0_1 = 4;
                    int sfb_base_col_1_1 = 4;
                    int _mma_a_lo_9 = make_warp_uniform((((smem_a_mma1_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 2048);
                    int _mma_b_lo_9 = make_warp_uniform((((smem_b_mma1_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_9) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_9) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot_1) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage_1 * 32 + (unsigned int)sfa_base_col_0_1) + 0, (unsigned int)tmem_sfb + (_mma_stage_1 * 32 + (unsigned int)sfb_base_col_1_1) + 0, ((((0) ? ((0) ? ((route_k_9 % K_tiles == 0) ? 1 : 0) : ((route_k_9 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_2_1 = 8;
                    int sfb_base_col_3_1 = 8;
                    int _mma_a_lo_10 = make_warp_uniform((((smem_a_mma2_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 2048);
                    int _mma_b_lo_10 = make_warp_uniform((((smem_b_mma2_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_10) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_10) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot_1) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage_1 * 32 + (unsigned int)sfa_base_col_2_1) + 0, (unsigned int)tmem_sfb + (_mma_stage_1 * 32 + (unsigned int)sfb_base_col_3_1) + 0, ((((0) ? ((0) ? ((route_k_9 % K_tiles == 0) ? 1 : 0) : ((route_k_9 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_4_1 = 12;
                    int sfb_base_col_5_1 = 12;
                    int _mma_a_lo_11 = make_warp_uniform((((smem_a_mma3_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 2048);
                    int _mma_b_lo_11 = make_warp_uniform((((smem_b_mma3_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_11) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_11) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot_1) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage_1 * 32 + (unsigned int)sfa_base_col_4_1) + 0, (unsigned int)tmem_sfb + (_mma_stage_1 * 32 + (unsigned int)sfb_base_col_5_1) + 0, ((((0) ? ((0) ? ((route_k_9 % K_tiles == 0) ? 1 : 0) : ((route_k_9 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_6_1 = 16;
                    int sfb_base_col_7_1 = 16;
                    int _mma_a_lo_12 = make_warp_uniform((((smem_a_mma4_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 2048);
                    int _mma_b_lo_12 = make_warp_uniform((((smem_b_mma4_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_12) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_12) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot_1) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage_1 * 32 + (unsigned int)sfa_base_col_6_1) + 0, (unsigned int)tmem_sfb + (_mma_stage_1 * 32 + (unsigned int)sfb_base_col_7_1) + 0, ((((0) ? ((0) ? ((route_k_9 % K_tiles == 0) ? 1 : 0) : ((route_k_9 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_8_1 = 20;
                    int sfb_base_col_9_1 = 20;
                    int _mma_a_lo_13 = make_warp_uniform((((smem_a_mma5_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 2048);
                    int _mma_b_lo_13 = make_warp_uniform((((smem_b_mma5_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_13) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_13) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot_1) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage_1 * 32 + (unsigned int)sfa_base_col_8_1) + 0, (unsigned int)tmem_sfb + (_mma_stage_1 * 32 + (unsigned int)sfb_base_col_9_1) + 0, ((((0) ? ((0) ? ((route_k_9 % K_tiles == 0) ? 1 : 0) : ((route_k_9 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_10_1 = 24;
                    int sfb_base_col_11_1 = 24;
                    int _mma_a_lo_14 = make_warp_uniform((((smem_a_mma6_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 2048);
                    int _mma_b_lo_14 = make_warp_uniform((((smem_b_mma6_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_14) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_14) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot_1) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage_1 * 32 + (unsigned int)sfa_base_col_10_1) + 0, (unsigned int)tmem_sfb + (_mma_stage_1 * 32 + (unsigned int)sfb_base_col_11_1) + 0, ((((0) ? ((0) ? ((route_k_9 % K_tiles == 0) ? 1 : 0) : ((route_k_9 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_12_1 = 28;
                    int sfb_base_col_13_1 = 28;
                    int _mma_a_lo_15 = make_warp_uniform((((smem_a_mma7_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 2048);
                    int _mma_b_lo_15 = make_warp_uniform((((smem_b_mma7_addr) >> 4) & 0x3FFF) + (_mma_stage_1) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_15) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_15) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot_1) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (_mma_stage_1 * 32 + (unsigned int)sfa_base_col_12_1) + 0, (unsigned int)tmem_sfb + (_mma_stage_1 * 32 + (unsigned int)sfb_base_col_13_1) + 0, ((((0) ? ((0) ? ((route_k_9 % K_tiles == 0) ? 1 : 0) : ((route_k_9 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    if (route_k_extent_6 > route_k_9 + 5) {
                        elect_commit(k_done_addr + (_mma_stage_1) * 8);
                    }
                    if (route_k_9 + 1 == route_k_extent_6) {
                        elect_commit(mma_full_addr + (acc_stage_1) * 8);
                    }
                }
                unsigned int static_valid_7 = 0;
                unsigned int static_x_7 = 0;
                unsigned int static_y_7 = 0;
                unsigned int valid_7 = static_valid_7;
                m_tile_7 = static_x_7;
                n_tile_7 = static_y_7;
                if (valid_7 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 11) {
        { // padding_main
            unsigned int work_stage_8 = 0;
            unsigned int m_tile_8 = blockIdx.x;
            unsigned int n_tile_8 = blockIdx.y;
            #pragma unroll 1
            for (unsigned int _tile_iter_8 = 0; _tile_iter_8 < ((1) ? 1 : grid_m * grid_n); _tile_iter_8++) {
                if (n_tile_8 >= 4) {
                    break;
                }
                unsigned int static_valid_8 = 0;
                unsigned int static_x_8 = 0;
                unsigned int static_y_8 = 0;
                unsigned int valid_8 = static_valid_8;
                m_tile_8 = static_x_8;
                n_tile_8 = static_y_8;
                if (valid_8 == 0) {
                    break;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
