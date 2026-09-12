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
#define TMEM_NCOLS 208
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 16
#define TMEM_SFB_OFFSET 144
#define NUM_K_PIPE_STAGES 4
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 132096
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
#define SMEM_SMEM_B_MMA0_OFF 132096
#define SMEM_SMEM_B_MMA0_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA0_STRIDE 2048
#define SMEM_SMEM_B_MMA1_OFF 132128
#define SMEM_SMEM_B_MMA1_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA1_STRIDE 2048
#define SMEM_SMEM_B_MMA2_OFF 132160
#define SMEM_SMEM_B_MMA2_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA2_STRIDE 2048
#define SMEM_SMEM_B_MMA3_OFF 132192
#define SMEM_SMEM_B_MMA3_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA3_STRIDE 2048
#define SMEM_SMEM_B_MMA4_OFF 133120
#define SMEM_SMEM_B_MMA4_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA4_STRIDE 2048
#define SMEM_SMEM_B_MMA5_OFF 133152
#define SMEM_SMEM_B_MMA5_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA5_STRIDE 2048
#define SMEM_SMEM_B_MMA6_OFF 133184
#define SMEM_SMEM_B_MMA6_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA6_STRIDE 2048
#define SMEM_SMEM_B_MMA7_OFF 133216
#define SMEM_SMEM_B_MMA7_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA7_STRIDE 2048
#define SMEM_SMEM_A_MMA0_NEXT_OFF 1024
#define SMEM_SMEM_A_MMA0_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA0_NEXT_STRIDE 32768
#define SMEM_SMEM_B_MMA0_NEXT_OFF 132096
#define SMEM_SMEM_B_MMA0_NEXT_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA0_NEXT_STRIDE 2048
#define SMEM_SMEM_A_MMA2_NEXT_OFF 1024
#define SMEM_SMEM_A_MMA2_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA2_NEXT_STRIDE 32768
#define SMEM_SMEM_B_MMA2_NEXT_OFF 132096
#define SMEM_SMEM_B_MMA2_NEXT_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA2_NEXT_STRIDE 2048
#define SMEM_SMEM_A_MMA5_NEXT_OFF 1024
#define SMEM_SMEM_A_MMA5_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA5_NEXT_STRIDE 32768
#define SMEM_SMEM_B_MMA5_NEXT_OFF 132096
#define SMEM_SMEM_B_MMA5_NEXT_STAGE_BYTES 256
#define SMEM_SMEM_B_MMA5_NEXT_STRIDE 2048
#define SMEM_SMEM_SFA_OFF 140288
#define SMEM_SMEM_SFA_STAGE_BYTES 4096
#define SMEM_SMEM_SFA_STRIDE 4096
#define SMEM_SMEM_SFB_OFF 156672
#define SMEM_SMEM_SFB_STAGE_BYTES 256
#define SMEM_SMEM_SFB_STRIDE 256
#define SMEM_EPI_STAGING_OFF 157696
#define SMEM_EPI_STAGING_STAGE_BYTES 2048
#define SMEM_EPI_STAGING_STRIDE 2048
#define SMEM_EPI_STAGING_U64_OFF 157696
#define SMEM_EPI_STAGING_U64_STAGE_BYTES 2048
#define SMEM_EPI_STAGING_U64_STRIDE 2048
#define SMEM_WORK_RESPONSE_OFF 159744
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 159872
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


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
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

__global__ __launch_bounds__(512, 1) void
kernel_cake_warp_decode_b9a1d88b8a94f4f37426(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C_tma, __nv_bfloat16* __restrict__ C, float* __restrict__ scale_c, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, int* __restrict__ work_counter, int M, int K, int grid_m, int grid_n, int K_tiles)
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
    #define b_full_addr (mbar_base + 32)
    #define sfa_full_addr (mbar_base + 64)
    #define sfb_full_addr (mbar_base + 96)
    #define sfa_smem_free_addr (mbar_base + 128)
    #define sfb_smem_free_addr (mbar_base + 160)
    #define tmem_sfa_full_addr (mbar_base + 192)
    #define tmem_sfb_full_addr (mbar_base + 224)
    #define a_empty_addr (mbar_base + 256)
    #define b_empty_addr (mbar_base + 288)
    #define tmem_sfa_empty_addr (mbar_base + 320)
    #define tmem_sfb_empty_addr (mbar_base + 352)
    #define mma_full_addr (mbar_base + 384)
    #define mma_free_addr (mbar_base + 400)
    #define work_full_addr (mbar_base + 416)
    #define work_empty_addr (mbar_base + 440)
    #define throttle_full_addr (mbar_base + 464)
    #define throttle_empty_addr (mbar_base + 488)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 512);

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_b_addr = smem + 132096;
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
    uint8_t* smem_b_mma0 = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_b_mma0_addr = smem + 132096;
    uint8_t* smem_b_mma1 = reinterpret_cast<uint8_t*>(smem_raw + 132128);
    const int smem_b_mma1_addr = smem + 132128;
    uint8_t* smem_b_mma2 = reinterpret_cast<uint8_t*>(smem_raw + 132160);
    const int smem_b_mma2_addr = smem + 132160;
    uint8_t* smem_b_mma3 = reinterpret_cast<uint8_t*>(smem_raw + 132192);
    const int smem_b_mma3_addr = smem + 132192;
    uint8_t* smem_b_mma4 = reinterpret_cast<uint8_t*>(smem_raw + 133120);
    const int smem_b_mma4_addr = smem + 133120;
    uint8_t* smem_b_mma5 = reinterpret_cast<uint8_t*>(smem_raw + 133152);
    const int smem_b_mma5_addr = smem + 133152;
    uint8_t* smem_b_mma6 = reinterpret_cast<uint8_t*>(smem_raw + 133184);
    const int smem_b_mma6_addr = smem + 133184;
    uint8_t* smem_b_mma7 = reinterpret_cast<uint8_t*>(smem_raw + 133216);
    const int smem_b_mma7_addr = smem + 133216;
    uint8_t* smem_a_mma0_next = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma0_next_addr = smem + 1024;
    uint8_t* smem_b_mma0_next = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_b_mma0_next_addr = smem + 132096;
    uint8_t* smem_a_mma2_next = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma2_next_addr = smem + 1024;
    uint8_t* smem_b_mma2_next = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_b_mma2_next_addr = smem + 132096;
    uint8_t* smem_a_mma5_next = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma5_next_addr = smem + 1024;
    uint8_t* smem_b_mma5_next = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_b_mma5_next_addr = smem + 132096;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 140288);
    const int smem_sfa_addr = smem + 140288;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 156672);
    const int smem_sfb_addr = smem + 156672;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 157696);
    const int epi_staging_addr = smem + 157696;
    unsigned long long* epi_staging_u64 = reinterpret_cast<unsigned long long*>(smem_raw + 157696);
    const int epi_staging_u64_addr = smem + 157696;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 159744);
    const int work_response_addr = smem + 159744;

    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 64 barriers)
    // Mbarriers at smem_raw[0..512)

    if (warp == 10) {
        // --- pipeline 'k_pipe' ---
        // a_full: 4 barriers, init_count=1
        // a_empty: 4 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 0 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 256 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 8) {
        // --- pipeline 'k_pipe' ---
        // b_full: 4 barriers, init_count=1
        // b_empty: 4 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 32 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 288 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 11) {
        // --- pipeline 'k_pipe' ---
        // sfa_full: 4 barriers, init_count=1
        // sfa_smem_free: 4 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 64 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 128 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 9) {
        // --- pipeline 'k_pipe' ---
        // sfb_full: 4 barriers, init_count=1
        // sfb_smem_free: 4 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 96 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 160 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 12) {
        // --- pipeline 'k_pipe' ---
        // tmem_sfa_full: 4 barriers, init_count=1
        // tmem_sfa_empty: 4 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 192 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 320 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 4) {
        // --- pipeline 'k_pipe' ---
        // tmem_sfb_full: 4 barriers, init_count=1
        // tmem_sfb_empty: 4 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 224 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 4; _bar += 32) {
            mbarrier_init(smem + 352 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 1) {
        // --- pipeline 'mma_pipe' ---
        // mma_full: 2 barriers, init_count=1
        // mma_free: 2 barriers, init_count=4
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 2; _bar += 32) {
            mbarrier_init(smem + 384 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 2; _bar += 32) {
            mbarrier_init(smem + 400 + _bar * 8, 4);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 14) {
        // --- pipeline 'work_pipe' ---
        // work_full: 3 barriers, init_count=1
        // work_empty: 3 barriers, init_count=512
        // --- pipeline 'throttle_pipe' ---
        // throttle_full: 3 barriers, init_count=32
        // throttle_empty: 3 barriers, init_count=32
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 3; _bar += 32) {
            mbarrier_init(smem + 416 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 3; _bar += 32) {
            mbarrier_init(smem + 440 + _bar * 8, 512);
        }
        for (int _bar = lane; _bar < 6; _bar += 32) {
            mbarrier_init(smem + 464 + _bar * 8, 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (256 columns, 208 used)
    if (warp == 0) {
        int _tmem_hold = smem + 512;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sfa = taddr + 16;
    const int tmem_sfb = taddr + 144;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
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
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m * grid_n; _tile_iter++) {
                if (m_tile >= (unsigned int)grid_m || n_tile >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int token_base = n_tile * (unsigned int)BLOCK_N;
                int off_m = m_tile * (unsigned int)BLOCK_M;
                int expert = tile_expert[n_tile];
                float output_scale = scale_c[expert];
                int _mn_limit = tile_mn_limit[n_tile];
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
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
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
                    if (wide_feature < 64) {
                        epi_staging_u64[(token * 64 + wide_feature) / 4] = wide_word;
                    } else {
                        epi_staging_u64[(512 + token * 64 + wide_feature - 64) / 4] = wide_word;
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
                    if (wide_feature < 64) {
                        epi_staging_u64[(token_0 * 64 + wide_feature) / 4] = wide_word;
                    } else {
                        epi_staging_u64[(512 + token_0 * 64 + wide_feature - 64) / 4] = wide_word;
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows = (8 - _mn_limit % 8) % 8;
                        int local_token = padding_rows;
                        tma_store_4d((&C_tma), off_m, local_token, 1073741824, token_base - padding_rows + 1073741824, epi_staging_addr);
                        tma_store_4d((&C_tma), off_m + 64, local_token, 1073741824, token_base - padding_rows + 1073741824, epi_staging_addr + 1024);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(mma_free_addr + (acc_stage) * 8);
                }
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage) * 8, _phase_work_full, 10000000);
                }
                unsigned int valid = 0;
                unsigned int next_x = 0;
                unsigned int next_y = 0;
                int response_index = work_stage * 4;
                valid = work_response[response_index];
                next_x = work_response[response_index + 1];
                next_y = work_response[response_index + 2];
                mbarrier_arrive(work_empty_addr + (work_stage) * 8);
                work_stage += 1;
                if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                unsigned int valid_0 = valid;
                m_tile = next_x;
                n_tile = next_y;
                if (valid_0 == 0) {
                    break;
                }
            }
        }
    // ---- Role: copy_sfb ----
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // copy_sfb_main
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            const int lane_0 = lane;
            unsigned int word[1];
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_tmem_sfb_empty = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m * grid_n; _tile_iter_1++) {
                if (m_tile_1 >= (unsigned int)grid_m || n_tile_1 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int route_k_extent = K_tiles;
                #pragma unroll 1
                for (int _route_k = 0; _route_k < route_k_extent; _route_k++) {
                    mbarrier_wait(sfb_full_addr + (stage) * 8, _phase_sfb_full);
                    mbarrier_wait(tmem_sfb_empty_addr + (stage) * 8, _phase_tmem_sfb_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    const int q = 0;
                    for (int reg = 0; reg < 1; reg++) {
                        word[reg] = 0;
                        if (lane_0 + reg * 32 < 8) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg])) : "r"(smem_sfb_addr + stage * 256 + (unsigned int)((reg * 4 + lane_0 / 8) * 256) + (unsigned int)(q * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 16 + 128 + stage * 16), "r"(word[0]));
                    const int q_0 = 1;
                    for (int reg_1 = 0; reg_1 < 1; reg_1++) {
                        word[reg_1] = 0;
                        if (lane_0 + reg_1 * 32 < 8) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_1])) : "r"(smem_sfb_addr + stage * 256 + (unsigned int)((reg_1 * 4 + lane_0 / 8) * 256) + (unsigned int)(q_0 * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 16 + 128 + stage * 16 + 2), "r"(word[0]));
                    const int q_1 = 2;
                    for (int reg_2 = 0; reg_2 < 1; reg_2++) {
                        word[reg_2] = 0;
                        if (lane_0 + reg_2 * 32 < 8) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_2])) : "r"(smem_sfb_addr + stage * 256 + (unsigned int)((reg_2 * 4 + lane_0 / 8) * 256) + (unsigned int)(q_1 * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 16 + 128 + stage * 16 + 4), "r"(word[0]));
                    const int q_2 = 3;
                    for (int reg_3 = 0; reg_3 < 1; reg_3++) {
                        word[reg_3] = 0;
                        if (lane_0 + reg_3 * 32 < 8) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_3])) : "r"(smem_sfb_addr + stage * 256 + (unsigned int)((reg_3 * 4 + lane_0 / 8) * 256) + (unsigned int)(q_2 * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 16 + 128 + stage * 16 + 6), "r"(word[0]));
                    const int q_3 = 4;
                    for (int reg_4 = 0; reg_4 < 1; reg_4++) {
                        word[reg_4] = 0;
                        if (lane_0 + reg_4 * 32 < 8) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_4])) : "r"(smem_sfb_addr + stage * 256 + (unsigned int)((reg_4 * 4 + lane_0 / 8) * 256) + (unsigned int)(q_3 * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 16 + 128 + stage * 16 + 8), "r"(word[0]));
                    const int q_4 = 5;
                    for (int reg_5 = 0; reg_5 < 1; reg_5++) {
                        word[reg_5] = 0;
                        if (lane_0 + reg_5 * 32 < 8) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_5])) : "r"(smem_sfb_addr + stage * 256 + (unsigned int)((reg_5 * 4 + lane_0 / 8) * 256) + (unsigned int)(q_4 * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 16 + 128 + stage * 16 + 10), "r"(word[0]));
                    const int q_5 = 6;
                    for (int reg_6 = 0; reg_6 < 1; reg_6++) {
                        word[reg_6] = 0;
                        if (lane_0 + reg_6 * 32 < 8) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_6])) : "r"(smem_sfb_addr + stage * 256 + (unsigned int)((reg_6 * 4 + lane_0 / 8) * 256) + (unsigned int)(q_5 * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 16 + 128 + stage * 16 + 12), "r"(word[0]));
                    const int q_6 = 7;
                    for (int reg_7 = 0; reg_7 < 1; reg_7++) {
                        word[reg_7] = 0;
                        if (lane_0 + reg_7 * 32 < 8) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_7])) : "r"(smem_sfb_addr + stage * 256 + (unsigned int)((reg_7 * 4 + lane_0 / 8) * 256) + (unsigned int)(q_6 * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 16 + 128 + stage * 16 + 14), "r"(word[0]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            mbarrier_arrive(tmem_sfb_full_addr + (stage) * 8);
                            mbarrier_arrive(sfb_smem_free_addr + (stage) * 8);
                        }
                    }
                    stage += 1;
                    if (stage == 4) { stage = 0; _phase_sfb_full ^= 1; _phase_tmem_sfb_empty ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_1) * 8, _phase_work_full_1, 10000000);
                }
                unsigned int valid_1 = 0;
                unsigned int next_x_1 = 0;
                unsigned int next_y_1 = 0;
                int response_index_1 = work_stage_1 * 4;
                valid_1 = work_response[response_index_1];
                next_x_1 = work_response[response_index_1 + 1];
                next_y_1 = work_response[response_index_1 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
                work_stage_1 += 1;
                if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                unsigned int valid_0_1 = valid_1;
                m_tile_1 = next_x_1;
                n_tile_1 = next_y_1;
                if (valid_0_1 == 0) {
                    break;
                }
            }
        }
    // ---- Role: load_b ----
    } else if (warp == 8) {
        { // load_b_main
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_b_empty = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m * grid_n; _tile_iter_2++) {
                if (m_tile_2 >= (unsigned int)grid_m || n_tile_2 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int route_k_extent_1 = K_tiles;
                #pragma unroll 1
                for (int route_k = 0; route_k < route_k_extent_1; route_k++) {
                    int iter_k = route_k;
                    int route_tile = n_tile_2;
                    mbarrier_wait(b_empty_addr + (stage_1) * 8, _phase_b_empty);
                    if (elect_sync()) {
                        tma_3d_gmem2smem(smem_b_addr + stage_1 * 2048, (&B), iter_k * 512, 0, route_tile, b_full_addr + (stage_1) * 8);
                        tma_3d_gmem2smem(smem_b_addr + stage_1 * 2048 + 1024, (&B), iter_k * 512 + 256, 0, route_tile, b_full_addr + (stage_1) * 8);
                        mbarrier_arrive_expect_tx(b_full_addr + (stage_1) * 8, 2048);
                    }
                    stage_1 += 1;
                    if (stage_1 == 4) { stage_1 = 0; _phase_b_empty ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_2) * 8, _phase_work_full_2, 10000000);
                }
                unsigned int valid_2 = 0;
                unsigned int next_x_2 = 0;
                unsigned int next_y_2 = 0;
                int response_index_2 = work_stage_2 * 4;
                valid_2 = work_response[response_index_2];
                next_x_2 = work_response[response_index_2 + 1];
                next_y_2 = work_response[response_index_2 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
                work_stage_2 += 1;
                if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                unsigned int valid_0_2 = valid_2;
                m_tile_2 = next_x_2;
                n_tile_2 = next_y_2;
                if (valid_0_2 == 0) {
                    break;
                }
            }
        }
    // ---- Role: load_sfb ----
    } else if (warp == 9) {
        { // load_sfb_main
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_sfb_smem_free = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m * grid_n; _tile_iter_3++) {
                if (m_tile_3 >= (unsigned int)grid_m || n_tile_3 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int route_k_extent_2 = K_tiles;
                #pragma unroll 1
                for (int route_k_1 = 0; route_k_1 < route_k_extent_2; route_k_1++) {
                    int iter_k_1 = route_k_1;
                    int route_tile_1 = n_tile_3;
                    mbarrier_wait(sfb_smem_free_addr + (stage_2) * 8, _phase_sfb_smem_free);
                    if (elect_sync()) {
                        tma_3d_gmem2smem(smem_sfb_addr + stage_2 * 256, (&SFB), 0, iter_k_1 * 8, route_tile_1, sfb_full_addr + (stage_2) * 8);
                        mbarrier_arrive_expect_tx(sfb_full_addr + (stage_2) * 8, 256);
                    }
                    stage_2 += 1;
                    if (stage_2 == 4) { stage_2 = 0; _phase_sfb_smem_free ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_3) * 8, _phase_work_full_3, 10000000);
                }
                unsigned int valid_3 = 0;
                unsigned int next_x_3 = 0;
                unsigned int next_y_3 = 0;
                int response_index_3 = work_stage_3 * 4;
                valid_3 = work_response[response_index_3];
                next_x_3 = work_response[response_index_3 + 1];
                next_y_3 = work_response[response_index_3 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
                work_stage_3 += 1;
                if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                unsigned int valid_0_3 = valid_3;
                m_tile_3 = next_x_3;
                n_tile_3 = next_y_3;
                if (valid_0_3 == 0) {
                    break;
                }
            }
        }
    // ---- Role: load_a ----
    } else if (warp == 10) {
        { // load_a_main
            unsigned int stage_3 = 0;
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_a_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < grid_m * grid_n; _tile_iter_4++) {
                if (m_tile_4 >= (unsigned int)grid_m || n_tile_4 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                throttle_stage += 1;
                if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                int route_k_extent_3 = K_tiles;
                #pragma unroll 1
                for (int route_k_2 = 0; route_k_2 < route_k_extent_3; route_k_2++) {
                    int iter_k_2 = route_k_2;
                    int expert_1 = tile_expert[n_tile_4];
                    mbarrier_wait(a_empty_addr + (stage_3) * 8, _phase_a_empty);
                    if (elect_sync()) {
                        tma_3d_gmem2smem(smem_a_addr + stage_3 * 32768, (&A), iter_k_2 * 512, m_tile_4 * (unsigned int)BLOCK_M, expert_1, a_full_addr + (stage_3) * 8);
                        tma_3d_gmem2smem(smem_a_addr + stage_3 * 32768 + 16384, (&A), iter_k_2 * 512 + 256, m_tile_4 * (unsigned int)BLOCK_M, expert_1, a_full_addr + (stage_3) * 8);
                        mbarrier_arrive_expect_tx(a_full_addr + (stage_3) * 8, 32768);
                    }
                    stage_3 += 1;
                    if (stage_3 == 4) { stage_3 = 0; _phase_a_empty ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_4) * 8, _phase_work_full_4, 10000000);
                }
                unsigned int valid_4 = 0;
                unsigned int next_x_4 = 0;
                unsigned int next_y_4 = 0;
                int response_index_4 = work_stage_4 * 4;
                valid_4 = work_response[response_index_4];
                next_x_4 = work_response[response_index_4 + 1];
                next_y_4 = work_response[response_index_4 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
                work_stage_4 += 1;
                if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
                unsigned int valid_0_4 = valid_4;
                m_tile_4 = next_x_4;
                n_tile_4 = next_y_4;
                if (valid_0_4 == 0) {
                    break;
                }
            }
        }
    // ---- Role: load_sfa ----
    } else if (warp == 11) {
        { // load_sfa_main
            unsigned int stage_4 = 0;
            unsigned int work_stage_5 = 0;
            unsigned int m_tile_5 = blockIdx.x;
            unsigned int n_tile_5 = blockIdx.y;
            unsigned int _phase_sfa_smem_free = 1;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < grid_m * grid_n; _tile_iter_5++) {
                if (m_tile_5 >= (unsigned int)grid_m || n_tile_5 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int route_k_extent_4 = K_tiles;
                #pragma unroll 1
                for (int route_k_3 = 0; route_k_3 < route_k_extent_4; route_k_3++) {
                    int iter_k_3 = route_k_3;
                    int expert_2 = tile_expert[n_tile_5];
                    mbarrier_wait(sfa_smem_free_addr + (stage_4) * 8, _phase_sfa_smem_free);
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_sfa_addr + stage_4 * 4096, (&SFA), 0, 0, iter_k_3 * 8, (unsigned int)(expert_2 * grid_m) + m_tile_5, sfa_full_addr + (stage_4) * 8);
                        mbarrier_arrive_expect_tx(sfa_full_addr + (stage_4) * 8, 4096);
                    }
                    stage_4 += 1;
                    if (stage_4 == 4) { stage_4 = 0; _phase_sfa_smem_free ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_5) * 8, _phase_work_full_5, 10000000);
                }
                unsigned int valid_5 = 0;
                unsigned int next_x_5 = 0;
                unsigned int next_y_5 = 0;
                int response_index_5 = work_stage_5 * 4;
                valid_5 = work_response[response_index_5];
                next_x_5 = work_response[response_index_5 + 1];
                next_y_5 = work_response[response_index_5 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
                work_stage_5 += 1;
                if (work_stage_5 == 3) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                unsigned int valid_0_5 = valid_5;
                m_tile_5 = next_x_5;
                n_tile_5 = next_y_5;
                if (valid_0_5 == 0) {
                    break;
                }
            }
        }
    // ---- Role: copy_sfa ----
    } else if (warp == 12) {
        { // copy_sfa_main
            unsigned int stage_5 = 0;
            unsigned int work_stage_6 = 0;
            unsigned int m_tile_6 = blockIdx.x;
            unsigned int n_tile_6 = blockIdx.y;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_tmem_sfa_empty = 1;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_6 = 0; _tile_iter_6 < grid_m * grid_n; _tile_iter_6++) {
                if (m_tile_6 >= (unsigned int)grid_m || n_tile_6 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int route_k_extent_5 = K_tiles;
                #pragma unroll 1
                for (int _route_k_1 = 0; _route_k_1 < route_k_extent_5; _route_k_1++) {
                    mbarrier_wait(sfa_full_addr + (stage_5) * 8, _phase_sfa_full);
                    mbarrier_wait(tmem_sfa_empty_addr + (stage_5) * 8, _phase_tmem_sfa_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_5 * 32)), "l"(_tcgen05_cp_desc_0)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 4))), "l"(_tcgen05_cp_desc_1)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 8))), "l"(_tcgen05_cp_desc_2)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 12))), "l"(_tcgen05_cp_desc_3)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 16))), "l"(_tcgen05_cp_desc_4)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 2560)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 20))), "l"(_tcgen05_cp_desc_5)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 3072)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 24))), "l"(_tcgen05_cp_desc_6)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 3584)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 28))), "l"(_tcgen05_cp_desc_7)
                                : "memory");
                        }
                    }
                    elect_commit2(tmem_sfa_full_addr + (stage_5) * 8, sfa_smem_free_addr + (stage_5) * 8);
                    stage_5 += 1;
                    if (stage_5 == 4) { stage_5 = 0; _phase_sfa_full ^= 1; _phase_tmem_sfa_empty ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_6) * 8, _phase_work_full_6, 10000000);
                }
                unsigned int valid_6 = 0;
                unsigned int next_x_6 = 0;
                unsigned int next_y_6 = 0;
                int response_index_6 = work_stage_6 * 4;
                valid_6 = work_response[response_index_6];
                next_x_6 = work_response[response_index_6 + 1];
                next_y_6 = work_response[response_index_6 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                work_stage_6 += 1;
                if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
                unsigned int valid_0_6 = valid_6;
                m_tile_6 = next_x_6;
                n_tile_6 = next_y_6;
                if (valid_0_6 == 0) {
                    break;
                }
            }
        }
    // ---- Role: mma ----
    } else if (warp == 13) {
        { // mma_main
            unsigned int k_stage = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int work_stage_7 = 0;
            unsigned int k_phase = 0;
            unsigned int a_token = 0;
            unsigned int b_token = 0;
            unsigned int sfa_token = 0;
            unsigned int sfb_token = 0;
            unsigned int m_tile_7 = blockIdx.x;
            unsigned int n_tile_7 = blockIdx.y;
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_tmem_sfa_full = 0;
            unsigned int _phase_tmem_sfb_full = 0;
            unsigned int _phase_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_7 = 0; _tile_iter_7 < grid_m * grid_n; _tile_iter_7++) {
                if (m_tile_7 >= (unsigned int)grid_m || n_tile_7 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                int route_k_extent_6 = K_tiles;
                #pragma unroll 1
                for (int route_k_4 = 0; route_k_4 < route_k_extent_6; route_k_4++) {
                    int route_slot = ((0) ? route_k_4 / K_tiles : 0);
                    {
                        uint32_t _mbar_token_0 = mbarrier_try_wait(a_full_addr + (k_stage) * 8, k_phase);
                        a_token = _mbar_token_0;
                        uint32_t _mbar_token_1 = mbarrier_try_wait(b_full_addr + (k_stage) * 8, k_phase);
                        b_token = _mbar_token_1;
                        uint32_t _mbar_token_2 = mbarrier_try_wait(tmem_sfa_full_addr + (k_stage) * 8, k_phase);
                        sfa_token = _mbar_token_2;
                        uint32_t _mbar_token_3 = mbarrier_try_wait(tmem_sfb_full_addr + (k_stage) * 8, k_phase);
                        sfb_token = _mbar_token_3;
                        mbarrier_wait_token(a_full_addr + (k_stage) * 8, k_phase, a_token);
                        mbarrier_wait_token(b_full_addr + (k_stage) * 8, k_phase, b_token);
                        mbarrier_wait_token(tmem_sfa_full_addr + (k_stage) * 8, k_phase, sfa_token);
                        mbarrier_wait_token(tmem_sfb_full_addr + (k_stage) * 8, k_phase, sfb_token);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int sfa_base_col = 0;
                    int sfb_base_col = 0;
                    int _mma_a_lo_0 = make_warp_uniform((((smem_a_mma0_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_b_mma0_addr) >> 4) & 0x3FFF) + (k_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (k_stage * 32 + (unsigned int)sfa_base_col) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col) + 0, ((((1) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_0 = 4;
                    int sfb_base_col_1 = 2;
                    int _mma_a_lo_1 = make_warp_uniform((((smem_a_mma1_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_b_mma1_addr) >> 4) & 0x3FFF) + (k_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (k_stage * 32 + (unsigned int)sfa_base_col_0) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_1) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_2 = 8;
                    int sfb_base_col_3 = 4;
                    int _mma_a_lo_2 = make_warp_uniform((((smem_a_mma2_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_b_mma2_addr) >> 4) & 0x3FFF) + (k_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (k_stage * 32 + (unsigned int)sfa_base_col_2) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_3) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_4 = 12;
                    int sfb_base_col_5 = 6;
                    int _mma_a_lo_3 = make_warp_uniform((((smem_a_mma3_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_b_mma3_addr) >> 4) & 0x3FFF) + (k_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (k_stage * 32 + (unsigned int)sfa_base_col_4) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_5) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_6 = 16;
                    int sfb_base_col_7 = 8;
                    int _mma_a_lo_4 = make_warp_uniform((((smem_a_mma4_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_b_mma4_addr) >> 4) & 0x3FFF) + (k_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (k_stage * 32 + (unsigned int)sfa_base_col_6) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_7) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_8 = 20;
                    int sfb_base_col_9 = 10;
                    int _mma_a_lo_5 = make_warp_uniform((((smem_a_mma5_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    int _mma_b_lo_5 = make_warp_uniform((((smem_b_mma5_addr) >> 4) & 0x3FFF) + (k_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (k_stage * 32 + (unsigned int)sfa_base_col_8) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_9) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_10 = 24;
                    int sfb_base_col_11 = 12;
                    int _mma_a_lo_6 = make_warp_uniform((((smem_a_mma6_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    int _mma_b_lo_6 = make_warp_uniform((((smem_b_mma6_addr) >> 4) & 0x3FFF) + (k_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (k_stage * 32 + (unsigned int)sfa_base_col_10) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_11) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_12 = 28;
                    int sfb_base_col_13 = 14;
                    int _mma_a_lo_7 = make_warp_uniform((((smem_a_mma7_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_b_mma7_addr) >> 4) & 0x3FFF) + (k_stage) * 128);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 8)), a_desc + 0, b_desc + 0,
                                0x8020480U, (unsigned int)tmem_sfa + (k_stage * 32 + (unsigned int)sfa_base_col_12) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_13) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    if (elect_sync()) {
                        tcgen05_commit(a_empty_addr + (k_stage) * 8);
                        tcgen05_commit(b_empty_addr + (k_stage) * 8);
                        tcgen05_commit(tmem_sfa_empty_addr + (k_stage) * 8);
                        tcgen05_commit(tmem_sfb_empty_addr + (k_stage) * 8);
                    }
                    if (route_k_4 + 1 == route_k_extent_6) {
                        elect_commit(mma_full_addr + (acc_stage_1) * 8);
                    }
                    {
                        k_stage += 1;
                        if (k_stage == 4) { k_stage = 0; k_phase ^= 1; }
                    }
                }
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_7) * 8, _phase_work_full_7, 10000000);
                }
                unsigned int valid_7 = 0;
                unsigned int next_x_7 = 0;
                unsigned int next_y_7 = 0;
                int response_index_7 = work_stage_7 * 4;
                valid_7 = work_response[response_index_7];
                next_x_7 = work_response[response_index_7 + 1];
                next_y_7 = work_response[response_index_7 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                work_stage_7 += 1;
                if (work_stage_7 == 3) { work_stage_7 = 0; _phase_work_full_7 ^= 1; }
                unsigned int valid_0_7 = valid_7;
                m_tile_7 = next_x_7;
                n_tile_7 = next_y_7;
                if (valid_0_7 == 0) {
                    break;
                }
            }
        }
    // ---- Role: work_id ----
    } else if (warp == 14) {
        { // work_id_main
            unsigned int work_stage_8 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int m_tile_8 = blockIdx.x;
            unsigned int n_tile_8 = blockIdx.y;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_8 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_8 = 0; _tile_iter_8 < grid_m * grid_n; _tile_iter_8++) {
                if (m_tile_8 >= (unsigned int)grid_m || n_tile_8 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                throttle_stage_1 += 1;
                if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                if (elect_sync()) {
                    mbarrier_wait(work_empty_addr + (work_stage_8) * 8, _phase_work_empty);
                    int _atomic_old_0 = atomicAdd(work_counter, 1);
                    int next_linear = _atomic_old_0;
                    int response_index_8 = work_stage_8 * 4;
                    work_response[response_index_8] = next_linear < grid_m * num_non_exiting_ctas[0];
                    work_response[response_index_8 + 1] = next_linear % grid_m;
                    work_response[response_index_8 + 2] = next_linear / grid_m;
                    work_response[response_index_8 + 3] = 0;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_full_addr + (work_stage_8) * 8);
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_8) * 8, _phase_work_full_8, 10000000);
                }
                unsigned int valid_8 = 0;
                unsigned int next_x_8 = 0;
                unsigned int next_y_8 = 0;
                int response_index_9 = work_stage_8 * 4;
                valid_8 = work_response[response_index_9];
                next_x_8 = work_response[response_index_9 + 1];
                next_y_8 = work_response[response_index_9 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_8) * 8);
                work_stage_8 += 1;
                if (work_stage_8 == 3) { work_stage_8 = 0; _phase_work_empty ^= 1; _phase_work_full_8 ^= 1; }
                unsigned int valid_0_8 = valid_8;
                m_tile_8 = next_x_8;
                n_tile_8 = next_y_8;
                if (valid_0_8 == 0) {
                    break;
                }
            }
        }
    // ---- Role: padding ----
    } else if (warp == 15) {
        { // padding_main
            unsigned int work_stage_9 = 0;
            unsigned int m_tile_9 = blockIdx.x;
            unsigned int n_tile_9 = blockIdx.y;
            unsigned int _phase_work_full_9 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_9 = 0; _tile_iter_9 < grid_m * grid_n; _tile_iter_9++) {
                if (m_tile_9 >= (unsigned int)grid_m || n_tile_9 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_9) * 8, _phase_work_full_9, 10000000);
                }
                unsigned int valid_9 = 0;
                unsigned int next_x_9 = 0;
                unsigned int next_y_9 = 0;
                int response_index_10 = work_stage_9 * 4;
                valid_9 = work_response[response_index_10];
                next_x_9 = work_response[response_index_10 + 1];
                next_y_9 = work_response[response_index_10 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_9) * 8);
                work_stage_9 += 1;
                if (work_stage_9 == 3) { work_stage_9 = 0; _phase_work_full_9 ^= 1; }
                unsigned int valid_0_9 = valid_9;
                m_tile_9 = next_x_9;
                n_tile_9 = next_y_9;
                if (valid_0_9 == 0) {
                    break;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
