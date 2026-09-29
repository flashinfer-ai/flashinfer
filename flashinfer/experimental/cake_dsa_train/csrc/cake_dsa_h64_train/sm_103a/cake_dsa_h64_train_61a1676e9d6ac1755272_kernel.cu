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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_Q_K_OFF 1024
#define SMEM_Q_K_STAGE_BYTES 73728
#define SMEM_Q_K_STRIDE 73728
#define SMEM_Q_MN_OFF 1024
#define SMEM_Q_MN_STAGE_BYTES 16384
#define SMEM_Q_MN_STRIDE 16384
#define SMEM_QR_MN_OFF 66560
#define SMEM_QR_MN_STAGE_BYTES 8192
#define SMEM_QR_MN_STRIDE 8192
#define SMEM_DO_K_OFF 74752
#define SMEM_DO_K_STAGE_BYTES 65536
#define SMEM_DO_K_STRIDE 65536
#define SMEM_DO_MN_OFF 74752
#define SMEM_DO_MN_STAGE_BYTES 16384
#define SMEM_DO_MN_STRIDE 16384
#define SMEM_K_K_OFF 140288
#define SMEM_K_K_STAGE_BYTES 73728
#define SMEM_K_K_STRIDE 73728
#define SMEM_V_K_OFF 140288
#define SMEM_V_K_STAGE_BYTES 65536
#define SMEM_V_K_STRIDE 65536
#define SMEM_K_MN_OFF 140288
#define SMEM_K_MN_STAGE_BYTES 16384
#define SMEM_K_MN_STRIDE 16384
#define SMEM_KR_MN_OFF 205824
#define SMEM_KR_MN_STAGE_BYTES 8192
#define SMEM_KR_MN_STRIDE 8192
#define SMEM_K_BLK_OFF 140288
#define SMEM_K_BLK_STAGE_BYTES 8192
#define SMEM_K_BLK_STRIDE 8192
#define SMEM_P_K_OFF 214016
#define SMEM_P_K_STAGE_BYTES 8192
#define SMEM_P_K_STRIDE 8192
#define SMEM_DS_K_OFF 222208
#define SMEM_DS_K_STAGE_BYTES 8192
#define SMEM_DS_K_STRIDE 8192
#define SMEM_DS_MN_OFF 222208
#define SMEM_DS_MN_STAGE_BYTES 8192
#define SMEM_DS_MN_STRIDE 8192
#define SMEM_STATS_OFF 230400
#define SMEM_STATS_STAGE_BYTES 512
#define SMEM_STATS_STRIDE 512
#define SMEM_TILE_IDX_OFF 230912
#define SMEM_TILE_IDX_STAGE_BYTES 640
#define SMEM_TILE_IDX_STRIDE 640
#define SMEM_VALIDITY8_OFF 230912
#define SMEM_VALIDITY8_STAGE_BYTES 640
#define SMEM_VALIDITY8_STRIDE 640
#define SMEM_VALIDITY32_OFF 230912
#define SMEM_VALIDITY32_STAGE_BYTES 640
#define SMEM_VALIDITY32_STRIDE 640
#define SMEM_BLOCKS_WORD_OFF 231552
#define SMEM_BLOCKS_WORD_STAGE_BYTES 4
#define SMEM_BLOCKS_WORD_STRIDE 4
#define SMEM_TOTAL 231680
#define THREADS 640

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


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
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


__device__ __forceinline__ void tma_gather4_gmem2smem(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Canonical .shared::cta form for non-multicast gather4, matching
    // trtllm-gen / cuda_ptx and the PTX ISA qualifier order
    // (dim.dst.src.load_mode.completion_mechanism). Per the PTX grammar,
    // .shared::cluster is reserved for the multicast variant (ctaMask).
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
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

extern "C" {

__global__ __launch_bounds__(640, 1) void
kernel_cake_dsa_h64_train_61a1676e9d6ac1755272(CakeTensorMap const* q_latent, CakeTensorMap const* q_rope, CakeTensorMap const* dout, CakeTensorMap const* dq_latent, CakeTensorMap const* dq_rope, CakeTensorMap const* kv_latent, CakeTensorMap const* k_rope, float* __restrict__ lse, float* __restrict__ delta, int* __restrict__ indices, int* __restrict__ topk_length, float* __restrict__ dkv_f32, float* __restrict__ dkr_f32, int num_queries, int num_kv, int topk, int idx_stride, int indices_offset, int has_topk_length, int token_base, int token_step, float scale_log2, float sm_scale, int pass_lo, int pass_hi, int dq_mode, float* __restrict__ dq_partial, int* __restrict__ key_scratch, int* __restrict__ pass_counts)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define qdo_full_addr (mbar_base + 0)
    #define stats_full_addr (mbar_base + 8)
    #define blocks_ready_addr (mbar_base + 16)
    #define dq_done_addr (mbar_base + 24)
    #define idx_full_addr (mbar_base + 32)
    #define idx_free_addr (mbar_base + 48)
    #define k_full_addr (mbar_base + 64)
    #define k_free_addr (mbar_base + 72)
    #define s_full_addr (mbar_base + 80)
    #define s_free_addr (mbar_base + 88)
    #define dp_full_addr (mbar_base + 96)
    #define dp_free_addr (mbar_base + 104)
    #define p_full_addr (mbar_base + 112)
    #define p_free_addr (mbar_base + 120)
    #define ds_full_addr (mbar_base + 128)
    #define ds_free_addr (mbar_base + 136)
    #define dkv_a_full_addr (mbar_base + 144)
    #define dkv_a_drained_addr (mbar_base + 152)
    #define dkr_full_addr (mbar_base + 160)
    #define dkr_drained_addr (mbar_base + 168)
    #define dkv_b_full_addr (mbar_base + 176)
    #define dkv_b_drained_addr (mbar_base + 184)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(q_latent)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(q_rope)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(dout)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(dq_latent)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(dq_rope)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(kv_latent)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(k_rope)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* q_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_k_addr = smem + 1024;
    __nv_bfloat16* q_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_mn_addr = smem + 1024;
    __nv_bfloat16* qr_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int qr_mn_addr = smem + 66560;
    __nv_bfloat16* do_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int do_k_addr = smem + 74752;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int do_mn_addr = smem + 74752;
    __nv_bfloat16* k_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int k_k_addr = smem + 140288;
    __nv_bfloat16* v_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int v_k_addr = smem + 140288;
    __nv_bfloat16* k_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int k_mn_addr = smem + 140288;
    __nv_bfloat16* kr_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int kr_mn_addr = smem + 205824;
    __nv_bfloat16* k_blk = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int k_blk_addr = smem + 140288;
    __nv_bfloat16* p_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 214016);
    const int p_k_addr = smem + 214016;
    __nv_bfloat16* ds_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 222208);
    const int ds_k_addr = smem + 222208;
    __nv_bfloat16* ds_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 222208);
    const int ds_mn_addr = smem + 222208;
    float* stats = reinterpret_cast<float*>(smem_raw + 230400);
    const int stats_addr = smem + 230400;
    int* tile_idx = reinterpret_cast<int*>(smem_raw + 230912);
    const int tile_idx_addr = smem + 230912;
    uint8_t* validity8 = reinterpret_cast<uint8_t*>(smem_raw + 230912);
    const int validity8_addr = smem + 230912;
    unsigned int* validity32 = reinterpret_cast<unsigned int*>(smem_raw + 230912);
    const int validity32_addr = smem + 230912;
    int* blocks_word = reinterpret_cast<int*>(smem_raw + 231552);
    const int blocks_word_addr = smem + 231552;
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(q_latent)) : "memory"); }
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(q_rope)) : "memory"); }
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(dout)) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(kv_latent)) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(k_rope)) : "memory"); }
    if (warp == 4 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(dq_latent)) : "memory"); }
    if (warp == 4 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(dq_rope)) : "memory"); }

    // Mbarrier init (22 pipeline groups, 0 ordered-sequence groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // qdo_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // stats_full: 1 barriers, init_count=32
            mbarrier_init(smem + 8, 32);
            // blocks_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // dq_done: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // idx_full: 2 barriers, init_count=8
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            // idx_free: 2 barriers, init_count=16
            mbarrier_init(smem + 48, 16);
            mbarrier_init(smem + 56, 16);
            // k_full: 1 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            // k_free: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // s_free: 1 barriers, init_count=128
            mbarrier_init(smem + 88, 128);
            // dp_full: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // dp_free: 1 barriers, init_count=128
            mbarrier_init(smem + 104, 128);
            // p_full: 1 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            // p_free: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // ds_full: 1 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            // ds_free: 1 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            // dkv_a_full: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // dkv_a_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 152, 256);
            // dkr_full: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // dkr_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 168, 256);
            // dkv_b_full: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // dkv_b_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 184, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 4) {
        int _tmem_hold = smem + 192;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;

    // ---- Role: gather ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // gather_main
            int pw = warp;
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks = blocks_word[0];
            if (elect_sync()) {
                #pragma unroll 1
                for (int i = 0; i < blocks; i++) {
                    int slot = i % 2;
                    mbarrier_wait(idx_full_addr + (slot) * 8, i / 2 & 1);
                    int rows[16];
                    rows[0] = tile_idx[slot * 80 + pw * 4];
                    rows[1] = tile_idx[slot * 80 + pw * 4 + 1];
                    rows[2] = tile_idx[slot * 80 + pw * 4 + 2];
                    rows[3] = tile_idx[slot * 80 + pw * 4 + 3];
                    rows[4] = tile_idx[slot * 80 + 16 + pw * 4];
                    rows[5] = tile_idx[slot * 80 + 16 + pw * 4 + 1];
                    rows[6] = tile_idx[slot * 80 + 16 + pw * 4 + 2];
                    rows[7] = tile_idx[slot * 80 + 16 + pw * 4 + 3];
                    rows[8] = tile_idx[slot * 80 + 32 + pw * 4];
                    rows[9] = tile_idx[slot * 80 + 32 + pw * 4 + 1];
                    rows[10] = tile_idx[slot * 80 + 32 + pw * 4 + 2];
                    rows[11] = tile_idx[slot * 80 + 32 + pw * 4 + 3];
                    rows[12] = tile_idx[slot * 80 + 48 + pw * 4];
                    rows[13] = tile_idx[slot * 80 + 48 + pw * 4 + 1];
                    rows[14] = tile_idx[slot * 80 + 48 + pw * 4 + 2];
                    rows[15] = tile_idx[slot * 80 + 48 + pw * 4 + 3];
                    mbarrier_arrive(idx_free_addr + (slot) * 8);
                    mbarrier_wait(k_free_addr, i & 1 ^ 1);
                    mbarrier_arrive_expect_tx(k_full_addr, 18432);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 0 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 0 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 0 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 0 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 8192 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 64 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 8192 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 64 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 8192 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 64 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 8192 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 64 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 16384 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 128 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 16384 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 128 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 16384 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 128 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 16384 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 128 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 24576 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 192 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 24576 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 192 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 24576 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 192 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 24576 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 192 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 32768 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 256 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 32768 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 256 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 32768 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 256 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 32768 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 256 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 40960 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 320 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 40960 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 320 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 40960 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 320 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 40960 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 320 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 49152 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 384 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 49152 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 384 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 49152 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 384 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 49152 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 384 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 57344 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 448 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 57344 + 2048 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 448 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 57344 + 4096 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 448 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 57344 + 6144 + (unsigned int)(pw * 512)), "l"(((1) ? (kv_latent) : (k_rope))), "r"(((1) ? 448 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 65536 + (unsigned int)(pw * 512)), "l"(((0) ? (kv_latent) : (k_rope))), "r"(((0) ? 512 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 65536 + 2048 + (unsigned int)(pw * 512)), "l"(((0) ? (kv_latent) : (k_rope))), "r"(((0) ? 512 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 65536 + 4096 + (unsigned int)(pw * 512)), "l"(((0) ? (kv_latent) : (k_rope))), "r"(((0) ? 512 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + 65536 + 6144 + (unsigned int)(pw * 512)), "l"(((0) ? (kv_latent) : (k_rope))), "r"(((0) ? 512 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
    }
    // ---- Role: compute ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 128;");
        { // compute_main
            int token = token_base + token_step * blockIdx.x;
            int lane_0 = lane;
            int w = warp - 4;
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_1 = blocks_word[0];
            mbarrier_wait(stats_full_addr, 0);
            int h0 = w * 16 + lane_0 / 4;
            float lse0 = stats[h0];
            float lse1 = stats[h0 + 8];
            float dl0 = stats[64 + h0];
            float dl1 = stats[64 + h0 + 8];
            int kb = 2 * (lane_0 % 4);
            int r8 = lane_0 % 8;
            int m4 = lane_0 / 8;
            int kbase = m4 / 2 * 8 + r8;
            int hchunk = (2 * w + m4 % 2 ^ r8) * 16;
            #pragma unroll 1
            for (int i_1 = 0; i_1 < blocks_1; i_1++) {
                unsigned int par = i_1 & 1;
                int slot_1 = i_1 % 2;
                mbarrier_wait(idx_full_addr + (slot_1) * 8, i_1 / 2 & 1);
                unsigned int v0 = validity32[slot_1 * 80 + 64];
                unsigned int v1 = validity32[slot_1 * 80 + 65];
                if (lane_0 == 0) {
                    mbarrier_arrive(idx_free_addr + (slot_1) * 8);
                }
                mbarrier_wait(s_full_addr, par);
                float s[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s[0])), "=r"(*reinterpret_cast<uint32_t*>(&s[1])), "=r"(*reinterpret_cast<uint32_t*>(&s[2])), "=r"(*reinterpret_cast<uint32_t*>(&s[3])), "=r"(*reinterpret_cast<uint32_t*>(&s[4])), "=r"(*reinterpret_cast<uint32_t*>(&s[5])), "=r"(*reinterpret_cast<uint32_t*>(&s[6])), "=r"(*reinterpret_cast<uint32_t*>(&s[7])), "=r"(*reinterpret_cast<uint32_t*>(&s[8])), "=r"(*reinterpret_cast<uint32_t*>(&s[9])), "=r"(*reinterpret_cast<uint32_t*>(&s[10])), "=r"(*reinterpret_cast<uint32_t*>(&s[11])), "=r"(*reinterpret_cast<uint32_t*>(&s[12])), "=r"(*reinterpret_cast<uint32_t*>(&s[13])), "=r"(*reinterpret_cast<uint32_t*>(&s[14])), "=r"(*reinterpret_cast<uint32_t*>(&s[15])), "=r"(*reinterpret_cast<uint32_t*>(&s[16])), "=r"(*reinterpret_cast<uint32_t*>(&s[17])), "=r"(*reinterpret_cast<uint32_t*>(&s[18])), "=r"(*reinterpret_cast<uint32_t*>(&s[19])), "=r"(*reinterpret_cast<uint32_t*>(&s[20])), "=r"(*reinterpret_cast<uint32_t*>(&s[21])), "=r"(*reinterpret_cast<uint32_t*>(&s[22])), "=r"(*reinterpret_cast<uint32_t*>(&s[23])), "=r"(*reinterpret_cast<uint32_t*>(&s[24])), "=r"(*reinterpret_cast<uint32_t*>(&s[25])), "=r"(*reinterpret_cast<uint32_t*>(&s[26])), "=r"(*reinterpret_cast<uint32_t*>(&s[27])), "=r"(*reinterpret_cast<uint32_t*>(&s[28])), "=r"(*reinterpret_cast<uint32_t*>(&s[29])), "=r"(*reinterpret_cast<uint32_t*>(&s[30])), "=r"(*reinterpret_cast<uint32_t*>(&s[31]))
                    : "r"(tmem_tmem));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float p[32];
                float _fma_0 = __fmaf_rn(s[0], scale_log2, -lse0);
                float _exp2_0 = approx_exp2(_fma_0);
                p[0] = _exp2_0;
                float _fma_1 = __fmaf_rn(s[2], scale_log2, -lse1);
                float _exp2_1 = approx_exp2(_fma_1);
                p[2] = _exp2_1;
                unsigned int bit = ((1) ? v0 : v1) >> (unsigned int)kb & 1;
                if (bit == 0) {
                    p[0] = 0.0f;
                    p[2] = 0.0f;
                }
                float _fma_2 = __fmaf_rn(s[1], scale_log2, -lse0);
                float _exp2_2 = approx_exp2(_fma_2);
                p[1] = _exp2_2;
                float _fma_3 = __fmaf_rn(s[3], scale_log2, -lse1);
                float _exp2_3 = approx_exp2(_fma_3);
                p[3] = _exp2_3;
                unsigned int bit_0 = ((1) ? v0 : v1) >> (unsigned int)(1 + kb) & 1;
                if (bit_0 == 0) {
                    p[1] = 0.0f;
                    p[3] = 0.0f;
                }
                float _fma_4 = __fmaf_rn(s[4], scale_log2, -lse0);
                float _exp2_4 = approx_exp2(_fma_4);
                p[4] = _exp2_4;
                float _fma_5 = __fmaf_rn(s[6], scale_log2, -lse1);
                float _exp2_5 = approx_exp2(_fma_5);
                p[6] = _exp2_5;
                unsigned int bit_1 = ((1) ? v0 : v1) >> (unsigned int)(8 + kb) & 1;
                if (bit_1 == 0) {
                    p[4] = 0.0f;
                    p[6] = 0.0f;
                }
                float _fma_6 = __fmaf_rn(s[5], scale_log2, -lse0);
                float _exp2_6 = approx_exp2(_fma_6);
                p[5] = _exp2_6;
                float _fma_7 = __fmaf_rn(s[7], scale_log2, -lse1);
                float _exp2_7 = approx_exp2(_fma_7);
                p[7] = _exp2_7;
                unsigned int bit_2 = ((1) ? v0 : v1) >> (unsigned int)(9 + kb) & 1;
                if (bit_2 == 0) {
                    p[5] = 0.0f;
                    p[7] = 0.0f;
                }
                float _fma_8 = __fmaf_rn(s[8], scale_log2, -lse0);
                float _exp2_8 = approx_exp2(_fma_8);
                p[8] = _exp2_8;
                float _fma_9 = __fmaf_rn(s[10], scale_log2, -lse1);
                float _exp2_9 = approx_exp2(_fma_9);
                p[10] = _exp2_9;
                unsigned int bit_3 = ((1) ? v0 : v1) >> (unsigned int)(16 + kb) & 1;
                if (bit_3 == 0) {
                    p[8] = 0.0f;
                    p[10] = 0.0f;
                }
                float _fma_10 = __fmaf_rn(s[9], scale_log2, -lse0);
                float _exp2_10 = approx_exp2(_fma_10);
                p[9] = _exp2_10;
                float _fma_11 = __fmaf_rn(s[11], scale_log2, -lse1);
                float _exp2_11 = approx_exp2(_fma_11);
                p[11] = _exp2_11;
                unsigned int bit_4 = ((1) ? v0 : v1) >> (unsigned int)(17 + kb) & 1;
                if (bit_4 == 0) {
                    p[9] = 0.0f;
                    p[11] = 0.0f;
                }
                float _fma_12 = __fmaf_rn(s[12], scale_log2, -lse0);
                float _exp2_12 = approx_exp2(_fma_12);
                p[12] = _exp2_12;
                float _fma_13 = __fmaf_rn(s[14], scale_log2, -lse1);
                float _exp2_13 = approx_exp2(_fma_13);
                p[14] = _exp2_13;
                unsigned int bit_5 = ((1) ? v0 : v1) >> (unsigned int)(24 + kb) & 1;
                if (bit_5 == 0) {
                    p[12] = 0.0f;
                    p[14] = 0.0f;
                }
                float _fma_14 = __fmaf_rn(s[13], scale_log2, -lse0);
                float _exp2_14 = approx_exp2(_fma_14);
                p[13] = _exp2_14;
                float _fma_15 = __fmaf_rn(s[15], scale_log2, -lse1);
                float _exp2_15 = approx_exp2(_fma_15);
                p[15] = _exp2_15;
                unsigned int bit_6 = ((1) ? v0 : v1) >> (unsigned int)(25 + kb) & 1;
                if (bit_6 == 0) {
                    p[13] = 0.0f;
                    p[15] = 0.0f;
                }
                float _fma_16 = __fmaf_rn(s[16], scale_log2, -lse0);
                float _exp2_16 = approx_exp2(_fma_16);
                p[16] = _exp2_16;
                float _fma_17 = __fmaf_rn(s[18], scale_log2, -lse1);
                float _exp2_17 = approx_exp2(_fma_17);
                p[18] = _exp2_17;
                unsigned int bit_7 = ((0) ? v0 : v1) >> (unsigned int)kb & 1;
                if (bit_7 == 0) {
                    p[16] = 0.0f;
                    p[18] = 0.0f;
                }
                float _fma_18 = __fmaf_rn(s[17], scale_log2, -lse0);
                float _exp2_18 = approx_exp2(_fma_18);
                p[17] = _exp2_18;
                float _fma_19 = __fmaf_rn(s[19], scale_log2, -lse1);
                float _exp2_19 = approx_exp2(_fma_19);
                p[19] = _exp2_19;
                unsigned int bit_8 = ((0) ? v0 : v1) >> (unsigned int)(1 + kb) & 1;
                if (bit_8 == 0) {
                    p[17] = 0.0f;
                    p[19] = 0.0f;
                }
                float _fma_20 = __fmaf_rn(s[20], scale_log2, -lse0);
                float _exp2_20 = approx_exp2(_fma_20);
                p[20] = _exp2_20;
                float _fma_21 = __fmaf_rn(s[22], scale_log2, -lse1);
                float _exp2_21 = approx_exp2(_fma_21);
                p[22] = _exp2_21;
                unsigned int bit_9 = ((0) ? v0 : v1) >> (unsigned int)(8 + kb) & 1;
                if (bit_9 == 0) {
                    p[20] = 0.0f;
                    p[22] = 0.0f;
                }
                float _fma_22 = __fmaf_rn(s[21], scale_log2, -lse0);
                float _exp2_22 = approx_exp2(_fma_22);
                p[21] = _exp2_22;
                float _fma_23 = __fmaf_rn(s[23], scale_log2, -lse1);
                float _exp2_23 = approx_exp2(_fma_23);
                p[23] = _exp2_23;
                unsigned int bit_10 = ((0) ? v0 : v1) >> (unsigned int)(9 + kb) & 1;
                if (bit_10 == 0) {
                    p[21] = 0.0f;
                    p[23] = 0.0f;
                }
                float _fma_24 = __fmaf_rn(s[24], scale_log2, -lse0);
                float _exp2_24 = approx_exp2(_fma_24);
                p[24] = _exp2_24;
                float _fma_25 = __fmaf_rn(s[26], scale_log2, -lse1);
                float _exp2_25 = approx_exp2(_fma_25);
                p[26] = _exp2_25;
                unsigned int bit_11 = ((0) ? v0 : v1) >> (unsigned int)(16 + kb) & 1;
                if (bit_11 == 0) {
                    p[24] = 0.0f;
                    p[26] = 0.0f;
                }
                float _fma_26 = __fmaf_rn(s[25], scale_log2, -lse0);
                float _exp2_26 = approx_exp2(_fma_26);
                p[25] = _exp2_26;
                float _fma_27 = __fmaf_rn(s[27], scale_log2, -lse1);
                float _exp2_27 = approx_exp2(_fma_27);
                p[27] = _exp2_27;
                unsigned int bit_12 = ((0) ? v0 : v1) >> (unsigned int)(17 + kb) & 1;
                if (bit_12 == 0) {
                    p[25] = 0.0f;
                    p[27] = 0.0f;
                }
                float _fma_28 = __fmaf_rn(s[28], scale_log2, -lse0);
                float _exp2_28 = approx_exp2(_fma_28);
                p[28] = _exp2_28;
                float _fma_29 = __fmaf_rn(s[30], scale_log2, -lse1);
                float _exp2_29 = approx_exp2(_fma_29);
                p[30] = _exp2_29;
                unsigned int bit_13 = ((0) ? v0 : v1) >> (unsigned int)(24 + kb) & 1;
                if (bit_13 == 0) {
                    p[28] = 0.0f;
                    p[30] = 0.0f;
                }
                float _fma_30 = __fmaf_rn(s[29], scale_log2, -lse0);
                float _exp2_30 = approx_exp2(_fma_30);
                p[29] = _exp2_30;
                float _fma_31 = __fmaf_rn(s[31], scale_log2, -lse1);
                float _exp2_31 = approx_exp2(_fma_31);
                p[31] = _exp2_31;
                unsigned int bit_14 = ((0) ? v0 : v1) >> (unsigned int)(25 + kb) & 1;
                if (bit_14 == 0) {
                    p[29] = 0.0f;
                    p[31] = 0.0f;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr);
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                mbarrier_wait(p_free_addr, par ^ 1);
                uint32_t p_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 0], p[_lp*2+1 + 0]));
                    p_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int paddr = p_k_addr + (unsigned int)(kbase * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(paddr);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[3]))
                    : "memory");
                unsigned int paddr_15 = p_k_addr + (unsigned int)((kbase + 16) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(paddr_15);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[7]))
                    : "memory");
                unsigned int paddr_16 = p_k_addr + (unsigned int)((kbase + 32) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(paddr_16);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[11]))
                    : "memory");
                unsigned int paddr_17 = p_k_addr + (unsigned int)((kbase + 48) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(paddr_17);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                mbarrier_wait(dp_full_addr, par);
                float dp[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dp[0])), "=r"(*reinterpret_cast<uint32_t*>(&dp[1])), "=r"(*reinterpret_cast<uint32_t*>(&dp[2])), "=r"(*reinterpret_cast<uint32_t*>(&dp[3])), "=r"(*reinterpret_cast<uint32_t*>(&dp[4])), "=r"(*reinterpret_cast<uint32_t*>(&dp[5])), "=r"(*reinterpret_cast<uint32_t*>(&dp[6])), "=r"(*reinterpret_cast<uint32_t*>(&dp[7])), "=r"(*reinterpret_cast<uint32_t*>(&dp[8])), "=r"(*reinterpret_cast<uint32_t*>(&dp[9])), "=r"(*reinterpret_cast<uint32_t*>(&dp[10])), "=r"(*reinterpret_cast<uint32_t*>(&dp[11])), "=r"(*reinterpret_cast<uint32_t*>(&dp[12])), "=r"(*reinterpret_cast<uint32_t*>(&dp[13])), "=r"(*reinterpret_cast<uint32_t*>(&dp[14])), "=r"(*reinterpret_cast<uint32_t*>(&dp[15])), "=r"(*reinterpret_cast<uint32_t*>(&dp[16])), "=r"(*reinterpret_cast<uint32_t*>(&dp[17])), "=r"(*reinterpret_cast<uint32_t*>(&dp[18])), "=r"(*reinterpret_cast<uint32_t*>(&dp[19])), "=r"(*reinterpret_cast<uint32_t*>(&dp[20])), "=r"(*reinterpret_cast<uint32_t*>(&dp[21])), "=r"(*reinterpret_cast<uint32_t*>(&dp[22])), "=r"(*reinterpret_cast<uint32_t*>(&dp[23])), "=r"(*reinterpret_cast<uint32_t*>(&dp[24])), "=r"(*reinterpret_cast<uint32_t*>(&dp[25])), "=r"(*reinterpret_cast<uint32_t*>(&dp[26])), "=r"(*reinterpret_cast<uint32_t*>(&dp[27])), "=r"(*reinterpret_cast<uint32_t*>(&dp[28])), "=r"(*reinterpret_cast<uint32_t*>(&dp[29])), "=r"(*reinterpret_cast<uint32_t*>(&dp[30])), "=r"(*reinterpret_cast<uint32_t*>(&dp[31]))
                    : "r"(tmem_tmem + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ds[32];
                ds[0] = p[0] * (dp[0] - dl0) * sm_scale;
                ds[2] = p[2] * (dp[2] - dl1) * sm_scale;
                ds[1] = p[1] * (dp[1] - dl0) * sm_scale;
                ds[3] = p[3] * (dp[3] - dl1) * sm_scale;
                ds[4] = p[4] * (dp[4] - dl0) * sm_scale;
                ds[6] = p[6] * (dp[6] - dl1) * sm_scale;
                ds[5] = p[5] * (dp[5] - dl0) * sm_scale;
                ds[7] = p[7] * (dp[7] - dl1) * sm_scale;
                ds[8] = p[8] * (dp[8] - dl0) * sm_scale;
                ds[10] = p[10] * (dp[10] - dl1) * sm_scale;
                ds[9] = p[9] * (dp[9] - dl0) * sm_scale;
                ds[11] = p[11] * (dp[11] - dl1) * sm_scale;
                ds[12] = p[12] * (dp[12] - dl0) * sm_scale;
                ds[14] = p[14] * (dp[14] - dl1) * sm_scale;
                ds[13] = p[13] * (dp[13] - dl0) * sm_scale;
                ds[15] = p[15] * (dp[15] - dl1) * sm_scale;
                ds[16] = p[16] * (dp[16] - dl0) * sm_scale;
                ds[18] = p[18] * (dp[18] - dl1) * sm_scale;
                ds[17] = p[17] * (dp[17] - dl0) * sm_scale;
                ds[19] = p[19] * (dp[19] - dl1) * sm_scale;
                ds[20] = p[20] * (dp[20] - dl0) * sm_scale;
                ds[22] = p[22] * (dp[22] - dl1) * sm_scale;
                ds[21] = p[21] * (dp[21] - dl0) * sm_scale;
                ds[23] = p[23] * (dp[23] - dl1) * sm_scale;
                ds[24] = p[24] * (dp[24] - dl0) * sm_scale;
                ds[26] = p[26] * (dp[26] - dl1) * sm_scale;
                ds[25] = p[25] * (dp[25] - dl0) * sm_scale;
                ds[27] = p[27] * (dp[27] - dl1) * sm_scale;
                ds[28] = p[28] * (dp[28] - dl0) * sm_scale;
                ds[30] = p[30] * (dp[30] - dl1) * sm_scale;
                ds[29] = p[29] * (dp[29] - dl0) * sm_scale;
                ds[31] = p[31] * (dp[31] - dl1) * sm_scale;
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dp_free_addr);
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                mbarrier_wait(ds_free_addr, par ^ 1);
                uint32_t ds_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ds[_lp*2 + 0], ds[_lp*2+1 + 0]));
                    ds_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                unsigned int daddr = ds_k_addr + (unsigned int)(kbase * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(daddr);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[3]))
                    : "memory");
                unsigned int daddr_18 = ds_k_addr + (unsigned int)((kbase + 16) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(daddr_18);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[7]))
                    : "memory");
                unsigned int daddr_19 = ds_k_addr + (unsigned int)((kbase + 32) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(daddr_19);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[11]))
                    : "memory");
                unsigned int daddr_20 = ds_k_addr + (unsigned int)((kbase + 48) * 128) + (unsigned int)hchunk;
                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(daddr_20);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[15]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(ds_full_addr);
            }
            mbarrier_wait(dq_done_addr, 0);
            float dq[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq[31]))
                : "r"(tmem_tmem + 192));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq[_lp*2 + 0], dq[_lp*2+1 + 0]));
                dq_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead = 8 * (m4 / 2) + r8;
            int qchunk = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead * 128) + (unsigned int)((qchunk ^ r8) * 16);
            uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(qaddr);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[3]))
                : "memory");
            int qhead_1 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_2 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_3 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_1 * 128) + (unsigned int)((qchunk_2 ^ r8) * 16);
            uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(qaddr_3);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[7]))
                : "memory");
            int qhead_4 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_5 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_6 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_4 * 128) + (unsigned int)((qchunk_5 ^ r8) * 16);
            uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(qaddr_6);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[11]))
                : "memory");
            int qhead_7 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_8 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_9 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_7 * 128) + (unsigned int)((qchunk_8 ^ r8) * 16);
            uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(qaddr_9);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[15]))
                : "memory");
            float dq_10[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_10[31]))
                : "r"(tmem_tmem + 192 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_10_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_10[_lp*2 + 0], dq_10[_lp*2+1 + 0]));
                dq_10_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_11 = 8 * (m4 / 2) + r8;
            int qchunk_12 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_13 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_11 * 128) + (unsigned int)((qchunk_12 ^ r8) * 16);
            uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(qaddr_13);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[3]))
                : "memory");
            int qhead_14 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_15 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_16 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_14 * 128) + (unsigned int)((qchunk_15 ^ r8) * 16);
            uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(qaddr_16);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[7]))
                : "memory");
            int qhead_17 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_18 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_19 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_17 * 128) + (unsigned int)((qchunk_18 ^ r8) * 16);
            uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(qaddr_19);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[11]))
                : "memory");
            int qhead_20 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_21 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_22 = k_blk_addr + (unsigned int)(w / 2 * 8192) + (unsigned int)(qhead_20 * 128) + (unsigned int)((qchunk_21 ^ r8) * 16);
            uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(qaddr_22);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_10_bf16[15]))
                : "memory");
            float dq_23[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_23[31]))
                : "r"(tmem_tmem + 192 + 64));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_23_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_23[_lp*2 + 0], dq_23[_lp*2+1 + 0]));
                dq_23_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_24 = 8 * (m4 / 2) + r8;
            int qchunk_25 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_26 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_24 * 128) + (unsigned int)((qchunk_25 ^ r8) * 16);
            uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(qaddr_26);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[3]))
                : "memory");
            int qhead_27 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_28 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_29 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_27 * 128) + (unsigned int)((qchunk_28 ^ r8) * 16);
            uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(qaddr_29);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[7]))
                : "memory");
            int qhead_30 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_31 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_32 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_30 * 128) + (unsigned int)((qchunk_31 ^ r8) * 16);
            uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(qaddr_32);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[11]))
                : "memory");
            int qhead_33 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_34 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_35 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_33 * 128) + (unsigned int)((qchunk_34 ^ r8) * 16);
            uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(qaddr_35);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_23_bf16[15]))
                : "memory");
            float dq_36[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_36[31]))
                : "r"(tmem_tmem + 192 + 64 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_36_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_36[_lp*2 + 0], dq_36[_lp*2+1 + 0]));
                dq_36_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_37 = 8 * (m4 / 2) + r8;
            int qchunk_38 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_39 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_37 * 128) + (unsigned int)((qchunk_38 ^ r8) * 16);
            uint32_t _stmatrix_addr_20 = static_cast<uint32_t>(qaddr_39);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_20), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[3]))
                : "memory");
            int qhead_40 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_41 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_42 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_40 * 128) + (unsigned int)((qchunk_41 ^ r8) * 16);
            uint32_t _stmatrix_addr_21 = static_cast<uint32_t>(qaddr_42);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_21), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[7]))
                : "memory");
            int qhead_43 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_44 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_45 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_43 * 128) + (unsigned int)((qchunk_44 ^ r8) * 16);
            uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(qaddr_45);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[11]))
                : "memory");
            int qhead_46 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_47 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_48 = k_blk_addr + (unsigned int)((2 + w / 2) * 8192) + (unsigned int)(qhead_46 * 128) + (unsigned int)((qchunk_47 ^ r8) * 16);
            uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(qaddr_48);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_36_bf16[15]))
                : "memory");
            float dq_49[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_49[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_49[31]))
                : "r"(tmem_tmem + 192 + 128));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_49_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_49[_lp*2 + 0], dq_49[_lp*2+1 + 0]));
                dq_49_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_50 = 8 * (m4 / 2) + r8;
            int qchunk_51 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_52 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_50 * 128) + (unsigned int)((qchunk_51 ^ r8) * 16);
            uint32_t _stmatrix_addr_24 = static_cast<uint32_t>(qaddr_52);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[3]))
                : "memory");
            int qhead_53 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_54 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_55 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_53 * 128) + (unsigned int)((qchunk_54 ^ r8) * 16);
            uint32_t _stmatrix_addr_25 = static_cast<uint32_t>(qaddr_55);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_25), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[7]))
                : "memory");
            int qhead_56 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_57 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_58 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_56 * 128) + (unsigned int)((qchunk_57 ^ r8) * 16);
            uint32_t _stmatrix_addr_26 = static_cast<uint32_t>(qaddr_58);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_26), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[11]))
                : "memory");
            int qhead_59 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_60 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_61 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_59 * 128) + (unsigned int)((qchunk_60 ^ r8) * 16);
            uint32_t _stmatrix_addr_27 = static_cast<uint32_t>(qaddr_61);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_27), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_49_bf16[15]))
                : "memory");
            float dq_62[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_62[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_62[31]))
                : "r"(tmem_tmem + 192 + 128 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_62_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_62[_lp*2 + 0], dq_62[_lp*2+1 + 0]));
                dq_62_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_63 = 8 * (m4 / 2) + r8;
            int qchunk_64 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_65 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_63 * 128) + (unsigned int)((qchunk_64 ^ r8) * 16);
            uint32_t _stmatrix_addr_28 = static_cast<uint32_t>(qaddr_65);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_28), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[3]))
                : "memory");
            int qhead_66 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_67 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_68 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_66 * 128) + (unsigned int)((qchunk_67 ^ r8) * 16);
            uint32_t _stmatrix_addr_29 = static_cast<uint32_t>(qaddr_68);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_29), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[7]))
                : "memory");
            int qhead_69 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_70 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_71 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_69 * 128) + (unsigned int)((qchunk_70 ^ r8) * 16);
            uint32_t _stmatrix_addr_30 = static_cast<uint32_t>(qaddr_71);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_30), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[11]))
                : "memory");
            int qhead_72 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_73 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_74 = k_blk_addr + (unsigned int)((4 + w / 2) * 8192) + (unsigned int)(qhead_72 * 128) + (unsigned int)((qchunk_73 ^ r8) * 16);
            uint32_t _stmatrix_addr_31 = static_cast<uint32_t>(qaddr_74);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_31), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_62_bf16[15]))
                : "memory");
            float dq_75[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_75[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_75[31]))
                : "r"(tmem_tmem + 192 + 192));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_75_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_75[_lp*2 + 0], dq_75[_lp*2+1 + 0]));
                dq_75_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_76 = 8 * (m4 / 2) + r8;
            int qchunk_77 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_78 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_76 * 128) + (unsigned int)((qchunk_77 ^ r8) * 16);
            uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(qaddr_78);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[3]))
                : "memory");
            int qhead_79 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_80 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_81 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_79 * 128) + (unsigned int)((qchunk_80 ^ r8) * 16);
            uint32_t _stmatrix_addr_33 = static_cast<uint32_t>(qaddr_81);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_33), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[7]))
                : "memory");
            int qhead_82 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_83 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_84 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_82 * 128) + (unsigned int)((qchunk_83 ^ r8) * 16);
            uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(qaddr_84);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[11]))
                : "memory");
            int qhead_85 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_86 = 4 * (w % 2) + m4 % 2;
            unsigned int qaddr_87 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_85 * 128) + (unsigned int)((qchunk_86 ^ r8) * 16);
            uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(qaddr_87);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_75_bf16[15]))
                : "memory");
            float dq_88[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_88[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[3])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[4])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[5])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[6])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[7])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[8])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[9])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[10])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[11])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[12])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[13])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[14])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[15])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[16])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[17])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[18])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[19])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[20])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[21])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[22])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[23])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[24])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[25])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[26])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[27])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[28])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[29])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[30])), "=r"(*reinterpret_cast<uint32_t*>(&dq_88[31]))
                : "r"(tmem_tmem + 192 + 192 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_88_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_88[_lp*2 + 0], dq_88[_lp*2+1 + 0]));
                dq_88_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int qhead_89 = 8 * (m4 / 2) + r8;
            int qchunk_90 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_91 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_89 * 128) + (unsigned int)((qchunk_90 ^ r8) * 16);
            uint32_t _stmatrix_addr_36 = static_cast<uint32_t>(qaddr_91);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_36), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[3]))
                : "memory");
            int qhead_92 = 16 + 8 * (m4 / 2) + r8;
            int qchunk_93 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_94 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_92 * 128) + (unsigned int)((qchunk_93 ^ r8) * 16);
            uint32_t _stmatrix_addr_37 = static_cast<uint32_t>(qaddr_94);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_37), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[7]))
                : "memory");
            int qhead_95 = 32 + 8 * (m4 / 2) + r8;
            int qchunk_96 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_97 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_95 * 128) + (unsigned int)((qchunk_96 ^ r8) * 16);
            uint32_t _stmatrix_addr_38 = static_cast<uint32_t>(qaddr_97);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_38), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[11]))
                : "memory");
            int qhead_98 = 48 + 8 * (m4 / 2) + r8;
            int qchunk_99 = 4 * (w % 2) + 2 + m4 % 2;
            unsigned int qaddr_100 = k_blk_addr + (unsigned int)((6 + w / 2) * 8192) + (unsigned int)(qhead_98 * 128) + (unsigned int)((qchunk_99 ^ r8) * 16);
            uint32_t _stmatrix_addr_39 = static_cast<uint32_t>(qaddr_100);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_39), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dq_88_bf16[15]))
                : "memory");
            float dqr[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dqr[0])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[1])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[2])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[3])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[4])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[5])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[6])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[7])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[8])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[9])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[10])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[11])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[12])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[13])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[14])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[15])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[16])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[17])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[18])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[19])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[20])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[21])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[22])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[23])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[24])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[25])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[26])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[27])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[28])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[29])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[30])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[31]))
                : "r"(tmem_tmem + 448));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dqr_bf16[16];
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dqr[_lp*2 + 0], dqr[_lp*2+1 + 0]));
                dqr_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            int rhead = 8 * (m4 / 2) + r8;
            int rchunk = 2 * w + m4 % 2;
            unsigned int raddr = k_blk_addr + 65536 + (unsigned int)(rhead * 128) + (unsigned int)((rchunk ^ r8) * 16);
            uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(raddr);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[3]))
                : "memory");
            int rhead_101 = 16 + 8 * (m4 / 2) + r8;
            int rchunk_102 = 2 * w + m4 % 2;
            unsigned int raddr_103 = k_blk_addr + 65536 + (unsigned int)(rhead_101 * 128) + (unsigned int)((rchunk_102 ^ r8) * 16);
            uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(raddr_103);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[7]))
                : "memory");
            int rhead_104 = 32 + 8 * (m4 / 2) + r8;
            int rchunk_105 = 2 * w + m4 % 2;
            unsigned int raddr_106 = k_blk_addr + 65536 + (unsigned int)(rhead_104 * 128) + (unsigned int)((rchunk_105 ^ r8) * 16);
            uint32_t _stmatrix_addr_42 = static_cast<uint32_t>(raddr_106);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_42), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[11]))
                : "memory");
            int rhead_107 = 48 + 8 * (m4 / 2) + r8;
            int rchunk_108 = 2 * w + m4 % 2;
            unsigned int raddr_109 = k_blk_addr + 65536 + (unsigned int)(rhead_107 * 128) + (unsigned int)((rchunk_108 ^ r8) * 16);
            uint32_t _stmatrix_addr_43 = static_cast<uint32_t>(raddr_109);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_43), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[15]))
                : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (w == 0) {
                if (elect_sync()) {
                    tma_store_4d(dq_latent, 0, 0, 0, token, k_blk_addr);
                    tma_store_4d(dq_latent, 0, 0, 1, token, k_blk_addr + 8192);
                    tma_store_4d(dq_latent, 0, 0, 2, token, k_blk_addr + 16384);
                    tma_store_4d(dq_latent, 0, 0, 3, token, k_blk_addr + 24576);
                    tma_store_4d(dq_latent, 0, 0, 4, token, k_blk_addr + 32768);
                    tma_store_4d(dq_latent, 0, 0, 5, token, k_blk_addr + 40960);
                    tma_store_4d(dq_latent, 0, 0, 6, token, k_blk_addr + 49152);
                    tma_store_4d(dq_latent, 0, 0, 7, token, k_blk_addr + 57344);
                    tma_store_4d(dq_rope, 0, 0, 0, token, k_blk_addr + 65536);
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }
    // ---- Role: reduce ----
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // reduce_main
            int lane_0_1 = lane;
            int w2 = (warp - 8) % 4;
            int wg = (warp - 8) / 4;
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_2 = blocks_word[0];
            int lpos = 32 * w2 + 4 * (lane_0_1 / 4);
            int rpos = 16 * w2 + 2 * (lane_0_1 / 4);
            #pragma unroll 1
            for (int i_2 = 0; i_2 < blocks_2; i_2++) {
                unsigned int par_1 = i_2 & 1;
                int slot_2 = i_2 % 2;
                mbarrier_wait(idx_full_addr + (slot_2) * 8, i_2 / 2 & 1);
                int keys[8];
                keys[0] = tile_idx[slot_2 * 80 + wg * 32 + 2 * (lane_0_1 % 4)];
                keys[1] = tile_idx[slot_2 * 80 + wg * 32 + 2 * (lane_0_1 % 4) + 1];
                keys[2] = tile_idx[slot_2 * 80 + wg * 32 + 8 + 2 * (lane_0_1 % 4)];
                keys[3] = tile_idx[slot_2 * 80 + wg * 32 + 8 + 2 * (lane_0_1 % 4) + 1];
                keys[4] = tile_idx[slot_2 * 80 + wg * 32 + 16 + 2 * (lane_0_1 % 4)];
                keys[5] = tile_idx[slot_2 * 80 + wg * 32 + 16 + 2 * (lane_0_1 % 4) + 1];
                keys[6] = tile_idx[slot_2 * 80 + wg * 32 + 24 + 2 * (lane_0_1 % 4)];
                keys[7] = tile_idx[slot_2 * 80 + wg * 32 + 24 + 2 * (lane_0_1 % 4) + 1];
                if (lane_0_1 == 0) {
                    mbarrier_arrive(idx_free_addr + (slot_2) * 8);
                }
                mbarrier_wait(dkv_a_full_addr, par_1);
                float a0[32];
                float a1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a0[0])), "=r"(*reinterpret_cast<uint32_t*>(&a0[1])), "=r"(*reinterpret_cast<uint32_t*>(&a0[2])), "=r"(*reinterpret_cast<uint32_t*>(&a0[3])), "=r"(*reinterpret_cast<uint32_t*>(&a0[4])), "=r"(*reinterpret_cast<uint32_t*>(&a0[5])), "=r"(*reinterpret_cast<uint32_t*>(&a0[6])), "=r"(*reinterpret_cast<uint32_t*>(&a0[7])), "=r"(*reinterpret_cast<uint32_t*>(&a0[8])), "=r"(*reinterpret_cast<uint32_t*>(&a0[9])), "=r"(*reinterpret_cast<uint32_t*>(&a0[10])), "=r"(*reinterpret_cast<uint32_t*>(&a0[11])), "=r"(*reinterpret_cast<uint32_t*>(&a0[12])), "=r"(*reinterpret_cast<uint32_t*>(&a0[13])), "=r"(*reinterpret_cast<uint32_t*>(&a0[14])), "=r"(*reinterpret_cast<uint32_t*>(&a0[15]))
                    : "r"(tmem_tmem + 64 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[15]))
                    : "r"(tmem_tmem + 64 + wg * 32 + 1048576));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a1[0])), "=r"(*reinterpret_cast<uint32_t*>(&a1[1])), "=r"(*reinterpret_cast<uint32_t*>(&a1[2])), "=r"(*reinterpret_cast<uint32_t*>(&a1[3])), "=r"(*reinterpret_cast<uint32_t*>(&a1[4])), "=r"(*reinterpret_cast<uint32_t*>(&a1[5])), "=r"(*reinterpret_cast<uint32_t*>(&a1[6])), "=r"(*reinterpret_cast<uint32_t*>(&a1[7])), "=r"(*reinterpret_cast<uint32_t*>(&a1[8])), "=r"(*reinterpret_cast<uint32_t*>(&a1[9])), "=r"(*reinterpret_cast<uint32_t*>(&a1[10])), "=r"(*reinterpret_cast<uint32_t*>(&a1[11])), "=r"(*reinterpret_cast<uint32_t*>(&a1[12])), "=r"(*reinterpret_cast<uint32_t*>(&a1[13])), "=r"(*reinterpret_cast<uint32_t*>(&a1[14])), "=r"(*reinterpret_cast<uint32_t*>(&a1[15]))
                    : "r"(tmem_tmem + 128 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[15]))
                    : "r"(tmem_tmem + 128 + wg * 32 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dkv_a_drained_addr);
                float rk[16];
                {
                    mbarrier_wait(dkr_full_addr, par_1);
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&rk[0])), "=r"(*reinterpret_cast<uint32_t*>(&rk[1])), "=r"(*reinterpret_cast<uint32_t*>(&rk[2])), "=r"(*reinterpret_cast<uint32_t*>(&rk[3])), "=r"(*reinterpret_cast<uint32_t*>(&rk[4])), "=r"(*reinterpret_cast<uint32_t*>(&rk[5])), "=r"(*reinterpret_cast<uint32_t*>(&rk[6])), "=r"(*reinterpret_cast<uint32_t*>(&rk[7])), "=r"(*reinterpret_cast<uint32_t*>(&rk[8])), "=r"(*reinterpret_cast<uint32_t*>(&rk[9])), "=r"(*reinterpret_cast<uint32_t*>(&rk[10])), "=r"(*reinterpret_cast<uint32_t*>(&rk[11])), "=r"(*reinterpret_cast<uint32_t*>(&rk[12])), "=r"(*reinterpret_cast<uint32_t*>(&rk[13])), "=r"(*reinterpret_cast<uint32_t*>(&rk[14])), "=r"(*reinterpret_cast<uint32_t*>(&rk[15]))
                        : "r"(tmem_tmem + wg * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dkr_drained_addr);
                }
                int key = keys[0];
                if (key >= 0) {
                    long long base = (long long)key * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base])), "f"(a0[0]), "f"(a0[2]), "f"(a0[16]), "f"(a0[18]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base + 128])), "f"(a1[0]), "f"(a1[2]), "f"(a1[16]), "f"(a1[18]) : "memory");
                    {
                        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key * 64 + (long long)rpos])), "f"(rk[0]), "f"(rk[2]) : "memory");
                    }
                }
                int key_0 = keys[1];
                if (key_0 >= 0) {
                    long long base_1 = (long long)key_0 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_1])), "f"(a0[1]), "f"(a0[3]), "f"(a0[17]), "f"(a0[19]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_1 + 128])), "f"(a1[1]), "f"(a1[3]), "f"(a1[17]), "f"(a1[19]) : "memory");
                    {
                        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_0 * 64 + (long long)rpos])), "f"(rk[1]), "f"(rk[3]) : "memory");
                    }
                }
                int key_1 = keys[2];
                if (key_1 >= 0) {
                    long long base_2 = (long long)key_1 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_2])), "f"(a0[4]), "f"(a0[6]), "f"(a0[20]), "f"(a0[22]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_2 + 128])), "f"(a1[4]), "f"(a1[6]), "f"(a1[20]), "f"(a1[22]) : "memory");
                    {
                        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_1 * 64 + (long long)rpos])), "f"(rk[4]), "f"(rk[6]) : "memory");
                    }
                }
                int key_2 = keys[3];
                if (key_2 >= 0) {
                    long long base_3 = (long long)key_2 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_3])), "f"(a0[5]), "f"(a0[7]), "f"(a0[21]), "f"(a0[23]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_3 + 128])), "f"(a1[5]), "f"(a1[7]), "f"(a1[21]), "f"(a1[23]) : "memory");
                    {
                        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_2 * 64 + (long long)rpos])), "f"(rk[5]), "f"(rk[7]) : "memory");
                    }
                }
                int key_3 = keys[4];
                if (key_3 >= 0) {
                    long long base_4 = (long long)key_3 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_4])), "f"(a0[8]), "f"(a0[10]), "f"(a0[24]), "f"(a0[26]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_4 + 128])), "f"(a1[8]), "f"(a1[10]), "f"(a1[24]), "f"(a1[26]) : "memory");
                    {
                        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_3 * 64 + (long long)rpos])), "f"(rk[8]), "f"(rk[10]) : "memory");
                    }
                }
                int key_4 = keys[5];
                if (key_4 >= 0) {
                    long long base_5 = (long long)key_4 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_5])), "f"(a0[9]), "f"(a0[11]), "f"(a0[25]), "f"(a0[27]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_5 + 128])), "f"(a1[9]), "f"(a1[11]), "f"(a1[25]), "f"(a1[27]) : "memory");
                    {
                        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_4 * 64 + (long long)rpos])), "f"(rk[9]), "f"(rk[11]) : "memory");
                    }
                }
                int key_5 = keys[6];
                if (key_5 >= 0) {
                    long long base_6 = (long long)key_5 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_6])), "f"(a0[12]), "f"(a0[14]), "f"(a0[28]), "f"(a0[30]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_6 + 128])), "f"(a1[12]), "f"(a1[14]), "f"(a1[28]), "f"(a1[30]) : "memory");
                    {
                        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_5 * 64 + (long long)rpos])), "f"(rk[12]), "f"(rk[14]) : "memory");
                    }
                }
                int key_6 = keys[7];
                if (key_6 >= 0) {
                    long long base_7 = (long long)key_6 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_7])), "f"(a0[13]), "f"(a0[15]), "f"(a0[29]), "f"(a0[31]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base_7 + 128])), "f"(a1[13]), "f"(a1[15]), "f"(a1[29]), "f"(a1[31]) : "memory");
                    {
                        asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" :: "l"(reinterpret_cast<uint64_t>(&dkr_f32[(long long)key_6 * 64 + (long long)rpos])), "f"(rk[13]), "f"(rk[15]) : "memory");
                    }
                }
                mbarrier_wait(dkv_b_full_addr, par_1);
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a0[0])), "=r"(*reinterpret_cast<uint32_t*>(&a0[1])), "=r"(*reinterpret_cast<uint32_t*>(&a0[2])), "=r"(*reinterpret_cast<uint32_t*>(&a0[3])), "=r"(*reinterpret_cast<uint32_t*>(&a0[4])), "=r"(*reinterpret_cast<uint32_t*>(&a0[5])), "=r"(*reinterpret_cast<uint32_t*>(&a0[6])), "=r"(*reinterpret_cast<uint32_t*>(&a0[7])), "=r"(*reinterpret_cast<uint32_t*>(&a0[8])), "=r"(*reinterpret_cast<uint32_t*>(&a0[9])), "=r"(*reinterpret_cast<uint32_t*>(&a0[10])), "=r"(*reinterpret_cast<uint32_t*>(&a0[11])), "=r"(*reinterpret_cast<uint32_t*>(&a0[12])), "=r"(*reinterpret_cast<uint32_t*>(&a0[13])), "=r"(*reinterpret_cast<uint32_t*>(&a0[14])), "=r"(*reinterpret_cast<uint32_t*>(&a0[15]))
                    : "r"(tmem_tmem + 64 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[15]))
                    : "r"(tmem_tmem + 64 + wg * 32 + 1048576));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a1[0])), "=r"(*reinterpret_cast<uint32_t*>(&a1[1])), "=r"(*reinterpret_cast<uint32_t*>(&a1[2])), "=r"(*reinterpret_cast<uint32_t*>(&a1[3])), "=r"(*reinterpret_cast<uint32_t*>(&a1[4])), "=r"(*reinterpret_cast<uint32_t*>(&a1[5])), "=r"(*reinterpret_cast<uint32_t*>(&a1[6])), "=r"(*reinterpret_cast<uint32_t*>(&a1[7])), "=r"(*reinterpret_cast<uint32_t*>(&a1[8])), "=r"(*reinterpret_cast<uint32_t*>(&a1[9])), "=r"(*reinterpret_cast<uint32_t*>(&a1[10])), "=r"(*reinterpret_cast<uint32_t*>(&a1[11])), "=r"(*reinterpret_cast<uint32_t*>(&a1[12])), "=r"(*reinterpret_cast<uint32_t*>(&a1[13])), "=r"(*reinterpret_cast<uint32_t*>(&a1[14])), "=r"(*reinterpret_cast<uint32_t*>(&a1[15]))
                    : "r"(tmem_tmem + 128 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[15]))
                    : "r"(tmem_tmem + 128 + wg * 32 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dkv_b_drained_addr);
                int key2 = keys[0];
                if (key2 >= 0) {
                    long long base2 = (long long)key2 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2 + 256])), "f"(a0[0]), "f"(a0[2]), "f"(a0[16]), "f"(a0[18]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2 + 384])), "f"(a1[0]), "f"(a1[2]), "f"(a1[16]), "f"(a1[18]) : "memory");
                }
                int key2_7 = keys[1];
                if (key2_7 >= 0) {
                    long long base2_1 = (long long)key2_7 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_1 + 256])), "f"(a0[1]), "f"(a0[3]), "f"(a0[17]), "f"(a0[19]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_1 + 384])), "f"(a1[1]), "f"(a1[3]), "f"(a1[17]), "f"(a1[19]) : "memory");
                }
                int key2_8 = keys[2];
                if (key2_8 >= 0) {
                    long long base2_2 = (long long)key2_8 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_2 + 256])), "f"(a0[4]), "f"(a0[6]), "f"(a0[20]), "f"(a0[22]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_2 + 384])), "f"(a1[4]), "f"(a1[6]), "f"(a1[20]), "f"(a1[22]) : "memory");
                }
                int key2_9 = keys[3];
                if (key2_9 >= 0) {
                    long long base2_3 = (long long)key2_9 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_3 + 256])), "f"(a0[5]), "f"(a0[7]), "f"(a0[21]), "f"(a0[23]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_3 + 384])), "f"(a1[5]), "f"(a1[7]), "f"(a1[21]), "f"(a1[23]) : "memory");
                }
                int key2_10 = keys[4];
                if (key2_10 >= 0) {
                    long long base2_4 = (long long)key2_10 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_4 + 256])), "f"(a0[8]), "f"(a0[10]), "f"(a0[24]), "f"(a0[26]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_4 + 384])), "f"(a1[8]), "f"(a1[10]), "f"(a1[24]), "f"(a1[26]) : "memory");
                }
                int key2_11 = keys[5];
                if (key2_11 >= 0) {
                    long long base2_5 = (long long)key2_11 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_5 + 256])), "f"(a0[9]), "f"(a0[11]), "f"(a0[25]), "f"(a0[27]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_5 + 384])), "f"(a1[9]), "f"(a1[11]), "f"(a1[25]), "f"(a1[27]) : "memory");
                }
                int key2_12 = keys[6];
                if (key2_12 >= 0) {
                    long long base2_6 = (long long)key2_12 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_6 + 256])), "f"(a0[12]), "f"(a0[14]), "f"(a0[28]), "f"(a0[30]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_6 + 384])), "f"(a1[12]), "f"(a1[14]), "f"(a1[28]), "f"(a1[30]) : "memory");
                }
                int key2_13 = keys[7];
                if (key2_13 >= 0) {
                    long long base2_7 = (long long)key2_13 * 512 + (long long)lpos;
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_7 + 256])), "f"(a0[13]), "f"(a0[15]), "f"(a0[29]), "f"(a0[31]) : "memory");
                    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dkv_f32[base2_7 + 384])), "f"(a1[13]), "f"(a1[15]), "f"(a1[29]), "f"(a1[31]) : "memory");
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 16) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // mma_main
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_3 = blocks_word[0];
            mbarrier_wait(qdo_full_addr, 0);
            if (elect_sync()) {
                #pragma unroll 1
                for (int i_3 = 0; i_3 < blocks_3; i_3++) {
                    unsigned int par_2 = i_3 & 1;
                    if (i_3 == 0) {
                        mbarrier_wait(k_full_addr, 0);
                        int _mma_a_lo_0 = ((q_k_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_0 = ((k_k_addr) >> 4) & 0x3FFF;
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_tmem), "r"(0));
                        tcgen05_commit(s_full_addr);
                    }
                    mbarrier_wait(dp_free_addr, par_2 ^ 1);
                    int _mma_a_lo_1 = ((do_k_addr) >> 4) & 0x3FFF;
                    int _mma_b_lo_1 = ((v_k_addr) >> 4) & 0x3FFF;
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem + (1048576))), "r"(0));
                    tcgen05_commit(dp_full_addr);
                    mbarrier_wait(p_full_addr, par_2);
                    if (i_3 > 0) {
                        mbarrier_wait(dkv_b_drained_addr, par_2 ^ 1);
                    }
                    int _mma_a_lo_2 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 1024;
                    int _mma_b_lo_2 = (((p_k_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem + (64))), "r"(0));
                    int _mma_a_lo_3 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 1024;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_2), "r"((tmem_tmem + (128))), "r"(0));
                    mbarrier_wait(ds_full_addr, par_2);
                    int _mma_a_lo_4 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 1024;
                    int _mma_b_lo_4 = (((ds_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_tmem + (192))), "r"(((i_3 == 0) ? 0 : 1)));
                    int _mma_a_lo_5 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 1024;
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
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_4), "r"((tmem_tmem + (256))), "r"(((i_3 == 0) ? 0 : 1)));
                    int _mma_a_lo_6 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 1024;
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
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_4), "r"((tmem_tmem + (320))), "r"(((i_3 == 0) ? 0 : 1)));
                    int _mma_a_lo_7 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 1024;
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
                    "mov.b32 id, 135365776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_4), "r"((tmem_tmem + (384))), "r"(((i_3 == 0) ? 0 : 1)));
                    int _mma_a_lo_8 = (((kr_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 68256912;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_4), "r"((tmem_tmem + (448))), "r"(((i_3 == 0) ? 0 : 1)));
                    tcgen05_commit(k_free_addr);
                    int _mma_a_lo_9 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (0) * 1024;
                    int _mma_b_lo_9 = (((ds_k_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"((tmem_tmem + (64))), "r"(1));
                    int _mma_a_lo_10 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (1) * 1024;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_9), "r"((tmem_tmem + (128))), "r"(1));
                    tcgen05_commit(dkv_a_full_addr);
                    int _mma_a_lo_11 = (((qr_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 68191376;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_9), "r"(tmem_tmem), "r"(0));
                    tcgen05_commit(dkr_full_addr);
                    mbarrier_wait(dkv_a_drained_addr, par_2);
                    int _mma_a_lo_12 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 1024;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_12), "r"(_mma_b_lo_2), "r"((tmem_tmem + (64))), "r"(0));
                    int _mma_a_lo_13 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 1024;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_13), "r"(_mma_b_lo_2), "r"((tmem_tmem + (128))), "r"(0));
                    tcgen05_commit(p_free_addr);
                    {
                        int _mma_a_lo_14 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2) * 1024;
                        int _mma_b_lo_14 = (((ds_k_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_14), "r"(_mma_b_lo_14), "r"((tmem_tmem + (64))), "r"(1));
                        int _mma_a_lo_15 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (3) * 1024;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_15), "r"(_mma_b_lo_14), "r"((tmem_tmem + (128))), "r"(1));
                        tcgen05_commit(dkv_b_full_addr);
                        tcgen05_commit(ds_free_addr);
                    }
                    if (blocks_3 > i_3 + 1) {
                        mbarrier_wait(dkr_drained_addr, par_2);
                        mbarrier_wait(k_full_addr, par_2 ^ 1);
                        mbarrier_wait(s_free_addr, par_2);
                        int _mma_a_lo_16 = ((q_k_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_16 = ((k_k_addr) >> 4) & 0x3FFF;
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_16), "r"(_mma_b_lo_16), "r"(tmem_tmem), "r"(0));
                        tcgen05_commit(s_full_addr);
                    }
                }
                tcgen05_commit(dq_done_addr);
            }
        }
    }
    // ---- Role: load ----
    if (warp == 17) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // load_main
            int token_1 = token_base + token_step * blockIdx.x;
            int lane_0_2 = lane;
            if (elect_sync()) {
                tma_4d_gmem2smem(q_k_addr, q_latent, 0, 0, 0, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr, dout, 0, 0, 0, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 8192, q_latent, 0, 0, 1, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 8192, dout, 0, 0, 1, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 16384, q_latent, 0, 0, 2, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 16384, dout, 0, 0, 2, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 24576, q_latent, 0, 0, 3, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 24576, dout, 0, 0, 3, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 32768, q_latent, 0, 0, 4, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 32768, dout, 0, 0, 4, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 40960, q_latent, 0, 0, 5, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 40960, dout, 0, 0, 5, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 49152, q_latent, 0, 0, 6, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 49152, dout, 0, 0, 6, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 57344, q_latent, 0, 0, 7, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 57344, dout, 0, 0, 7, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 65536, q_rope, 0, 0, 0, token_1, qdo_full_addr);
                mbarrier_arrive_expect_tx(qdo_full_addr, 139264);
            }
            stats[lane_0_2] = lse[token_1 * 64 + lane_0_2] * 1.4426950408889634f;
            stats[32 + lane_0_2] = lse[token_1 * 64 + 32 + lane_0_2] * 1.4426950408889634f;
            stats[64 + lane_0_2] = delta[token_1 * 64 + lane_0_2];
            stats[96 + lane_0_2] = delta[token_1 * 64 + 32 + lane_0_2];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(stats_full_addr);
        }
    }
    // ---- Role: metadata ----
    if (warp == 18) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // metadata_main
            int token_2 = token_base + token_step * blockIdx.x;
            int lane_0_3 = lane;
            int active = topk;
            if (has_topk_length != 0) {
                int _max_0 = ((topk_length[token_2]) > (0) ? (topk_length[token_2]) : (0));
                int _min_0 = ((_max_0) < (topk) ? (_max_0) : (topk));
                active = _min_0;
            }
            long long row_base = (long long)indices_offset + (long long)token_2 * (long long)idx_stride;
            int aligned4 = (int)(((idx_stride | indices_offset) & 3) == 0);
            int last_valid = -1;
            for (int blk = (active + 127) / 128 - 1; blk >= 0; blk--) {
                int pos = blk * 128 + lane_0_3 * 4;
                int best = -1;
                if (pos < active) {
                    if (aligned4 != 0) {
                        int _vec_load_0[4];
                        {
                            const int4* _ivptr_0 = reinterpret_cast<const int4*>(indices + (row_base + (long long)pos) + 0);
                            int4 _ivld_0;
                            _ivld_0 = *_ivptr_0;
                            _vec_load_0[0 + 0] = _ivld_0.x;
                            _vec_load_0[0 + 1] = _ivld_0.y;
                            _vec_load_0[0 + 2] = _ivld_0.z;
                            _vec_load_0[0 + 3] = _ivld_0.w;
                        }
                        if (_vec_load_0[0] >= 0 && _vec_load_0[0] < num_kv && active > pos) {
                            best = pos;
                        }
                        if (_vec_load_0[1] >= 0 && _vec_load_0[1] < num_kv && active > pos + 1) {
                            best = pos + 1;
                        }
                        if (_vec_load_0[2] >= 0 && _vec_load_0[2] < num_kv && active > pos + 2) {
                            best = pos + 2;
                        }
                        if (_vec_load_0[3] >= 0 && _vec_load_0[3] < num_kv && active > pos + 3) {
                            best = pos + 3;
                        }
                    } else {
                        int sv = indices[row_base + (long long)pos];
                        if (sv >= 0 && sv < num_kv && active > pos) {
                            best = pos;
                        }
                        int sv_0 = indices[row_base + (long long)(pos + 1)];
                        if (sv_0 >= 0 && sv_0 < num_kv && active > pos + 1) {
                            best = pos + 1;
                        }
                        int sv_1 = indices[row_base + (long long)(pos + 2)];
                        if (sv_1 >= 0 && sv_1 < num_kv && active > pos + 2) {
                            best = pos + 2;
                        }
                        int sv_2 = indices[row_base + (long long)(pos + 3)];
                        if (sv_2 >= 0 && sv_2 < num_kv && active > pos + 3) {
                            best = pos + 3;
                        }
                    }
                }
                float _warp_reduce_0 = best;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                best = _warp_reduce_0;
                if (best >= 0) {
                    last_valid = best;
                    break;
                }
            }
            int _max_1 = (((last_valid + 64) / 64) > (1) ? ((last_valid + 64) / 64) : (1));
            int blocks_4 = _max_1;
            if (lane_0_3 == 0) {
                blocks_word[0] = blocks_4;
                mbarrier_arrive(blocks_ready_addr);
            }
            if (lane_0_3 < 8) {
                #pragma unroll 1
                for (int i_4 = 0; i_4 < blocks_4; i_4++) {
                    int n = blocks_4 - 1 - i_4;
                    int slot_3 = i_4 % 2;
                    unsigned int mask = 0;
                    int position = n * 64 + lane_0_3 * 8;
                    int values[8];
                    int whole_tile = (int)(active >= n * 64 + 64 && ((idx_stride | indices_offset) & 7) == 0);
                    if (whole_tile != 0) {
                        int _vec_load_1[8];
                        {
                            uint32_t _iv_1_0;
                            uint32_t _iv_1_1;
                            uint32_t _iv_1_2;
                            uint32_t _iv_1_3;
                            uint32_t _iv_1_4;
                            uint32_t _iv_1_5;
                            uint32_t _iv_1_6;
                            uint32_t _iv_1_7;
                            asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(indices + (row_base + (long long)position) + (0))) : "memory");
                            _vec_load_1[0 + 0] = (int32_t)_iv_1_0;
                            _vec_load_1[0 + 1] = (int32_t)_iv_1_1;
                            _vec_load_1[0 + 2] = (int32_t)_iv_1_2;
                            _vec_load_1[0 + 3] = (int32_t)_iv_1_3;
                            _vec_load_1[0 + 4] = (int32_t)_iv_1_4;
                            _vec_load_1[0 + 5] = (int32_t)_iv_1_5;
                            _vec_load_1[0 + 6] = (int32_t)_iv_1_6;
                            _vec_load_1[0 + 7] = (int32_t)_iv_1_7;
                        }
                        values[0] = _vec_load_1[0];
                        values[1] = _vec_load_1[1];
                        values[2] = _vec_load_1[2];
                        values[3] = _vec_load_1[3];
                        values[4] = _vec_load_1[4];
                        values[5] = _vec_load_1[5];
                        values[6] = _vec_load_1[6];
                        values[7] = _vec_load_1[7];
                    } else {
                        int _min_1 = ((position) < (topk - 1) ? (position) : (topk - 1));
                        int clamped = _min_1;
                        int value = indices[row_base + (long long)clamped];
                        if (active <= position) {
                            value = -1;
                        }
                        values[0] = value;
                        int _min_2 = ((position + 1) < (topk - 1) ? (position + 1) : (topk - 1));
                        int clamped_0 = _min_2;
                        int value_1 = indices[row_base + (long long)clamped_0];
                        if (active <= position + 1) {
                            value_1 = -1;
                        }
                        values[1] = value_1;
                        int _min_3 = ((position + 2) < (topk - 1) ? (position + 2) : (topk - 1));
                        int clamped_2 = _min_3;
                        int value_3 = indices[row_base + (long long)clamped_2];
                        if (active <= position + 2) {
                            value_3 = -1;
                        }
                        values[2] = value_3;
                        int _min_4 = ((position + 3) < (topk - 1) ? (position + 3) : (topk - 1));
                        int clamped_4 = _min_4;
                        int value_5 = indices[row_base + (long long)clamped_4];
                        if (active <= position + 3) {
                            value_5 = -1;
                        }
                        values[3] = value_5;
                        int _min_5 = ((position + 4) < (topk - 1) ? (position + 4) : (topk - 1));
                        int clamped_6 = _min_5;
                        int value_7 = indices[row_base + (long long)clamped_6];
                        if (active <= position + 4) {
                            value_7 = -1;
                        }
                        values[4] = value_7;
                        int _min_6 = ((position + 5) < (topk - 1) ? (position + 5) : (topk - 1));
                        int clamped_8 = _min_6;
                        int value_9 = indices[row_base + (long long)clamped_8];
                        if (active <= position + 5) {
                            value_9 = -1;
                        }
                        values[5] = value_9;
                        int _min_7 = ((position + 6) < (topk - 1) ? (position + 6) : (topk - 1));
                        int clamped_10 = _min_7;
                        int value_11 = indices[row_base + (long long)clamped_10];
                        if (active <= position + 6) {
                            value_11 = -1;
                        }
                        values[6] = value_11;
                        int _min_8 = ((position + 7) < (topk - 1) ? (position + 7) : (topk - 1));
                        int clamped_12 = _min_8;
                        int value_13 = indices[row_base + (long long)clamped_12];
                        if (active <= position + 7) {
                            value_13 = -1;
                        }
                        values[7] = value_13;
                    }
                    mbarrier_wait(idx_free_addr + (slot_3) * 8, i_4 / 2 & 1 ^ 1);
                    int ok = (int)(values[0] >= 0 && values[0] < num_kv);
                    if (ok != 0) {
                        mask = mask | 1;
                    } else {
                        values[0] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8] = values[0];
                    int ok_0 = (int)(values[1] >= 0 && values[1] < num_kv);
                    if (ok_0 != 0) {
                        mask = mask | 2;
                    } else {
                        values[1] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 1] = values[1];
                    int ok_1 = (int)(values[2] >= 0 && values[2] < num_kv);
                    if (ok_1 != 0) {
                        mask = mask | 4;
                    } else {
                        values[2] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 2] = values[2];
                    int ok_2 = (int)(values[3] >= 0 && values[3] < num_kv);
                    if (ok_2 != 0) {
                        mask = mask | 8;
                    } else {
                        values[3] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 3] = values[3];
                    int ok_3 = (int)(values[4] >= 0 && values[4] < num_kv);
                    if (ok_3 != 0) {
                        mask = mask | 16;
                    } else {
                        values[4] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 4] = values[4];
                    int ok_4 = (int)(values[5] >= 0 && values[5] < num_kv);
                    if (ok_4 != 0) {
                        mask = mask | 32;
                    } else {
                        values[5] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 5] = values[5];
                    int ok_5 = (int)(values[6] >= 0 && values[6] < num_kv);
                    if (ok_5 != 0) {
                        mask = mask | 64;
                    } else {
                        values[6] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 6] = values[6];
                    int ok_6 = (int)(values[7] >= 0 && values[7] < num_kv);
                    if (ok_6 != 0) {
                        mask = mask | 128;
                    } else {
                        values[7] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 7] = values[7];
                    validity8[slot_3 * 320 + 256 + lane_0_3] = mask;
                    mbarrier_arrive(idx_full_addr + (slot_3) * 8);
                }
            }
        }
    }
    // ---- Role: spare ----
    if (warp == 19) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        // idle — no tasks assigned
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 4) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
