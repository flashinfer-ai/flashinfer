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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 128
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 8
#define TMEM_TMEM_STATS0_OFFSET 16
#define TMEM_TMEM_STATS1_OFFSET 48
#define TMEM_TMEM_O0_OFFSET 80
#define TMEM_TMEM_O1_OFFSET 88
#define NUM_Q_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 4
#define NUM_PAGE_PIPE_STAGES 6
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_CORR0_OFF 1024
#define SMEM_SMEM_CORR0_STAGE_BYTES 64
#define SMEM_SMEM_CORR0_STRIDE 64
#define SMEM_SMEM_CORR1_OFF 1088
#define SMEM_SMEM_CORR1_STAGE_BYTES 64
#define SMEM_SMEM_CORR1_STRIDE 64
#define SMEM_SMEM_EXCH0_OFF 1152
#define SMEM_SMEM_EXCH0_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_STRIDE 256
#define SMEM_SMEM_EXCH1_OFF 1408
#define SMEM_SMEM_EXCH1_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_STRIDE 256
#define SMEM_SMEM_EXCH0_U32_OFF 1152
#define SMEM_SMEM_EXCH0_U32_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_U32_STRIDE 256
#define SMEM_SMEM_EXCH1_U32_OFF 1408
#define SMEM_SMEM_EXCH1_U32_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_U32_STRIDE 256
#define SMEM_SMEM_QT_OFF 1664
#define SMEM_SMEM_QT_STAGE_BYTES 2048
#define SMEM_SMEM_QT_STRIDE 2048
#define SMEM_SMEM_KV_OFF 6144
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 6144
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P0_OFF 137216
#define SMEM_SMEM_P0_STAGE_BYTES 2048
#define SMEM_SMEM_P0_STRIDE 2048
#define SMEM_SMEM_P1_OFF 141312
#define SMEM_SMEM_P1_STAGE_BYTES 2048
#define SMEM_SMEM_P1_STRIDE 2048
#define SMEM_SMEM_PAGE_OFFSETS_OFF 143360
#define SMEM_SMEM_PAGE_OFFSETS_STAGE_BYTES 192
#define SMEM_SMEM_PAGE_OFFSETS_STRIDE 192
#define SMEM_SMEM_FINAL_SCALE0_OFF 143616
#define SMEM_SMEM_FINAL_SCALE0_STAGE_BYTES 32
#define SMEM_SMEM_FINAL_SCALE0_STRIDE 32
#define SMEM_SMEM_FINAL_SCALE1_OFF 143648
#define SMEM_SMEM_FINAL_SCALE1_STAGE_BYTES 32
#define SMEM_SMEM_FINAL_SCALE1_STRIDE 32
#define SMEM_SMEM_FINAL_INV_SUM_OFF 143680
#define SMEM_SMEM_FINAL_INV_SUM_STAGE_BYTES 32
#define SMEM_SMEM_FINAL_INV_SUM_STRIDE 32
#define SMEM_WORK_TOKEN_WORDS_OFF 144496
#define SMEM_WORK_TOKEN_WORDS_STAGE_BYTES 256
#define SMEM_WORK_TOKEN_WORDS_STRIDE 256
#define SMEM_SMEM_FINAL_MAX_OFF 144752
#define SMEM_SMEM_FINAL_MAX_STAGE_BYTES 32
#define SMEM_SMEM_FINAL_MAX_STRIDE 32
#define SMEM_SMEM_FINAL_SUM_OFF 144784
#define SMEM_SMEM_FINAL_SUM_STAGE_BYTES 32
#define SMEM_SMEM_FINAL_SUM_STRIDE 32
#define SMEM_SMEM_MERGE_FLAG_OFF 144816
#define SMEM_SMEM_MERGE_FLAG_STAGE_BYTES 16
#define SMEM_SMEM_MERGE_FLAG_STRIDE 16
#define SMEM_SCHED_SEQ_LENS_OFF 145408
#define SMEM_SCHED_SEQ_LENS_STAGE_BYTES 4096
#define SMEM_SCHED_SEQ_LENS_STRIDE 4096
#define SMEM_TOTAL 149504
#define THREADS 512
#define BLOCK_N 128
#define HEAD_DIM 128
#define TILE_Q 8
#define PAGE_SIZE 16
#define NUM_KV_STAGES 4

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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_st_x4_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32"
        " [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
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


__device__ __forceinline__ void tcgen05_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x4_wait(float* dst, int addr) {
    tmem_ld_x4(dst, addr);
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_balanced_gqa_decode_db1624c38dc2f7ee3831(const __grid_constant__ CUtensorMap Qt, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O_ptr, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, float* __restrict__ partial_o, float* __restrict__ partial_stats, unsigned int* __restrict__ tile_counters, unsigned int* __restrict__ queue_counters, int max_pages_per_seq, float softmax_scale, int num_q_heads, int num_kv_heads, int group_ratio, int batch_size, int q_len, unsigned int max_items)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 64)
    #define s_full_0_addr (mbar_base + 96)
    #define s_full_1_addr (mbar_base + 104)
    #define s_empty_0_addr (mbar_base + 112)
    #define s_empty_1_addr (mbar_base + 120)
    #define p_full_0_addr (mbar_base + 128)
    #define p_full_1_addr (mbar_base + 136)
    #define p_empty_0_addr (mbar_base + 144)
    #define p_empty_1_addr (mbar_base + 152)
    #define corr_scale_0_addr (mbar_base + 160)
    #define corr_scale_1_addr (mbar_base + 168)
    #define corr_empty_0_addr (mbar_base + 176)
    #define corr_empty_1_addr (mbar_base + 184)
    #define stats_empty_addr (mbar_base + 192)
    #define o_ready_0_addr (mbar_base + 200)
    #define o_ready_1_addr (mbar_base + 208)
    #define o_empty_0_addr (mbar_base + 216)
    #define o_empty_1_addr (mbar_base + 224)
    #define tmem_dealloc_addr (mbar_base + 232)
    #define page_offsets_full_addr (mbar_base + 240)
    #define page_offsets_empty_addr (mbar_base + 288)
    #define work_full_addr (mbar_base + 336)
    #define work_empty_addr (mbar_base + 368)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* smem_corr0 = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_corr0_addr = smem + 1024;
    float* smem_corr1 = reinterpret_cast<float*>(smem_raw + 1088);
    const int smem_corr1_addr = smem + 1088;
    float* smem_exch0 = reinterpret_cast<float*>(smem_raw + 1152);
    const int smem_exch0_addr = smem + 1152;
    float* smem_exch1 = reinterpret_cast<float*>(smem_raw + 1408);
    const int smem_exch1_addr = smem + 1408;
    unsigned int* smem_exch0_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1152);
    const int smem_exch0_u32_addr = smem + 1152;
    unsigned int* smem_exch1_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1408);
    const int smem_exch1_u32_addr = smem + 1408;
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1664);
    const int smem_qt_addr = smem + 1664;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_kv_addr = smem + 6144;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_v_addr = smem + 6144;
    __nv_bfloat16* smem_p0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137216);
    const int smem_p0_addr = smem + 137216;
    __nv_bfloat16* smem_p1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 141312);
    const int smem_p1_addr = smem + 141312;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 143360);
    const int smem_page_offsets_addr = smem + 143360;
    float* smem_final_scale0 = reinterpret_cast<float*>(smem_raw + 143616);
    const int smem_final_scale0_addr = smem + 143616;
    float* smem_final_scale1 = reinterpret_cast<float*>(smem_raw + 143648);
    const int smem_final_scale1_addr = smem + 143648;
    float* smem_final_inv_sum = reinterpret_cast<float*>(smem_raw + 143680);
    const int smem_final_inv_sum_addr = smem + 143680;
    unsigned int* work_token_words = reinterpret_cast<unsigned int*>(smem_raw + 144496);
    const int work_token_words_addr = smem + 144496;
    float* smem_final_max = reinterpret_cast<float*>(smem_raw + 144752);
    const int smem_final_max_addr = smem + 144752;
    float* smem_final_sum = reinterpret_cast<float*>(smem_raw + 144784);
    const int smem_final_sum_addr = smem + 144784;
    unsigned int* smem_merge_flag = reinterpret_cast<unsigned int*>(smem_raw + 144816);
    const int smem_merge_flag_addr = smem + 144816;
    int* sched_seq_lens = reinterpret_cast<int*>(smem_raw + 145408);
    const int sched_seq_lens_addr = smem + 145408;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Qt))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");

    // Mbarrier init (26 pipeline groups, 0 ordered-sequence groups, 50 barriers)
    // Mbarriers at smem_raw[0..400)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // s_full_0: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // s_full_1: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // s_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            // s_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 120, 128);
            // p_full_0: 1 barriers, init_count=256
            mbarrier_init(smem + 128, 256);
            // p_full_1: 1 barriers, init_count=256
            mbarrier_init(smem + 136, 256);
            // p_empty_0: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // p_empty_1: 1 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            // corr_scale_0: 1 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            // corr_scale_1: 1 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            // corr_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            // corr_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 184, 128);
            // stats_empty: 1 barriers, init_count=4
            mbarrier_init(smem + 192, 4);
            // o_ready_0: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            // o_ready_1: 1 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            // o_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 216, 128);
            // o_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 224, 128);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 232, 128);
            // --- pipeline 'page_pipe' ---
            // page_offsets_full: 6 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            // page_offsets_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            // work_empty: 4 barriers, init_count=480
            mbarrier_init(smem + 368, 480);
            mbarrier_init(smem + 376, 480);
            mbarrier_init(smem + 384, 480);
            mbarrier_init(smem + 392, 480);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 400);
    if (warp == 0) {
        int _tmem_hold = smem + 400;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_s1 = taddr + 8;
    const int tmem_tmem_stats0 = taddr + 16;
    const int tmem_tmem_stats1 = taddr + 48;
    const int tmem_tmem_o0 = taddr + 80;
    const int tmem_tmem_o1 = taddr + 88;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int is_wg1 = ((warp >= 4) ? 1 : 0);
            const int tmem_row_base_v = warp % 4 * 32;
            int my_tmem_s_base = taddr + (unsigned int)(((is_wg1 != 0) ? 8 : 0));
            int my_tmem_stats = taddr + (unsigned int)(((is_wg1 != 0) ? 48 : 16)) + (unsigned int)(tmem_row_base_v << 16);
            const int warp_in_wg = warp % 4;
            const int wg_tid = warp_in_wg * 32 + lane;
            int col_pair = wg_tid % 4;
            int col_pair_base = col_pair * 2;
            float* my_exch_ptr = ((is_wg1 != 0) ? smem_exch1 : smem_exch0);
            unsigned int* my_exch_u32_ptr = ((is_wg1 != 0) ? smem_exch1_u32 : smem_exch0_u32);
            float* my_corr_ptr = ((is_wg1 != 0) ? smem_corr1 : smem_corr0);
            __nv_bfloat16* my_p_base = ((is_wg1 != 0) ? smem_p1 : smem_p0);
            float sv[8];
            float sv_lo[4];
            float sv_hi[4];
            unsigned int work_stage_s = 0;
            unsigned int _phase_work_full = 0;
            mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
            unsigned int base = work_stage_s * 16;
            unsigned int valid = work_token_words[base];
            unsigned int row_tile = work_token_words[base + 1];
            unsigned int batch = work_token_words[base + 2];
            unsigned int kv_head = work_token_words[base + 3];
            unsigned int block_begin = work_token_words[base + 4];
            unsigned int block_end = work_token_words[base + 5];
            unsigned int seqlen_row = work_token_words[base + 6];
            unsigned int n_chunks = work_token_words[base + 7];
            unsigned int slot_tile_base = work_token_words[base + 8];
            unsigned int counter_idx = work_token_words[base + 9];
            unsigned int chunk = work_token_words[base + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
            work_stage_s += 1;
            if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
            unsigned int valid_s = valid;
            int block_begin_s = (int)block_begin;
            int block_end_s = (int)block_end;
            int seqlen_kv = (int)seqlen_row;
            unsigned int _phase_stats_empty_0 = 1;
            unsigned int _phase_s_full_1_0 = 0;
            unsigned int _phase_s_full_0_0 = 0;
            unsigned int _phase_corr_empty_1_0 = 1;
            unsigned int _phase_corr_empty_0_0 = 1;
            unsigned int _phase_p_empty_1_0 = 1;
            unsigned int _phase_p_empty_0_0 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < max_items; _tile_iter_s++) {
                if (valid_s == 0) {
                    break;
                }
                mbarrier_wait(stats_empty_addr, _phase_stats_empty_0);
                _phase_stats_empty_0 ^= 1;
                int num_n_blocks_total = block_end_s - block_begin_s;
                int cta_n_blocks = num_n_blocks_total + num_n_blocks_total % 2;
                int split_start_block = block_begin_s;
                int num_pairs = cta_n_blocks / 2;
                float row_max_pair[2];
                float row_sum_pair[2];
                row_max_pair[0] = -CAKE_INF;
                row_max_pair[1] = -CAKE_INF;
                row_sum_pair[0] = 0.0f;
                row_sum_pair[1] = 0.0f;
                uint32_t _amf_u_0 = __float_as_uint(-3.4028235e+38f);
                uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
                if (wg_tid < 8) {
                    my_exch_u32_ptr[wg_tid] = _amf_enc_0;
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                if (is_wg1 != 0) {
                    mbarrier_wait(s_full_1_addr, _phase_s_full_1_0);
                    _phase_s_full_1_0 ^= 1;
                } else {
                    mbarrier_wait(s_full_0_addr, _phase_s_full_0_0);
                    _phase_s_full_0_0 ^= 1;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[3]))
                    : "r"(my_tmem_s_base));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[3]))
                    : "r"(my_tmem_s_base + 1048576));
                if (is_wg1 != 0) {
                    mbarrier_arrive(s_empty_1_addr);
                } else {
                    mbarrier_arrive(s_empty_0_addr);
                }
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    sv[c] = sv_lo[c];
                    sv[c + 4] = sv_hi[c];
                }
                #pragma unroll 1
                for (int pair = 0; pair < num_pairs; pair++) {
                    int my_block = split_start_block + cta_n_blocks - 1 - 2 * pair - is_wg1;
                    int ldtm_row_base = warp_in_wg * 32 + lane / 4;
                    int kv_pos0 = my_block * BLOCK_N + ldtm_row_base;
                    int kv_pos1 = kv_pos0 + 8;
                    int kv_pos2 = kv_pos0 + 16;
                    int kv_pos3 = kv_pos0 + 24;
                    if (seqlen_kv < (my_block + 1) * BLOCK_N) {
                        if (kv_pos0 >= seqlen_kv) {
                            sv[0] = -3.4028235e+38f;
                            sv[1] = -3.4028235e+38f;
                        }
                        if (kv_pos1 >= seqlen_kv) {
                            sv[2] = -3.4028235e+38f;
                            sv[3] = -3.4028235e+38f;
                        }
                        if (kv_pos2 >= seqlen_kv) {
                            sv[4] = -3.4028235e+38f;
                            sv[5] = -3.4028235e+38f;
                        }
                        if (kv_pos3 >= seqlen_kv) {
                            sv[6] = -3.4028235e+38f;
                            sv[7] = -3.4028235e+38f;
                        }
                    }
                    float pair_max[2];
                    float _max_5 = max_noftz(sv[0], sv[2]);
                    float _max_6 = max_noftz(sv[4], sv[6]);
                    float _max_7 = max_noftz(_max_5, _max_6);
                    pair_max[0] = _max_7;
                    float _max_8 = max_noftz(sv[1], sv[3]);
                    float _max_9 = max_noftz(sv[5], sv[7]);
                    float _max_10 = max_noftz(_max_8, _max_9);
                    pair_max[1] = _max_10;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 2; c_1++) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 16);
                        float _max_11 = max_noftz(pair_max[c_1], _shfl_xor_0);
                        pair_max[c_1] = _max_11;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 8);
                        float _max_12 = max_noftz(pair_max[c_1], _shfl_xor_1);
                        pair_max[c_1] = _max_12;
                    }
                    float old_max_pair[2];
                    float new_max_pair[2];
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 2; c_2++) {
                        old_max_pair[c_2] = row_max_pair[c_2];
                        float _max_13 = max_noftz(row_max_pair[c_2], pair_max[c_2]);
                        new_max_pair[c_2] = _max_13;
                    }
                    if (lane < 8) {
                        uint32_t _amf_u_1 = __float_as_uint(new_max_pair[0]);
                        uint32_t _amf_mask_1 = -int32_t(_amf_u_1 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_1 = _amf_u_1 ^ _amf_mask_1;
                        uint32_t _amf_u_2 = __float_as_uint(new_max_pair[1]);
                        uint32_t _amf_mask_2 = -int32_t(_amf_u_2 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_2 = _amf_u_2 ^ _amf_mask_2;
                        atomicMax(&my_exch_u32_ptr[col_pair_base], _amf_enc_1);
                        atomicMax(&my_exch_u32_ptr[col_pair_base + 1], _amf_enc_2);
                    }
                    if (is_wg1 != 0) {
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                    }
                    uint32_t _amf_u_3 = my_exch_u32_ptr[col_pair_base];
                    uint32_t _amf_mask_3 = ((_amf_u_3 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_3 ^ _amf_mask_3);
                    new_max_pair[0] = _amf_dec_0;
                    uint32_t _amf_u_4 = my_exch_u32_ptr[col_pair_base + 1];
                    uint32_t _amf_mask_4 = ((_amf_u_4 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_1 = __uint_as_float(_amf_u_4 ^ _amf_mask_4);
                    new_max_pair[1] = _amf_dec_1;
                    float acc_scale_pair[2];
                    #pragma unroll
                    for (int c_3 = 0; c_3 < 2; c_3++) {
                        float delta = softmax_scale * (row_max_pair[c_3] - new_max_pair[c_3]);
                        float _expf_0 = __expf(delta);
                        acc_scale_pair[c_3] = ((row_max_pair[c_3] > -CAKE_INF) ? _expf_0 : 1.0f);
                    }
                    float stats_pair[4];
                    stats_pair[0] = old_max_pair[0];
                    stats_pair[1] = old_max_pair[1];
                    stats_pair[2] = new_max_pair[0];
                    stats_pair[3] = new_max_pair[1];
                    if (is_wg1 != 0) {
                        mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                        _phase_corr_empty_1_0 ^= 1;
                    } else {
                        mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                        _phase_corr_empty_0_0 ^= 1;
                    }
                    tmem_st_x4_f32(my_tmem_stats, stats_pair);
                    float max_scaled_pair[2];
                    #pragma unroll
                    for (int c_4 = 0; c_4 < 2; c_4++) {
                        float safe_max = ((new_max_pair[c_4] == -CAKE_INF) ? 0.0f : new_max_pair[c_4]);
                        max_scaled_pair[c_4] = safe_max * softmax_scale;
                    }
                    float exp_vals[8];
                    #pragma unroll
                    for (int c_5 = 0; c_5 < 8; c_5++) {
                        float _expf_1 = __expf(sv[c_5] * softmax_scale - max_scaled_pair[c_5 % 2]);
                        exp_vals[c_5] = _expf_1;
                    }
                    #pragma unroll
                    for (int c_6 = 0; c_6 < 2; c_6++) {
                        row_max_pair[c_6] = new_max_pair[c_6];
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(corr_scale_1_addr);
                    } else {
                        mbarrier_arrive(corr_scale_0_addr);
                    }
                    float pair_sum[2];
                    pair_sum[0] = exp_vals[0] + exp_vals[2] + exp_vals[4] + exp_vals[6];
                    pair_sum[1] = exp_vals[1] + exp_vals[3] + exp_vals[5] + exp_vals[7];
                    #pragma unroll
                    for (int c_7 = 0; c_7 < 2; c_7++) {
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[c_7], 16);
                        pair_sum[c_7] = pair_sum[c_7] + _shfl_xor_2;
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[c_7], 8);
                        pair_sum[c_7] = pair_sum[c_7] + _shfl_xor_3;
                        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[c_7], 4);
                        pair_sum[c_7] = pair_sum[c_7] + _shfl_xor_4;
                    }
                    #pragma unroll
                    for (int c_8 = 0; c_8 < 2; c_8++) {
                        row_sum_pair[c_8] = row_sum_pair[c_8] * acc_scale_pair[c_8] + pair_sum[c_8];
                    }
                    unsigned int regs_p[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(exp_vals[_lp*2 + 0], exp_vals[_lp*2+1 + 0]));
                        regs_p[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (is_wg1 != 0) {
                        mbarrier_wait(p_empty_1_addr, _phase_p_empty_1_0);
                        _phase_p_empty_1_0 ^= 1;
                    } else {
                        mbarrier_wait(p_empty_0_addr, _phase_p_empty_0_0);
                        _phase_p_empty_0_0 ^= 1;
                    }
                    int slice_idx = warp_in_wg / 2;
                    int warp_idx_in_slice = warp_in_wg % 2;
                    int mtx_idx = lane / 8;
                    int thr_row_idx = lane % 8;
                    int mtx_col_idx = warp_idx_in_slice * 4 + mtx_idx;
                    int seg_col_idx = mtx_col_idx ^ thr_row_idx;
                    int stsm_offset = slice_idx * 8 * 128 + thr_row_idx * 128 + seg_col_idx * 16;
                    const void* _stmatrix_ptr_5 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(my_p_base) + stsm_offset);
                    uint64_t _stmatrix_addr64_5;
                    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_addr64_5) : "l"(_stmatrix_ptr_5));
                    uint32_t _stmatrix_addr_5;
                    asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_addr_5) : "l"(_stmatrix_addr64_5));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[3]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(p_full_1_addr);
                    } else {
                        mbarrier_arrive(p_full_0_addr);
                    }
                    if (pair < num_pairs - 1) {
                        if (is_wg1 != 0) {
                            mbarrier_wait(s_full_1_addr, _phase_s_full_1_0);
                            _phase_s_full_1_0 ^= 1;
                        } else {
                            mbarrier_wait(s_full_0_addr, _phase_s_full_0_0);
                            _phase_s_full_0_0 ^= 1;
                        }
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[3]))
                            : "r"(my_tmem_s_base));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[3]))
                            : "r"(my_tmem_s_base + 1048576));
                        if (is_wg1 != 0) {
                            mbarrier_arrive(s_empty_1_addr);
                        } else {
                            mbarrier_arrive(s_empty_0_addr);
                        }
                        #pragma unroll
                        for (int c_9 = 0; c_9 < 4; c_9++) {
                            sv[c_9] = sv_lo[c_9];
                            sv[c_9 + 4] = sv_hi[c_9];
                        }
                    }
                }
                if (is_wg1 != 0) {
                    mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                    _phase_corr_empty_1_0 ^= 1;
                } else {
                    mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                    _phase_corr_empty_0_0 ^= 1;
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                if (lane < 4) {
                    my_exch_ptr[warp_in_wg * 8 + col_pair_base] = row_sum_pair[0];
                    my_exch_ptr[warp_in_wg * 8 + col_pair_base + 1] = row_sum_pair[1];
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                if (wg_tid < 4) {
                    my_corr_ptr[col_pair_base] = my_exch_ptr[col_pair_base] + my_exch_ptr[8 + col_pair_base] + my_exch_ptr[16 + col_pair_base] + my_exch_ptr[24 + col_pair_base];
                    my_corr_ptr[col_pair_base + 1] = my_exch_ptr[col_pair_base + 1] + my_exch_ptr[8 + col_pair_base + 1] + my_exch_ptr[16 + col_pair_base + 1] + my_exch_ptr[24 + col_pair_base + 1];
                    my_exch_ptr[col_pair_base] = row_max_pair[0];
                    my_exch_ptr[col_pair_base + 1] = row_max_pair[1];
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                if (is_wg1 != 0) {
                    mbarrier_arrive(corr_scale_1_addr);
                } else {
                    mbarrier_arrive(corr_scale_0_addr);
                }
                mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
                unsigned int base_0 = work_stage_s * 16;
                unsigned int valid_1 = work_token_words[base_0];
                unsigned int row_tile_2 = work_token_words[base_0 + 1];
                unsigned int batch_3 = work_token_words[base_0 + 2];
                unsigned int kv_head_4 = work_token_words[base_0 + 3];
                unsigned int block_begin_5 = work_token_words[base_0 + 4];
                unsigned int block_end_6 = work_token_words[base_0 + 5];
                unsigned int seqlen_row_7 = work_token_words[base_0 + 6];
                unsigned int n_chunks_8 = work_token_words[base_0 + 7];
                unsigned int slot_tile_base_9 = work_token_words[base_0 + 8];
                unsigned int counter_idx_10 = work_token_words[base_0 + 9];
                unsigned int chunk_11 = work_token_words[base_0 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
                work_stage_s += 1;
                if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
                valid_s = valid_1;
                block_begin_s = (int)block_begin_5;
                block_end_s = (int)block_end_6;
                seqlen_kv = (int)seqlen_row_7;
            }
        }
    // ---- Role: correction ----
    } else if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            const int warp_in_wg_c = warp % 4;
            const int wg_tid_c = warp_in_wg_c * 32 + lane;
            int d_idx = warp % 4 * 32 + lane;
            int slot_stride = q_len * num_kv_heads;
            unsigned int work_stage_c = 0;
            unsigned int _phase_work_full_1 = 0;
            mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
            unsigned int base_1 = work_stage_c * 16;
            unsigned int valid_2 = work_token_words[base_1];
            unsigned int row_tile_1 = work_token_words[base_1 + 1];
            unsigned int batch_1 = work_token_words[base_1 + 2];
            unsigned int kv_head_1 = work_token_words[base_1 + 3];
            unsigned int block_begin_1 = work_token_words[base_1 + 4];
            unsigned int block_end_1 = work_token_words[base_1 + 5];
            unsigned int seqlen_row_1 = work_token_words[base_1 + 6];
            unsigned int n_chunks_1 = work_token_words[base_1 + 7];
            unsigned int slot_tile_base_1 = work_token_words[base_1 + 8];
            unsigned int counter_idx_1 = work_token_words[base_1 + 9];
            unsigned int chunk_1 = work_token_words[base_1 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
            work_stage_c += 1;
            if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
            unsigned int valid_c = valid_2;
            int row_tile_c = (int)row_tile_1;
            int kv_head_idx = (int)kv_head_1;
            int block_begin_c = (int)block_begin_1;
            int block_end_c = (int)block_end_1;
            int n_chunks_c = (int)n_chunks_1;
            int slot_tile_base_c = (int)slot_tile_base_1;
            int counter_idx_c = (int)counter_idx_1;
            int chunk_c = (int)chunk_1;
            unsigned int _phase_corr_scale_0_0 = 0;
            unsigned int _phase_corr_scale_1_0 = 0;
            unsigned int _phase_o_ready_0_0 = 0;
            unsigned int _phase_o_ready_1_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < max_items; _tile_iter_c++) {
                if (valid_c == 0) {
                    break;
                }
                int num_n_blocks_total_1 = block_end_c - block_begin_c;
                int my_slot = slot_tile_base_c + chunk_c * slot_stride;
                int partial_o_base = my_slot * 1024 + d_idx;
                int partial_stats_base = my_slot * 16;
                int cta_n_blocks_1 = num_n_blocks_total_1 + num_n_blocks_total_1 % 2;
                int num_pairs_1 = cta_n_blocks_1 / 2;
                mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                _phase_corr_scale_0_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_arrive(corr_empty_0_addr);
                mbarrier_arrive(p_full_0_addr);
                mbarrier_wait(corr_scale_1_addr, _phase_corr_scale_1_0);
                _phase_corr_scale_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_arrive(corr_empty_1_addr);
                mbarrier_arrive(p_full_1_addr);
                #pragma unroll 1
                for (int pair_1 = 1; pair_1 < num_pairs_1; pair_1++) {
                    mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                    _phase_corr_scale_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[4];
                    tmem_ld_x4(&_tmem_load_0[0], taddr + 16 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float acc0_pair[2];
                    #pragma unroll
                    for (int c_10 = 0; c_10 < 2; c_10++) {
                        float max_diff0 = _tmem_load_0[c_10] - _tmem_load_0[c_10 + 2];
                        float scaled_diff0 = softmax_scale * max_diff0;
                        float _expf_2 = __expf(scaled_diff0);
                        acc0_pair[c_10] = ((max_diff0 != 0.0f) ? _expf_2 : 1.0f);
                    }
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_wait(o_ready_0_addr, _phase_o_ready_0_0);
                    _phase_o_ready_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int rescale_pred_0 = ((acc0_pair[0] != 1.0f) ? 1 : 0);
                    rescale_pred_0 = rescale_pred_0 | ((acc0_pair[1] != 1.0f) ? 1 : 0);
                    int _vote_1 = __any_sync(0xFFFFFFFF, rescale_pred_0 != 0);
                    if (_vote_1 != 0) {
                        float o0_lo[4];
                        float o0_hi[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[3]))
                            : "r"(taddr + 80));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[3]))
                            : "r"(taddr + 80 + 1048576));
                        float o0[8];
                        #pragma unroll
                        for (int h = 0; h < 4; h++) {
                            o0[h] = o0_lo[h];
                            o0[h + 4] = o0_hi[h];
                        }
                        #pragma unroll
                        for (int h_1 = 0; h_1 < 8; h_1++) {
                            o0[h_1] = o0[h_1] * acc0_pair[h_1 % 2];
                        }
                        #pragma unroll
                        for (int h_2 = 0; h_2 < 4; h_2++) {
                            o0_lo[h_2] = o0[h_2];
                            o0_hi[h_2] = o0[h_2 + 4];
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[3])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(o_empty_0_addr);
                    mbarrier_arrive(p_full_0_addr);
                    mbarrier_wait(corr_scale_1_addr, _phase_corr_scale_1_0);
                    _phase_corr_scale_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[4];
                    tmem_ld_x4(&_tmem_load_1[0], taddr + 48 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float acc1_pair[2];
                    #pragma unroll
                    for (int c_11 = 0; c_11 < 2; c_11++) {
                        float max_diff1 = _tmem_load_1[c_11] - _tmem_load_1[c_11 + 2];
                        float scaled_diff1 = softmax_scale * max_diff1;
                        float _expf_3 = __expf(scaled_diff1);
                        acc1_pair[c_11] = ((max_diff1 != 0.0f) ? _expf_3 : 1.0f);
                    }
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_wait(o_ready_1_addr, _phase_o_ready_1_0);
                    _phase_o_ready_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int rescale_pred_1 = ((acc1_pair[0] != 1.0f) ? 1 : 0);
                    rescale_pred_1 = rescale_pred_1 | ((acc1_pair[1] != 1.0f) ? 1 : 0);
                    int _vote_2 = __any_sync(0xFFFFFFFF, rescale_pred_1 != 0);
                    if (_vote_2 != 0) {
                        float o1_lo[4];
                        float o1_hi[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[3]))
                            : "r"(taddr + 88));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[3]))
                            : "r"(taddr + 88 + 1048576));
                        float o1[8];
                        #pragma unroll
                        for (int h_3 = 0; h_3 < 4; h_3++) {
                            o1[h_3] = o1_lo[h_3];
                            o1[h_3 + 4] = o1_hi[h_3];
                        }
                        #pragma unroll
                        for (int h_4 = 0; h_4 < 8; h_4++) {
                            o1[h_4] = o1[h_4] * acc1_pair[h_4 % 2];
                        }
                        #pragma unroll
                        for (int h_5 = 0; h_5 < 4; h_5++) {
                            o1_lo[h_5] = o1[h_5];
                            o1_hi[h_5] = o1[h_5 + 4];
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 88), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 88 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[3])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(o_empty_1_addr);
                    mbarrier_arrive(p_full_1_addr);
                }
                mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                _phase_corr_scale_0_0 ^= 1;
                mbarrier_wait(corr_scale_1_addr, _phase_corr_scale_1_0);
                _phase_corr_scale_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (wg_tid_c < 8) {
                    int stats_head = wg_tid_c;
                    float m0 = smem_exch0[stats_head];
                    float m1 = smem_exch1[stats_head];
                    float s0 = smem_corr0[stats_head];
                    float s1 = smem_corr1[stats_head];
                    float _max_14 = max_noftz(m0, m1);
                    float fm = _max_14;
                    float d0 = softmax_scale * (m0 - fm);
                    float d1 = softmax_scale * (m1 - fm);
                    float _expf_4 = __expf(d0);
                    float final_scale0 = ((m0 == -CAKE_INF) ? 0.0f : _expf_4);
                    float _expf_5 = __expf(d1);
                    float final_scale1 = ((m1 == -CAKE_INF) ? 0.0f : _expf_5);
                    float final_sum = s0 * final_scale0 + s1 * final_scale1;
                    smem_final_scale0[stats_head] = final_scale0;
                    smem_final_scale1[stats_head] = final_scale1;
                    smem_final_inv_sum[stats_head] = 1.0f / final_sum;
                    smem_final_max[stats_head] = fm;
                    smem_final_sum[stats_head] = final_sum;
                }
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                mbarrier_arrive(corr_empty_0_addr);
                mbarrier_arrive(corr_empty_1_addr);
                float lane_scale0 = 0.0f;
                float lane_scale1 = 0.0f;
                float lane_inv_sum = 0.0f;
                if (lane < 8) {
                    lane_scale0 = smem_final_scale0[lane];
                    lane_scale1 = smem_final_scale1[lane];
                    lane_inv_sum = smem_final_inv_sum[lane];
                }
                mbarrier_wait(o_ready_0_addr, _phase_o_ready_0_0);
                _phase_o_ready_0_0 ^= 1;
                mbarrier_wait(o_ready_1_addr, _phase_o_ready_1_0);
                _phase_o_ready_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_2[8];
                tmem_ld_x8(&_tmem_load_2[0], taddr + 80 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[8];
                tmem_ld_x8(&_tmem_load_3[0], taddr + 88 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int publish_split = ((n_chunks_c > 1) ? 1 : 0);
                if (publish_split == 0) {
                    #pragma unroll
                    for (int h_6 = 0; h_6 < 8; h_6++) {
                        float _shfl_11;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_11) : "f"(lane_scale0), "r"(h_6));
                        float scale0_h = _shfl_11;
                        float _shfl_12;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_12) : "f"(lane_scale1), "r"(h_6));
                        float scale1_h = _shfl_12;
                        float _shfl_13;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_13) : "f"(lane_inv_sum), "r"(h_6));
                        float inv_sum_h = _shfl_13;
                        float merged = _tmem_load_2[h_6] * scale0_h + _tmem_load_3[h_6] * scale1_h;
                        float final_o = merged * inv_sum_h;
                        if (h_6 < group_ratio) {
                            int q_head = kv_head_idx * group_ratio + h_6;
                            int o_idx = (row_tile_c * num_q_heads + q_head) * HEAD_DIM + d_idx;
                            *(reinterpret_cast<__nv_bfloat16*>(O_ptr + o_idx) + (0)) = __float2bfloat16_rn(final_o);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int h_7 = 0; h_7 < 8; h_7++) {
                        float _shfl_14;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_14) : "f"(lane_scale0), "r"(h_7));
                        float scale0_h_1 = _shfl_14;
                        float _shfl_15;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_15) : "f"(lane_scale1), "r"(h_7));
                        float scale1_h_1 = _shfl_15;
                        float merged_p = _tmem_load_2[h_7] * scale0_h_1 + _tmem_load_3[h_7] * scale1_h_1;
                        *(reinterpret_cast<float*>(partial_o + (partial_o_base + h_7 * HEAD_DIM)) + (0)) = merged_p;
                    }
                    if (wg_tid_c < 8) {
                        *(reinterpret_cast<float*>(partial_stats + (partial_stats_base + wg_tid_c)) + (0)) = smem_final_max[wg_tid_c];
                        *(reinterpret_cast<float*>(partial_stats + (partial_stats_base + 8 + wg_tid_c)) + (0)) = smem_final_sum[wg_tid_c];
                    }
                }
                mbarrier_arrive(o_empty_0_addr);
                mbarrier_arrive(o_empty_1_addr);
                if (elect_sync()) {
                    mbarrier_arrive(stats_empty_addr);
                }
                if (publish_split != 0) {
                    __threadfence();
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
                    if (wg_tid_c == 0) {
                        unsigned int _atomic_old_1 = atomicAdd(&tile_counters[counter_idx_c], 1);
                        unsigned int old_count = _atomic_old_1;
                        smem_merge_flag[0] = (((int)old_count + 1 == n_chunks_c) ? 1 : 0);
                    }
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
                    unsigned int merge_flag = smem_merge_flag[0];
                    if (merge_flag != 0) {
                        __threadfence();
                        float merge_max = -CAKE_INF;
                        #pragma unroll 1
                        for (int c_a = 0; c_a < n_chunks_c; c_a++) {
                            if (lane < 8) {
                                int slot_a = slot_tile_base_c + c_a * slot_stride;
                                float m_a = partial_stats[slot_a * 16 + lane];
                                float _max_15 = max_noftz(merge_max, m_a);
                                merge_max = _max_15;
                            }
                        }
                        float merge_acc[8];
                        #pragma unroll
                        for (int h_8 = 0; h_8 < 8; h_8++) {
                            merge_acc[h_8] = 0.0f;
                        }
                        float merge_den = 0.0f;
                        #pragma unroll 1
                        for (int c_b = 0; c_b < n_chunks_c; c_b++) {
                            int slot_b = slot_tile_base_c + c_b * slot_stride;
                            float w_lane = 0.0f;
                            if (lane < 8) {
                                float m_b = partial_stats[slot_b * 16 + lane];
                                float l_b = partial_stats[slot_b * 16 + 8 + lane];
                                if (m_b > -CAKE_INF) {
                                    float _expf_6 = __expf(softmax_scale * (m_b - merge_max));
                                    w_lane = _expf_6;
                                }
                                merge_den = merge_den + w_lane * l_b;
                            }
                            int o_base_b = slot_b * 1024 + d_idx;
                            #pragma unroll
                            for (int h_9 = 0; h_9 < 8; h_9++) {
                                float _shfl_16;
                                asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_16) : "f"(w_lane), "r"(h_9));
                                float w_h = _shfl_16;
                                float o_b = partial_o[o_base_b + h_9 * HEAD_DIM];
                                float _fma_0 = __fmaf_rn(o_b, w_h, merge_acc[h_9]);
                                merge_acc[h_9] = _fma_0;
                            }
                        }
                        float inv_den = 0.0f;
                        if (lane < 8) {
                            inv_den = 1.0f / merge_den;
                        }
                        #pragma unroll
                        for (int h_10 = 0; h_10 < 8; h_10++) {
                            float _shfl_17;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_17) : "f"(inv_den), "r"(h_10));
                            float inv_h = _shfl_17;
                            if (h_10 < group_ratio) {
                                int q_head_m = kv_head_idx * group_ratio + h_10;
                                int o_idx_m = (row_tile_c * num_q_heads + q_head_m) * HEAD_DIM + d_idx;
                                *(reinterpret_cast<__nv_bfloat16*>(O_ptr + o_idx_m) + (0)) = __float2bfloat16_rn(merge_acc[h_10] * inv_h);
                            }
                        }
                        if (wg_tid_c == 0) {
                            *(reinterpret_cast<unsigned int*>(tile_counters + counter_idx_c) + (0)) = 0;
                        }
                    }
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
                }
                mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
                unsigned int base_0_1 = work_stage_c * 16;
                unsigned int valid_1_1 = work_token_words[base_0_1];
                unsigned int row_tile_2_1 = work_token_words[base_0_1 + 1];
                unsigned int batch_3_1 = work_token_words[base_0_1 + 2];
                unsigned int kv_head_4_1 = work_token_words[base_0_1 + 3];
                unsigned int block_begin_5_1 = work_token_words[base_0_1 + 4];
                unsigned int block_end_6_1 = work_token_words[base_0_1 + 5];
                unsigned int seqlen_row_7_1 = work_token_words[base_0_1 + 6];
                unsigned int n_chunks_8_1 = work_token_words[base_0_1 + 7];
                unsigned int slot_tile_base_9_1 = work_token_words[base_0_1 + 8];
                unsigned int counter_idx_10_1 = work_token_words[base_0_1 + 9];
                unsigned int chunk_11_1 = work_token_words[base_0_1 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                work_stage_c += 1;
                if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
                valid_c = valid_1_1;
                row_tile_c = (int)row_tile_2_1;
                kv_head_idx = (int)kv_head_4_1;
                block_begin_c = (int)block_begin_5_1;
                block_end_c = (int)block_end_6_1;
                n_chunks_c = (int)n_chunks_8_1;
                slot_tile_base_c = (int)slot_tile_base_9_1;
                counter_idx_c = (int)counter_idx_10_1;
                chunk_c = (int)chunk_11_1;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    // ---- Role: mma_warp ----
    } else if (warp == 12) {
        { // mma_warp_main
            unsigned int work_stage_m = 0;
            unsigned int q_cons_stage = 0;
            unsigned int q_cons_phase = 0;
            unsigned int _phase_work_full_2 = 0;
            mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
            unsigned int base_2 = work_stage_m * 16;
            unsigned int valid_3 = work_token_words[base_2];
            unsigned int row_tile_3 = work_token_words[base_2 + 1];
            unsigned int batch_2 = work_token_words[base_2 + 2];
            unsigned int kv_head_2 = work_token_words[base_2 + 3];
            unsigned int block_begin_2 = work_token_words[base_2 + 4];
            unsigned int block_end_2 = work_token_words[base_2 + 5];
            unsigned int seqlen_row_2 = work_token_words[base_2 + 6];
            unsigned int n_chunks_2 = work_token_words[base_2 + 7];
            unsigned int slot_tile_base_2 = work_token_words[base_2 + 8];
            unsigned int counter_idx_2 = work_token_words[base_2 + 9];
            unsigned int chunk_2 = work_token_words[base_2 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
            work_stage_m += 1;
            if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
            unsigned int valid_m = valid_3;
            int block_begin_m = (int)block_begin_2;
            int block_end_m = (int)block_end_2;
            unsigned int _phase_s_empty_0_0 = 1;
            unsigned int _phase_s_empty_1_0 = 1;
            unsigned int _phase_o_empty_0_0 = 1;
            unsigned int _phase_p_full_0_0 = 0;
            unsigned int _phase_o_empty_1_0 = 1;
            unsigned int _phase_p_full_1_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < max_items; _tile_iter_m++) {
                if (valid_m == 0) {
                    break;
                }
                int num_n_blocks_total_2 = block_end_m - block_begin_m;
                int cta_n_blocks_2 = num_n_blocks_total_2 + num_n_blocks_total_2 % 2;
                int num_pairs_2 = cta_n_blocks_2 / 2;
                int inst0_stage = 0;
                int first_pv0 = 1;
                int first_pv1 = 1;
                mbarrier_wait(q_full_addr + (q_cons_stage) * 8, q_cons_phase);
                mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                _phase_s_empty_0_0 ^= 1;
                mbarrier_wait(kv_full_addr, 0);
                int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 2048);
                int _mma_b_lo_0 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 128);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                }
                elect_commit(s_full_0_addr);
                elect_commit(kv_empty_addr);
                mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                _phase_s_empty_1_0 ^= 1;
                mbarrier_wait(kv_full_addr + 8, 0);
                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (1) * 2048);
                int _mma_b_lo_1 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 128);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                    }
                }
                elect_commit(s_full_1_addr);
                elect_commit(kv_empty_addr + 8);
                #pragma unroll 1
                for (int pair_2 = 0; pair_2 < num_pairs_2 - 1; pair_2++) {
                    int s0_1 = inst0_stage;
                    int s1_1 = (inst0_stage + 1) % 4;
                    int s0_next = (inst0_stage + 2) % 4;
                    int s1_next = (inst0_stage + 3) % 4;
                    mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                    _phase_s_empty_0_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s0_next) * 8, 0);
                    int _mma_a_lo_2 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s0_next) * 2048);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 128);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                        }
                    }
                    elect_commit(s_full_0_addr);
                    elect_commit(kv_empty_addr + (s0_next) * 8);
                    mbarrier_wait(o_empty_0_addr, _phase_o_empty_0_0);
                    _phase_o_empty_0_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s0_1) * 8, 1);
                    mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                    _phase_p_full_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_1) * 2048);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                        uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, ((first_pv0) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                    }
                    elect_commit2(kv_empty_addr + (s0_1) * 8, o_ready_0_addr);
                    elect_commit(p_empty_0_addr);
                    mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                    _phase_s_empty_1_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s1_next) * 8, 0);
                    int _mma_a_lo_4 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1_next) * 2048);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 128);
                    {
                        uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134349968, 1);
                        }
                    }
                    elect_commit(s_full_1_addr);
                    elect_commit(kv_empty_addr + (s1_next) * 8);
                    mbarrier_wait(o_empty_1_addr, _phase_o_empty_1_0);
                    _phase_o_empty_1_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s1_1) * 8, 1);
                    mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                    _phase_p_full_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_1) * 2048);
                    int _mma_b_lo_5 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                        uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_5);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, ((first_pv1) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                        }
                    }
                    elect_commit2(kv_empty_addr + (s1_1) * 8, o_ready_1_addr);
                    elect_commit(p_empty_1_addr);
                    inst0_stage = s0_next;
                    first_pv0 = 0;
                    first_pv1 = 0;
                }
                elect_commit(q_empty_addr + (q_cons_stage) * 8);
                q_cons_stage += 1;
                if (q_cons_stage == 2) { q_cons_stage = 0; q_cons_phase ^= 1; }
                int s0_last = inst0_stage;
                int s1_last = (inst0_stage + 1) % 4;
                mbarrier_wait(o_empty_0_addr, _phase_o_empty_0_0);
                _phase_o_empty_0_0 ^= 1;
                mbarrier_wait(kv_full_addr + (s0_last) * 8, 1);
                mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                _phase_p_full_0_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_last) * 2048);
                int _mma_b_lo_6 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
                {
                    uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_6);
                    uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_6);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, ((first_pv0) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                    }
                }
                elect_commit2(kv_empty_addr + (s0_last) * 8, o_ready_0_addr);
                elect_commit(p_empty_0_addr);
                mbarrier_wait(o_empty_1_addr, _phase_o_empty_1_0);
                _phase_o_empty_1_0 ^= 1;
                mbarrier_wait(kv_full_addr + (s1_last) * 8, 1);
                mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                _phase_p_full_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_last) * 2048);
                int _mma_b_lo_7 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
                {
                    uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_7);
                    uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, ((first_pv1) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                    }
                }
                elect_commit2(kv_empty_addr + (s1_last) * 8, o_ready_1_addr);
                elect_commit(p_empty_1_addr);
                mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
                unsigned int base_0_2 = work_stage_m * 16;
                unsigned int valid_1_2 = work_token_words[base_0_2];
                unsigned int row_tile_2_2 = work_token_words[base_0_2 + 1];
                unsigned int batch_3_2 = work_token_words[base_0_2 + 2];
                unsigned int kv_head_4_2 = work_token_words[base_0_2 + 3];
                unsigned int block_begin_5_2 = work_token_words[base_0_2 + 4];
                unsigned int block_end_6_2 = work_token_words[base_0_2 + 5];
                unsigned int seqlen_row_7_2 = work_token_words[base_0_2 + 6];
                unsigned int n_chunks_8_2 = work_token_words[base_0_2 + 7];
                unsigned int slot_tile_base_9_2 = work_token_words[base_0_2 + 8];
                unsigned int counter_idx_10_2 = work_token_words[base_0_2 + 9];
                unsigned int chunk_11_2 = work_token_words[base_0_2 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                work_stage_m += 1;
                if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
                valid_m = valid_1_2;
                block_begin_m = (int)block_begin_5_2;
                block_end_m = (int)block_end_6_2;
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
        }
    // ---- Role: load_pgoff ----
    } else if (warp == 13) {
        { // load_pgoff_main
            unsigned int page_prod_stage = 0;
            unsigned int page_prod_phase = 1;
            unsigned int work_stage_p = 0;
            unsigned int _phase_work_full_3 = 0;
            mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
            unsigned int base_3 = work_stage_p * 16;
            unsigned int valid_4 = work_token_words[base_3];
            unsigned int row_tile_4 = work_token_words[base_3 + 1];
            unsigned int batch_4 = work_token_words[base_3 + 2];
            unsigned int kv_head_3 = work_token_words[base_3 + 3];
            unsigned int block_begin_3 = work_token_words[base_3 + 4];
            unsigned int block_end_3 = work_token_words[base_3 + 5];
            unsigned int seqlen_row_3 = work_token_words[base_3 + 6];
            unsigned int n_chunks_3 = work_token_words[base_3 + 7];
            unsigned int slot_tile_base_3 = work_token_words[base_3 + 8];
            unsigned int counter_idx_3 = work_token_words[base_3 + 9];
            unsigned int chunk_3 = work_token_words[base_3 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
            work_stage_p += 1;
            if (work_stage_p == 4) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
            unsigned int valid_p = valid_4;
            int batch_idx_p = (int)batch_4;
            int kv_head_idx_p = (int)kv_head_3;
            int block_begin_p = (int)block_begin_3;
            int block_end_p = (int)block_end_3;
            int seqlen_kv_p = (int)seqlen_row_3;
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < max_items; _tile_iter_p++) {
                if (valid_p == 0) {
                    break;
                }
                int num_n_blocks_p = block_end_p - block_begin_p;
                int cta_n_blocks_p = num_n_blocks_p + num_n_blocks_p % 2;
                int max_pg_p = (seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1;
                int pt_base_p = batch_idx_p * max_pages_per_seq;
                #pragma unroll 1
                for (int ni_p = 0; ni_p < cta_n_blocks_p; ni_p++) {
                    int n_block_p = block_begin_p + cta_n_blocks_p - 1 - ni_p;
                    int logical_page_base_p = n_block_p * 8;
                    mbarrier_wait(page_offsets_empty_addr + (page_prod_stage) * 8, page_prod_phase);
                    if (elect_sync()) {
                        int smem_page_base_p = page_prod_stage * 8;
                        if (max_pg_p >= logical_page_base_p + 7 && pt_base_p % 4 == 0) {
                            int _vec_load_0[4];
                            {
                                const int4* _ivptr_0 = reinterpret_cast<const int4*>(page_table + pt_base_p + logical_page_base_p);
                                int4 _ivld_0;
                                _ivld_0 = *_ivptr_0;
                                _vec_load_0[0 + 0] = _ivld_0.x;
                                _vec_load_0[0 + 1] = _ivld_0.y;
                                _vec_load_0[0 + 2] = _ivld_0.z;
                                _vec_load_0[0 + 3] = _ivld_0.w;
                            }
                            int _vec_load_1[4];
                            {
                                const int4* _ivptr_1 = reinterpret_cast<const int4*>(page_table + pt_base_p + logical_page_base_p + 4);
                                int4 _ivld_1;
                                _ivld_1 = *_ivptr_1;
                                _vec_load_1[0 + 0] = _ivld_1.x;
                                _vec_load_1[0 + 1] = _ivld_1.y;
                                _vec_load_1[0 + 2] = _ivld_1.z;
                                _vec_load_1[0 + 3] = _ivld_1.w;
                            }
                            #pragma unroll
                            for (int pg_i_p = 0; pg_i_p < 4; pg_i_p++) {
                                smem_page_offsets[smem_page_base_p + pg_i_p] = _vec_load_0[pg_i_p];
                                smem_page_offsets[smem_page_base_p + pg_i_p + 4] = _vec_load_1[pg_i_p];
                            }
                        } else {
                            #pragma unroll
                            for (int pg_i_p_1 = 0; pg_i_p_1 < 8; pg_i_p_1++) {
                                int safe_page_idx_p = logical_page_base_p + pg_i_p_1;
                                if (safe_page_idx_p > max_pg_p) {
                                    safe_page_idx_p = max_pg_p;
                                }
                                smem_page_offsets[smem_page_base_p + pg_i_p_1] = page_table[pt_base_p + safe_page_idx_p];
                            }
                        }
                        mbarrier_arrive(page_offsets_full_addr + (page_prod_stage) * 8);
                    }
                    page_prod_stage += 1;
                    if (page_prod_stage == 6) { page_prod_stage = 0; page_prod_phase ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
                unsigned int base_0_3 = work_stage_p * 16;
                unsigned int valid_1_3 = work_token_words[base_0_3];
                unsigned int row_tile_2_3 = work_token_words[base_0_3 + 1];
                unsigned int batch_3_3 = work_token_words[base_0_3 + 2];
                unsigned int kv_head_4_3 = work_token_words[base_0_3 + 3];
                unsigned int block_begin_5_3 = work_token_words[base_0_3 + 4];
                unsigned int block_end_6_3 = work_token_words[base_0_3 + 5];
                unsigned int seqlen_row_7_3 = work_token_words[base_0_3 + 6];
                unsigned int n_chunks_8_3 = work_token_words[base_0_3 + 7];
                unsigned int slot_tile_base_9_3 = work_token_words[base_0_3 + 8];
                unsigned int counter_idx_10_3 = work_token_words[base_0_3 + 9];
                unsigned int chunk_11_3 = work_token_words[base_0_3 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
                work_stage_p += 1;
                if (work_stage_p == 4) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
                valid_p = valid_1_3;
                batch_idx_p = (int)batch_3_3;
                kv_head_idx_p = (int)kv_head_4_3;
                block_begin_p = (int)block_begin_5_3;
                block_end_p = (int)block_end_6_3;
                seqlen_kv_p = (int)seqlen_row_7_3;
            }
        }
    // ---- Role: scheduler ----
    } else if (warp == 14) {
        { // scheduler_main
            int lane_0 = lane;
            int num_ctas = gridDim.x;
            int items_per_chunk = q_len * num_kv_heads;
            int num_groups = (batch_size + 32 - 1) / 32;
            #pragma unroll 4
            for (int gs = 0; gs < num_groups; gs++) {
                int bs = gs * 32 + lane_0;
                if (bs < batch_size) {
                    sched_seq_lens[bs] = seq_lens_kv[bs];
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            unsigned int total_pairs = 0;
            unsigned int p_max = 0;
            unsigned int p_min = 4294967295;
            #pragma unroll 1
            for (int g1 = 0; g1 < num_groups; g1++) {
                int b1 = g1 * 32 + lane_0;
                unsigned int pairs1 = 0;
                unsigned int pairs1_min = 4294967295;
                if (b1 < batch_size) {
                    int s1_2 = sched_seq_lens[b1] - (q_len - 1);
                    pairs1 = (unsigned int)((s1_2 + 255) / 256);
                    pairs1_min = pairs1;
                }
                unsigned int _warp_redux_u32_0;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(pairs1));
                total_pairs += _warp_redux_u32_0;
                unsigned int _warp_redux_u32_1;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(pairs1));
                unsigned int _max_0 = ((p_max) > (_warp_redux_u32_1) ? (p_max) : (_warp_redux_u32_1));
                p_max = _max_0;
                unsigned int _warp_redux_u32_2;
                asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"(pairs1_min));
                unsigned int _min_0 = ((p_min) < (_warp_redux_u32_2) ? (p_min) : (_warp_redux_u32_2));
                p_min = _min_0;
            }
            unsigned int total_work = total_pairs * (unsigned int)items_per_chunk;
            unsigned int ideal_pairs = (total_work + (unsigned int)num_ctas - 1) / (unsigned int)num_ctas;
            unsigned int balance_k = (ideal_pairs + 64 - 1) / 64;
            if (balance_k < 1) {
                balance_k = 1;
            }
            if (balance_k > 8) {
                balance_k = 8;
            }
            unsigned int chunk_divisor = balance_k * (unsigned int)num_ctas;
            unsigned int chunk_pairs_u = (total_work + chunk_divisor - 1) / chunk_divisor;
            if (chunk_pairs_u < 2) {
                chunk_pairs_u = 2;
            }
            if (chunk_pairs_u < p_max) {
                unsigned int split_ok = 0;
                if (p_max > ideal_pairs + chunk_pairs_u) {
                    split_ok = 1;
                }
                if (chunk_pairs_u < 2 * (p_max - p_min)) {
                    split_ok = 1;
                }
                if (split_ok == 0) {
                    chunk_pairs_u = p_max;
                }
            }
            int whole_items = batch_size * items_per_chunk;
            if (whole_items <= num_ctas) {
                int n_even = num_ctas / whole_items;
                if (n_even > 1) {
                    if (p_max >= 8 * (p_max - p_min)) {
                        unsigned int l_even = (p_max + (unsigned int)n_even - 1) / (unsigned int)n_even;
                        if (l_even < 2) {
                            l_even = 2;
                        }
                        if (l_even < p_max) {
                            chunk_pairs_u = l_even;
                        }
                    }
                }
            }
            unsigned int whole_u = (unsigned int)whole_items;
            unsigned int ctas_u = (unsigned int)num_ctas;
            if (whole_items > num_ctas) {
                if (p_max >= 8 * (p_max - p_min)) {
                    unsigned int k_max_div = 8 * ctas_u;
                    unsigned int l_min = (total_work + k_max_div - 1) / k_max_div;
                    if (l_min < 2) {
                        l_min = 2;
                    }
                    unsigned int cand = p_max;
                    if (lane_0 < 8) {
                        unsigned int k_div = (unsigned int)(lane_0 + 1) * ctas_u;
                        cand = (total_work + k_div - 1) / k_div;
                    }
                    if (lane_0 >= 8) {
                        if (lane_0 < 15) {
                            unsigned int n_div = (unsigned int)(lane_0 - 6);
                            cand = (p_max + n_div - 1) / n_div;
                        }
                    }
                    if (cand < 2) {
                        cand = 2;
                    }
                    if (cand < l_min) {
                        cand = p_max;
                    }
                    unsigned int full_per_tile = p_max / cand;
                    unsigned int remainder = p_max - full_per_tile * cand;
                    unsigned int unit = 4 * cand + 4;
                    unsigned int full_items = full_per_tile * whole_u;
                    unsigned int full_waves = full_items / ctas_u;
                    unsigned int n_last = full_items - full_waves * ctas_u;
                    unsigned int cost = full_waves * unit * ctas_u;
                    unsigned int rem_items = 0;
                    if (remainder > 0) {
                        rem_items = whole_u;
                    }
                    unsigned int rem_unit = 4 * remainder + 4;
                    unsigned int idle = ctas_u - n_last;
                    unsigned int absorbed = idle * (unit / rem_unit);
                    if (absorbed > rem_items) {
                        absorbed = rem_items;
                    }
                    unsigned int _min_1 = ((idle) < (rem_items) ? (idle) : (rem_items));
                    unsigned int active = n_last + _min_1;
                    if (n_last > 0) {
                        int _max_1 = ((96) > (active) ? (96) : (active));
                        cost += unit * (unsigned int)_max_1;
                        rem_items -= absorbed;
                    }
                    unsigned int rounds = (rem_items + ctas_u - 1) / ctas_u;
                    if (rem_items > 0) {
                        unsigned int _min_2 = ((rem_items) < (ctas_u) ? (rem_items) : (ctas_u));
                        int _max_2 = ((96) > (_min_2) ? (96) : (_min_2));
                        cost += rounds * rem_unit * (unsigned int)_max_2;
                    }
                    if (full_per_tile > 1) {
                        cost += 3 * ctas_u;
                    }
                    if (full_per_tile <= 1) {
                        if (remainder > 0) {
                            cost += 3 * ctas_u;
                        }
                    }
                    unsigned int cand_cost = cost;
                    if (cand >= p_max) {
                        cand_cost = 4294967295;
                    }
                    unsigned int full_per_tile_0 = p_max / p_max;
                    unsigned int remainder_1 = p_max - full_per_tile_0 * p_max;
                    unsigned int unit_2 = 4 * p_max + 4;
                    unsigned int full_items_3 = full_per_tile_0 * whole_u;
                    unsigned int full_waves_4 = full_items_3 / ctas_u;
                    unsigned int n_last_5 = full_items_3 - full_waves_4 * ctas_u;
                    unsigned int cost_6 = full_waves_4 * unit_2 * ctas_u;
                    unsigned int rem_items_7 = 0;
                    if (remainder_1 > 0) {
                        rem_items_7 = whole_u;
                    }
                    unsigned int rem_unit_8 = 4 * remainder_1 + 4;
                    unsigned int idle_9 = ctas_u - n_last_5;
                    unsigned int absorbed_10 = idle_9 * (unit_2 / rem_unit_8);
                    if (absorbed_10 > rem_items_7) {
                        absorbed_10 = rem_items_7;
                    }
                    unsigned int _min_3 = ((idle_9) < (rem_items_7) ? (idle_9) : (rem_items_7));
                    unsigned int active_11 = n_last_5 + _min_3;
                    if (n_last_5 > 0) {
                        int _max_3 = ((96) > (active_11) ? (96) : (active_11));
                        cost_6 += unit_2 * (unsigned int)_max_3;
                        rem_items_7 -= absorbed_10;
                    }
                    unsigned int rounds_12 = (rem_items_7 + ctas_u - 1) / ctas_u;
                    if (rem_items_7 > 0) {
                        unsigned int _min_4 = ((rem_items_7) < (ctas_u) ? (rem_items_7) : (ctas_u));
                        int _max_4 = ((96) > (_min_4) ? (96) : (_min_4));
                        cost_6 += rounds_12 * rem_unit_8 * (unsigned int)_max_4;
                    }
                    if (full_per_tile_0 > 1) {
                        cost_6 += 3 * ctas_u;
                    }
                    if (full_per_tile_0 <= 1) {
                        if (remainder_1 > 0) {
                            cost_6 += 3 * ctas_u;
                        }
                    }
                    unsigned int whole_cost = cost_6;
                    unsigned int _warp_redux_u32_3;
                    asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_3) : "r"(cand_cost));
                    unsigned int min_cost = _warp_redux_u32_3;
                    unsigned int pick = 0;
                    if (cand_cost == min_cost) {
                        pick = cand;
                    }
                    unsigned int _warp_redux_u32_4;
                    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_4) : "r"(pick));
                    unsigned int best_l = _warp_redux_u32_4;
                    chunk_pairs_u = p_max;
                    if (min_cost < whole_cost) {
                        if (whole_cost - min_cost >= whole_cost / 50) {
                            chunk_pairs_u = best_l;
                        }
                    }
                }
            }
            int chunk_pairs = (int)chunk_pairs_u;
            float _rcp_0 = approx_rcp((float)chunk_pairs_u);
            float chunk_rcp = _rcp_0;
            unsigned int n_full_chunks = 0;
            unsigned int n_chunks_total = 0;
            unsigned int rem_bucket_total[4];
            #pragma unroll
            for (int bi = 0; bi < 4; bi++) {
                rem_bucket_total[bi] = 0;
            }
            #pragma unroll 1
            for (int g2 = 0; g2 < num_groups; g2++) {
                int b2 = g2 * 32 + lane_0;
                unsigned int full2 = 0;
                unsigned int n2 = 0;
                unsigned int pack2 = 0;
                if (b2 < batch_size) {
                    int s2 = sched_seq_lens[b2] - (q_len - 1);
                    unsigned int pairs2 = (unsigned int)((s2 + 255) / 256);
                    unsigned int q = (unsigned int)((float)pairs2 * chunk_rcp);
                    if (q > 0) {
                        if (pairs2 < q * chunk_pairs_u) {
                            q = q - 1;
                        }
                    }
                    if (pairs2 >= (q + 1) * chunk_pairs_u) {
                        q = q + 1;
                    }
                    if (pairs2 > q * chunk_pairs_u) {
                        q = q + 1;
                    }
                    n2 = q;
                    full2 = n2;
                    if (pairs2 < n2 * chunk_pairs_u) {
                        full2 = n2 - 1;
                        unsigned int rem2 = pairs2 - full2 * chunk_pairs_u;
                        int rb2 = 1;
                        #pragma unroll
                        for (int _rb = 0; _rb < 2; _rb++) {
                            if (chunk_pairs_u > rem2 << (unsigned int)rb2) {
                                rb2 = rb2 + 1;
                            }
                        }
                        pack2 = (unsigned int)(1 << 8 * (rb2 - 1));
                    }
                }
                unsigned int _warp_redux_u32_5;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_5) : "r"(full2));
                n_full_chunks += _warp_redux_u32_5;
                unsigned int _warp_redux_u32_6;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_6) : "r"(n2));
                n_chunks_total += _warp_redux_u32_6;
                unsigned int _warp_redux_u32_7;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_7) : "r"(pack2));
                unsigned int pack_group = _warp_redux_u32_7;
                #pragma unroll
                for (int bi_1 = 1; bi_1 < 4; bi_1++) {
                    rem_bucket_total[bi_1] = rem_bucket_total[bi_1] + (pack_group >> (unsigned int)(8 * (bi_1 - 1)) & 255);
                }
            }
            unsigned int bucket_end[4];
            bucket_end[0] = n_full_chunks * (unsigned int)items_per_chunk;
            #pragma unroll
            for (int bi_2 = 1; bi_2 < 4; bi_2++) {
                bucket_end[bi_2] = bucket_end[bi_2 - 1] + rem_bucket_total[bi_2] * (unsigned int)items_per_chunk;
            }
            unsigned int total_items = n_chunks_total * (unsigned int)items_per_chunk;
            if (blockIdx.x == 0) {
                if (lane_0 == 0) {
                    *(reinterpret_cast<unsigned int*>(queue_counters + 2) + (0)) = chunk_pairs_u;
                    *(reinterpret_cast<unsigned int*>(queue_counters + 3) + (0)) = total_items;
                }
            }
            unsigned int work_stage_sched = 0;
            int cur_group_b[4];
            unsigned int before_b[4];
            unsigned int si_before_b[4];
            unsigned int st_before_b[4];
            #pragma unroll
            for (int bi_3 = 0; bi_3 < 4; bi_3++) {
                cur_group_b[bi_3] = 0;
                before_b[bi_3] = 0;
                si_before_b[bi_3] = 0;
                st_before_b[bi_3] = 0;
            }
            unsigned int first_claim = 1;
            unsigned int _phase_work_empty = 1;
            #pragma unroll 1
            for (unsigned int _claim = 0; _claim < max_items + 1; _claim++) {
                mbarrier_wait(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty);
                unsigned int ticket_lane0 = blockIdx.x;
                if (first_claim == 0) {
                    if (lane_0 == 0) {
                        unsigned int _atomic_old_0 = atomicAdd(&queue_counters[0], 1);
                        ticket_lane0 = _atomic_old_0 + (unsigned int)num_ctas;
                    }
                }
                first_claim = 0;
                unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, ticket_lane0, 0);
                unsigned int ticket = _shfl_0;
                unsigned int token_base = work_stage_sched * 16;
                unsigned int valid_tok = ((ticket < total_items) ? 1 : 0);
                if (valid_tok != 0) {
                    int bucket = 0;
                    unsigned int bucket_start = 0;
                    #pragma unroll
                    for (int bi_4 = 0; bi_4 < 3; bi_4++) {
                        if (ticket >= bucket_end[bi_4]) {
                            bucket = bi_4 + 1;
                            bucket_start = bucket_end[bi_4];
                        }
                    }
                    unsigned int local_items = ticket - bucket_start;
                    unsigned int local_chunk = local_items / (unsigned int)items_per_chunk;
                    int in_chunk = (int)(local_items - local_chunk * (unsigned int)items_per_chunk);
                    int cursor_group = 0;
                    unsigned int before = 0;
                    unsigned int si_before = 0;
                    unsigned int st_before = 0;
                    #pragma unroll
                    for (int bi_5 = 0; bi_5 < 4; bi_5++) {
                        if (bucket == bi_5) {
                            cursor_group = cur_group_b[bi_5];
                            before = before_b[bi_5];
                            si_before = si_before_b[bi_5];
                            st_before = st_before_b[bi_5];
                        }
                    }
                    int s3 = 0;
                    int pairs3 = 0;
                    int n3 = 0;
                    int fullc3 = 0;
                    unsigned int mine3 = 0;
                    unsigned int split_items3 = 0;
                    unsigned int split_tiles3 = 0;
                    unsigned int incl3 = 0;
                    unsigned int incl_si3 = 0;
                    unsigned int incl_st3 = 0;
                    int b3 = 0;
                    #pragma unroll 1
                    for (int _adv = 0; _adv < 32; _adv++) {
                        b3 = cursor_group * 32 + lane_0;
                        s3 = 0;
                        pairs3 = 0;
                        n3 = 0;
                        fullc3 = 0;
                        mine3 = 0;
                        split_items3 = 0;
                        split_tiles3 = 0;
                        if (b3 < batch_size) {
                            s3 = sched_seq_lens[b3];
                            pairs3 = (s3 - (q_len - 1) + 255) / 256;
                            unsigned int q_1 = (unsigned int)((float)(unsigned int)pairs3 * chunk_rcp);
                            if (q_1 > 0) {
                                if (q_1 * chunk_pairs_u > (unsigned int)pairs3) {
                                    q_1 = q_1 - 1;
                                }
                            }
                            if ((q_1 + 1) * chunk_pairs_u <= (unsigned int)pairs3) {
                                q_1 = q_1 + 1;
                            }
                            if (q_1 * chunk_pairs_u < (unsigned int)pairs3) {
                                q_1 = q_1 + 1;
                            }
                            n3 = (int)q_1;
                            fullc3 = n3;
                            int rem3 = 0;
                            if (pairs3 < n3 * chunk_pairs) {
                                fullc3 = n3 - 1;
                                rem3 = pairs3 - fullc3 * chunk_pairs;
                            }
                            if (bucket == 0) {
                                mine3 = (unsigned int)fullc3;
                            } else if (rem3 > 0) {
                                int rb3 = 1;
                                #pragma unroll
                                for (int _rb3 = 0; _rb3 < 2; _rb3++) {
                                    if (chunk_pairs > rem3 << rb3) {
                                        rb3 = rb3 + 1;
                                    }
                                }
                                if (rb3 == bucket) {
                                    mine3 = 1;
                                }
                            }
                            if (n3 > 1) {
                                split_items3 = (unsigned int)n3;
                                split_tiles3 = 1;
                            }
                        }
                        uint32_t _warp_scan_sum_u32_0 = mine3;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(16));
                        incl3 = _warp_scan_sum_u32_0;
                        uint32_t _warp_scan_sum_u32_1 = split_items3;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(16));
                        incl_si3 = _warp_scan_sum_u32_1;
                        uint32_t _warp_scan_sum_u32_2 = split_tiles3;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(16));
                        incl_st3 = _warp_scan_sum_u32_2;
                        unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, incl3, 31);
                        unsigned int group_total = _shfl_1;
                        if (local_chunk < before + group_total) {
                            break;
                        }
                        before = before + group_total;
                        unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, incl_si3, 31);
                        si_before = si_before + _shfl_2;
                        unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, incl_st3, 31);
                        st_before = st_before + _shfl_3;
                        cursor_group = cursor_group + 1;
                    }
                    #pragma unroll
                    for (int bi_6 = 0; bi_6 < 4; bi_6++) {
                        if (bucket == bi_6) {
                            cur_group_b[bi_6] = cursor_group;
                            before_b[bi_6] = before;
                            si_before_b[bi_6] = si_before;
                            st_before_b[bi_6] = st_before;
                        }
                    }
                    unsigned int in_group = local_chunk - before;
                    unsigned int excl3 = incl3 - mine3;
                    unsigned int excl_si3 = incl_si3 - split_items3;
                    unsigned int excl_st3 = incl_st3 - split_tiles3;
                    int hit3 = 0;
                    if (mine3 > 0) {
                        if (excl3 <= in_group) {
                            if (in_group < incl3) {
                                hit3 = 1;
                            }
                        }
                    }
                    unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, hit3 != 0);
                    unsigned int hit_mask = _vote_0;
                    int _ffs_0 = __ffs(hit_mask);
                    int hit_lane = _ffs_0 - 1;
                    int _shfl_4 = __shfl_sync(0xFFFFFFFF, b3, hit_lane);
                    int sel_batch = _shfl_4;
                    int _shfl_5 = __shfl_sync(0xFFFFFFFF, s3, hit_lane);
                    int sel_seqlen = _shfl_5;
                    int _shfl_6 = __shfl_sync(0xFFFFFFFF, n3, hit_lane);
                    int sel_n = _shfl_6;
                    int _shfl_7 = __shfl_sync(0xFFFFFFFF, fullc3, hit_lane);
                    int sel_full = _shfl_7;
                    unsigned int _shfl_8 = __shfl_sync(0xFFFFFFFF, excl3, hit_lane);
                    int sel_excl = (int)_shfl_8;
                    unsigned int _shfl_9 = __shfl_sync(0xFFFFFFFF, excl_si3, hit_lane);
                    int sel_excl_si = (int)_shfl_9;
                    unsigned int _shfl_10 = __shfl_sync(0xFFFFFFFF, excl_st3, hit_lane);
                    int sel_excl_st = (int)_shfl_10;
                    int sel_chunk_off = (int)in_group - sel_excl;
                    int chunk_idx = sel_full;
                    if (bucket == 0) {
                        chunk_idx = sel_chunk_off;
                    }
                    int q_row = in_chunk / num_kv_heads;
                    int kv_head_sel = in_chunk - q_row * num_kv_heads;
                    int seqlen_row_4 = sel_seqlen - (q_len - 1 - q_row);
                    int n_blocks_row = (seqlen_row_4 + BLOCK_N - 1) / BLOCK_N;
                    int block_begin_4 = 2 * chunk_idx * chunk_pairs;
                    int block_end_4 = 2 * (chunk_idx + 1) * chunk_pairs;
                    if (chunk_idx + 1 == sel_n) {
                        block_end_4 = n_blocks_row;
                    }
                    int slot_tile_base_4 = 0;
                    int counter_idx_4 = 0;
                    if (sel_n > 1) {
                        slot_tile_base_4 = ((int)si_before + sel_excl_si) * items_per_chunk + q_row * num_kv_heads + kv_head_sel;
                        counter_idx_4 = ((int)st_before + sel_excl_st) * items_per_chunk + q_row * num_kv_heads + kv_head_sel;
                    }
                    if (lane_0 == 0) {
                        work_token_words[token_base + 1] = (unsigned int)(sel_batch * q_len + q_row);
                        work_token_words[token_base + 2] = (unsigned int)sel_batch;
                        work_token_words[token_base + 3] = (unsigned int)kv_head_sel;
                        work_token_words[token_base + 4] = (unsigned int)block_begin_4;
                        work_token_words[token_base + 5] = (unsigned int)block_end_4;
                        work_token_words[token_base + 6] = (unsigned int)seqlen_row_4;
                        work_token_words[token_base + 7] = (unsigned int)sel_n;
                        work_token_words[token_base + 8] = (unsigned int)slot_tile_base_4;
                        work_token_words[token_base + 9] = (unsigned int)counter_idx_4;
                        work_token_words[token_base + 10] = (unsigned int)chunk_idx;
                    }
                }
                if (lane_0 == 0) {
                    work_token_words[token_base] = valid_tok;
                    mbarrier_arrive(work_full_addr + (work_stage_sched) * 8);
                }
                work_stage_sched += 1;
                if (work_stage_sched == 4) { work_stage_sched = 0; _phase_work_empty ^= 1; }
                if (valid_tok == 0) {
                    break;
                }
            }
            if (lane_0 == 0) {
                uint32_t _atomic_inc_old_0;
                asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                    : "=r"(_atomic_inc_old_0) : "l"(&queue_counters[1]), "r"(static_cast<uint32_t>(num_ctas - 1)) : "memory");
                unsigned int done_old = _atomic_inc_old_0;
                if ((int)done_old == num_ctas - 1) {
                    *(reinterpret_cast<unsigned int*>(queue_counters) + (0)) = 0;
                }
            }
        }
    // ---- Role: load_warp ----
    } else if (warp == 15) {
        { // load_warp_main
            unsigned int q_prod_stage = 0;
            unsigned int q_prod_phase = 1;
            unsigned int page_cons_stage = 0;
            unsigned int page_cons_phase = 0;
            unsigned int work_stage_l = 0;
            {
                int spec_tiles = batch_size * q_len * num_kv_heads;
                int spec_id = blockIdx.x;
                if (spec_id < spec_tiles) {
                    if (elect_sync()) {
                        int spec_row_tile = spec_id / num_kv_heads;
                        int spec_head = spec_id - spec_row_tile * num_kv_heads;
                        int spec_batch = spec_row_tile / q_len;
                        int spec_row = spec_row_tile - spec_batch * q_len;
                        int spec_seqlen = seq_lens_kv[spec_batch] - (q_len - 1 - spec_row);
                        int spec_blocks = (spec_seqlen + BLOCK_N - 1) / BLOCK_N;
                        int spec_max_pg = (spec_seqlen + PAGE_SIZE - 1) / PAGE_SIZE - 1;
                        int spec_pt_base = spec_batch * max_pages_per_seq;
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&Qt))), "r"((int)(0)), "r"((int)((spec_row_tile * num_kv_heads + spec_head) * TILE_Q)), "r"((int)(0)) : "memory");
                        #pragma unroll
                        for (int spec_nb = 0; spec_nb < 2; spec_nb++) {
                            int spec_block = spec_blocks - 1 - spec_nb;
                            if (spec_block >= 0) {
                                #pragma unroll
                                for (int spec_pg = 0; spec_pg < 8; spec_pg++) {
                                    int spec_page_idx = spec_block * 8 + spec_pg;
                                    if (spec_page_idx > spec_max_pg) {
                                        spec_page_idx = spec_max_pg;
                                    }
                                    int spec_page = page_table[spec_pt_base + spec_page_idx] * num_kv_heads + spec_head;
                                    #pragma unroll
                                    for (int spec_hg = 0; spec_hg < 2; spec_hg++) {
                                        asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&K))), "r"((int)(0)), "r"((int)(0)), "r"((int)(spec_hg)), "r"((int)(spec_page)) : "memory");
                                    }
                                }
                            }
                        }
                    }
                }
            }
            unsigned int _phase_work_full_4 = 0;
            mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_4);
            unsigned int base_4 = work_stage_l * 16;
            unsigned int valid_5 = work_token_words[base_4];
            unsigned int row_tile_5 = work_token_words[base_4 + 1];
            unsigned int batch_5 = work_token_words[base_4 + 2];
            unsigned int kv_head_5 = work_token_words[base_4 + 3];
            unsigned int block_begin_6 = work_token_words[base_4 + 4];
            unsigned int block_end_5 = work_token_words[base_4 + 5];
            unsigned int seqlen_row_5 = work_token_words[base_4 + 6];
            unsigned int n_chunks_4 = work_token_words[base_4 + 7];
            unsigned int slot_tile_base_5 = work_token_words[base_4 + 8];
            unsigned int counter_idx_5 = work_token_words[base_4 + 9];
            unsigned int chunk_4 = work_token_words[base_4 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
            work_stage_l += 1;
            if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
            unsigned int valid_l = valid_5;
            int row_tile_l = (int)row_tile_5;
            int kv_head_idx_1 = (int)kv_head_5;
            int block_begin_l = (int)block_begin_6;
            int block_end_l = (int)block_end_5;
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < max_items; _tile_iter_l++) {
                if (valid_l == 0) {
                    break;
                }
                int num_n_blocks_total_3 = block_end_l - block_begin_l;
                int cta_n_blocks_3 = num_n_blocks_total_3 + num_n_blocks_total_3 % 2;
                mbarrier_wait(q_empty_addr + (q_prod_stage) * 8, q_prod_phase);
                if (elect_sync()) {
                    int off_qt = (row_tile_l * num_kv_heads + kv_head_idx_1) * TILE_Q;
                    mbarrier_arrive_expect_tx(q_full_addr + (q_prod_stage) * 8, TILE_Q * HEAD_DIM * 2);
                    tma_3d_gmem2smem(smem_qt_addr + q_prod_stage * 2048, (&Qt), 0, off_qt, 0, q_full_addr + (q_prod_stage) * 8);
                    int kv_stage = 0;
                    int kv_phase = 1;
                    int prefill = ((cta_n_blocks_3 < 2) ? cta_n_blocks_3 : 2);
                    #pragma unroll 1
                    for (int ni = 0; ni < prefill; ni++) {
                        int page_stage_unwrapped = page_cons_stage + (unsigned int)ni;
                        int page_stage = page_stage_unwrapped;
                        if (page_stage >= 6) {
                            page_stage = page_stage - 6;
                        }
                        int page_phase = page_cons_phase;
                        if (page_stage_unwrapped >= 6) {
                            page_phase = page_phase ^ 1;
                        }
                        int pg_base = page_stage * 8;
                        mbarrier_wait(page_offsets_full_addr + (page_stage) * 8, page_phase);
                        mbarrier_wait(kv_empty_addr + (kv_stage) * 8, kv_phase);
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                        int ldst = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                        int pg_k[8];
                        #pragma unroll
                        for (int pg_i = 0; pg_i < 8; pg_i++) {
                            pg_k[pg_i] = smem_page_offsets[pg_base + pg_i];
                        }
                        #pragma unroll
                        for (int pg_i_1 = 0; pg_i_1 < 8; pg_i_1++) {
                            int pg0 = pg_k[pg_i_1] * num_kv_heads + kv_head_idx_1;
                            #pragma unroll
                            for (int hg = 0; hg < 2; hg++) {
                                int toff = hg * 16384 + pg_i_1 * 2048;
                                tma_4d_gmem2smem(ldst + toff, (&K), 0, 0, hg, pg0, kv_full_addr + (kv_stage) * 8);
                            }
                        }
                        kv_stage += 1;
                        if (kv_stage == 4) { kv_stage = 0; kv_phase ^= 1; }
                    }
                    #pragma unroll 1
                    for (int ni_1 = 0; ni_1 < cta_n_blocks_3; ni_1++) {
                        int next_ni = ni_1 + 2;
                        if (next_ni < cta_n_blocks_3) {
                            int k_stage = next_ni % NUM_KV_STAGES;
                            int next_page_stage_unwrapped = page_cons_stage + 2;
                            int next_page_stage = next_page_stage_unwrapped;
                            if (next_page_stage >= 6) {
                                next_page_stage = next_page_stage - 6;
                            }
                            int next_page_phase = page_cons_phase;
                            if (next_page_stage_unwrapped >= 6) {
                                next_page_phase = next_page_phase ^ 1;
                            }
                            int npg_base = next_page_stage * 8;
                            mbarrier_wait(page_offsets_full_addr + (next_page_stage) * 8, next_page_phase);
                            mbarrier_wait(kv_empty_addr + (k_stage) * 8, 1);
                            mbarrier_arrive_expect_tx(kv_full_addr + (k_stage) * 8, 32768);
                            int kdst = smem_kv_addr + (unsigned int)(k_stage * 32768);
                            int pg_nk[8];
                            #pragma unroll
                            for (int pg_i_2 = 0; pg_i_2 < 8; pg_i_2++) {
                                pg_nk[pg_i_2] = smem_page_offsets[npg_base + pg_i_2];
                            }
                            #pragma unroll
                            for (int pg_i_3 = 0; pg_i_3 < 8; pg_i_3++) {
                                int npg0 = pg_nk[pg_i_3] * num_kv_heads + kv_head_idx_1;
                                #pragma unroll
                                for (int hg_1 = 0; hg_1 < 2; hg_1++) {
                                    int ntoff = hg_1 * 16384 + pg_i_3 * 2048;
                                    tma_4d_gmem2smem(kdst + ntoff, (&K), 0, 0, hg_1, npg0, kv_full_addr + (k_stage) * 8);
                                }
                            }
                        }
                        int stage = ni_1 % 4;
                        int vpg_base = page_cons_stage * 8;
                        mbarrier_wait(kv_empty_addr + (stage) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 32768);
                        int vdst = smem_kv_addr + (unsigned int)(stage * 32768);
                        int pg_v[8];
                        #pragma unroll
                        for (int pg_i_4 = 0; pg_i_4 < 8; pg_i_4++) {
                            pg_v[pg_i_4] = smem_page_offsets[vpg_base + pg_i_4];
                        }
                        #pragma unroll
                        for (int pg_i_5 = 0; pg_i_5 < 8; pg_i_5++) {
                            int vpg0 = pg_v[pg_i_5] * num_kv_heads + kv_head_idx_1;
                            #pragma unroll
                            for (int hg_2 = 0; hg_2 < 2; hg_2++) {
                                int vtoff = hg_2 * 16384 + pg_i_5 * 2048;
                                tma_4d_gmem2smem(vdst + vtoff, (&V), 0, 0, hg_2, vpg0, kv_full_addr + (stage) * 8);
                            }
                        }
                        mbarrier_arrive(page_offsets_empty_addr + (page_cons_stage) * 8);
                        page_cons_stage += 1;
                        if (page_cons_stage == 6) { page_cons_stage = 0; page_cons_phase ^= 1; }
                    }
                }
                q_prod_stage += 1;
                if (q_prod_stage == 2) { q_prod_stage = 0; q_prod_phase ^= 1; }
                mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_4);
                unsigned int base_0_4 = work_stage_l * 16;
                unsigned int valid_1_4 = work_token_words[base_0_4];
                unsigned int row_tile_2_4 = work_token_words[base_0_4 + 1];
                unsigned int batch_3_4 = work_token_words[base_0_4 + 2];
                unsigned int kv_head_4_4 = work_token_words[base_0_4 + 3];
                unsigned int block_begin_5_4 = work_token_words[base_0_4 + 4];
                unsigned int block_end_6_4 = work_token_words[base_0_4 + 5];
                unsigned int seqlen_row_7_4 = work_token_words[base_0_4 + 6];
                unsigned int n_chunks_8_4 = work_token_words[base_0_4 + 7];
                unsigned int slot_tile_base_9_4 = work_token_words[base_0_4 + 8];
                unsigned int counter_idx_10_4 = work_token_words[base_0_4 + 9];
                unsigned int chunk_11_4 = work_token_words[base_0_4 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                work_stage_l += 1;
                if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
                valid_l = valid_1_4;
                row_tile_l = (int)row_tile_2_4;
                kv_head_idx_1 = (int)kv_head_4_4;
                block_begin_l = (int)block_begin_5_4;
                block_end_l = (int)block_end_6_4;
            }
        }
    }

    // Cleanup

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
