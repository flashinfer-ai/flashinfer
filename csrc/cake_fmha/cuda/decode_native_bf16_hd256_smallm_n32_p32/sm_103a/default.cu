/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
struct __align__(128) CakeFmhaTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeFmhaTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeFmhaTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeFmhaTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeFmhaTensorMapPack { CakeFmhaTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeFmhaTensorMap) >= alignof(CUtensorMap), "CakeFmhaTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_FMHA_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O_HI_OFFSET 128
#define TMEM_TMEM_O_LO_OFFSET 192
#define NUM_KV_PIPE_STAGES 2
#define NUM_SM_PIPE_STAGES 2
#define SMEM_SMEM_ST_A_OFF 191488
#define SMEM_SMEM_ST_A_STAGE_BYTES 8256
#define SMEM_SMEM_ST_A_STRIDE 8256
#define SMEM_SMEM_ST_B_OFF 208000
#define SMEM_SMEM_ST_B_STAGE_BYTES 8256
#define SMEM_SMEM_ST_B_STRIDE 8256
#define SMEM_SMEM_SCALE_OFF 5120
#define SMEM_SMEM_SCALE_STAGE_BYTES 512
#define SMEM_SMEM_SCALE_STRIDE 512
#define SMEM_SMEM_SUM_OFF 2304
#define SMEM_SMEM_SUM_STAGE_BYTES 256
#define SMEM_SMEM_SUM_STRIDE 256
#define SMEM_SMEM_MAX_OFF 2560
#define SMEM_SMEM_MAX_STAGE_BYTES 256
#define SMEM_SMEM_MAX_STRIDE 256
#define SMEM_SMEM_RED_OFF 3072
#define SMEM_SMEM_RED_STAGE_BYTES 48
#define SMEM_SMEM_RED_STRIDE 48
#define SMEM_SMEM_W_OFF 4096
#define SMEM_SMEM_W_STAGE_BYTES 1024
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_SMEM_Q_HI_OFF 11264
#define SMEM_SMEM_Q_HI_STAGE_BYTES 8192
#define SMEM_SMEM_Q_HI_STRIDE 8192
#define SMEM_SMEM_Q_LO_OFF 27648
#define SMEM_SMEM_Q_LO_STAGE_BYTES 8192
#define SMEM_SMEM_Q_LO_STRIDE 8192
#define SMEM_SMEM_KV_HI_OFF 44032
#define SMEM_SMEM_KV_HI_STAGE_BYTES 32768
#define SMEM_SMEM_KV_HI_STRIDE 32768
#define SMEM_SMEM_KV_LO_OFF 109568
#define SMEM_SMEM_KV_LO_STAGE_BYTES 32768
#define SMEM_SMEM_KV_LO_STRIDE 32768
#define SMEM_SMEM_V_HI_OFF 44032
#define SMEM_SMEM_V_HI_STAGE_BYTES 32768
#define SMEM_SMEM_V_HI_STRIDE 32768
#define SMEM_SMEM_V_LO_OFF 109568
#define SMEM_SMEM_V_LO_STAGE_BYTES 32768
#define SMEM_SMEM_V_LO_STRIDE 32768
#define SMEM_SMEM_P_OFF 175104
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_TOTAL 224512
#define THREADS 512
#ifndef Q_LEN
#define Q_LEN 8
#endif
#ifndef GROUP
#define GROUP 8
#endif
#ifndef Q_BOX_ROWS
#define Q_BOX_ROWS 8
#endif
#ifndef NUM_SPLIT
#define NUM_SPLIT 148
#endif
#define BLOCK_N 128
#define HEAD_DIM 256
#define HEAD_DIM_HALF 128
#define NUM_KV_STAGES 2

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


__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_fmha_decode_native_bf16_hd256_smallm_n32_p32(CakeFmhaTensorMap const* Q, CakeFmhaTensorMap const* K, CakeFmhaTensorMap const* V, __nv_bfloat16* __restrict__ partial_O_ptr, float* __restrict__ partial_LSE_ptr, __nv_bfloat16* __restrict__ O_ptr, float* __restrict__ LSE_ptr, unsigned int* __restrict__ counters, int* __restrict__ page_table, int* __restrict__ seq_lens, int max_pages_per_seq, float softmax_scale_log2, int num_q_heads, int num_kv_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define kv_full_addr (mbar_base + 8)
    #define kv_empty_addr (mbar_base + 24)
    #define s_full_addr (mbar_base + 40)
    #define corr_scale_addr (mbar_base + 56)
    #define p_full_addr (mbar_base + 64)
    #define o_ready_addr (mbar_base + 72)
    #define p_empty_addr (mbar_base + 80)
    #define stats_full_addr (mbar_base + 88)
    #define tmem_dealloc_addr (mbar_base + 96)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(V)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* smem_st_a = reinterpret_cast<float*>(smem_raw + 191488);
    const int smem_st_a_addr = smem + 191488;
    float* smem_st_b = reinterpret_cast<float*>(smem_raw + 208000);
    const int smem_st_b_addr = smem + 208000;
    float* smem_scale = reinterpret_cast<float*>(smem_raw + 5120);
    const int smem_scale_addr = smem + 5120;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 2304);
    const int smem_sum_addr = smem + 2304;
    float* smem_max = reinterpret_cast<float*>(smem_raw + 2560);
    const int smem_max_addr = smem + 2560;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 3072);
    const int smem_red_addr = smem + 3072;
    float* smem_w = reinterpret_cast<float*>(smem_raw + 4096);
    const int smem_w_addr = smem + 4096;
    __nv_bfloat16* smem_q_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 11264);
    const int smem_q_hi_addr = smem + 11264;
    __nv_bfloat16* smem_q_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 27648);
    const int smem_q_lo_addr = smem + 27648;
    __nv_bfloat16* smem_kv_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 44032);
    const int smem_kv_hi_addr = smem + 44032;
    __nv_bfloat16* smem_kv_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 109568);
    const int smem_kv_lo_addr = smem + 109568;
    __nv_bfloat16* smem_v_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 44032);
    const int smem_v_hi_addr = smem + 44032;
    __nv_bfloat16* smem_v_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 109568);
    const int smem_v_lo_addr = smem + 109568;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 175104);
    const int smem_p_addr = smem + 175104;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Q)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory");

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 13 barriers)
    // Mbarriers at smem_raw[0..104)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // kv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // kv_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // corr_scale: 1 barriers, init_count=256
            mbarrier_init(smem + 56, 256);
            // p_full: 1 barriers, init_count=384
            mbarrier_init(smem + 64, 384);
            // o_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // p_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // stats_full: 1 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 96, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 224 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 104);
    if (warp == 0) {
        int _tmem_hold = smem + 104;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o_hi = taddr + 128;
    const int tmem_tmem_o_lo = taddr + 192;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int work = blockIdx.x;
            int tile_idx = work / NUM_SPLIT;
            int split_idx = work % NUM_SPLIT;
            int batch_idx = tile_idx / num_kv_heads;
            int kv_head_idx = tile_idx % num_kv_heads;
            int kv_len = seq_lens[batch_idx];
            int num_blocks = (kv_len + BLOCK_N - 1) / BLOCK_N;
            int base_blocks = num_blocks / NUM_SPLIT;
            int extra_blocks = num_blocks % NUM_SPLIT;
            int cnt = base_blocks;
            int start = extra_blocks * (base_blocks + 1) + (split_idx - extra_blocks) * base_blocks;
            if (split_idx < extra_blocks) {
                cnt = base_blocks + 1;
                start = split_idx * (base_blocks + 1);
            }
            int prefix = kv_len - Q_LEN;
            int wg = warp / 4;
            const int warp_in_wg = warp % 4;
            const int lane_base = warp_in_wg * 32;
            int wg_tid = warp_in_wg * 32 + lane;
            int col_local = wg_tid / 8;
            int quarter = wg_tid % 8;
            int col_base = wg * 16;
            int my_col = col_base + col_local;
            int tok_base = quarter * 8;
            int my_s_base = taddr + (unsigned int)(lane_base << 16) + (unsigned int)col_base;
            float* st_ptr = ((wg != 0) ? smem_st_b : smem_st_a);
            int vis_col = prefix + 1 + my_col / GROUP;
            float row_max = -CAKE_FMHA_INF;
            float psum = 0.0f;
            int sm_stage = 0;
            int sm_phase = 0;
            unsigned int _phase_p_empty_0 = 1;
            #pragma unroll 1
            for (int n = 0; n < cnt; n++) {
                mbarrier_wait(s_full_addr + (sm_stage) * 8, sm_phase);
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], my_s_base + sm_stage * 32);
                int tok_w = lane_base + lane;
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    st_ptr[c * 129 + tok_w] = _tmem_load_0[c];
                }
                if (wg != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                int my_block = start + cnt - 1 - n;
                int blk_pos = my_block * BLOCK_N + tok_base;
                float svals[16];
                float lmax = -CAKE_FMHA_INF;
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        float s_ji = st_ptr[col_local * 129 + tok_base + 64 * j + i];
                        if (vis_col <= blk_pos + 64 * j + i) {
                            s_ji = -CAKE_FMHA_INF;
                        }
                        svals[j * 8 + i] = s_ji;
                        float _max_0 = max_noftz(lmax, s_ji);
                        lmax = _max_0;
                    }
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, lmax, 1);
                float _max_1 = max_noftz(lmax, _shfl_xor_0);
                lmax = _max_1;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, lmax, 2);
                float _max_2 = max_noftz(lmax, _shfl_xor_1);
                lmax = _max_2;
                {
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, lmax, 4);
                    float _max_3 = max_noftz(lmax, _shfl_xor_2);
                    lmax = _max_3;
                }
                float _max_4 = max_noftz(row_max, lmax);
                float new_max = _max_4;
                float acc_scale = 1.0f;
                if (row_max > -CAKE_FMHA_INF) {
                    float _exp2_0 = approx_exp2(softmax_scale_log2 * (row_max - new_max));
                    acc_scale = _exp2_0;
                }
                if (quarter == 0) {
                    smem_scale[sm_stage * 64 + my_col] = acc_scale;
                }
                mbarrier_arrive(corr_scale_addr);
                float safe_max = ((new_max == -CAKE_FMHA_INF) ? 0.0f : new_max);
                float p_vals[16];
                float lsum = 0.0f;
                #pragma unroll
                for (int k = 0; k < 16; k++) {
                    float _exp2_1 = approx_exp2((svals[k] - safe_max) * softmax_scale_log2);
                    float p_k = _exp2_1;
                    p_vals[k] = p_k;
                    lsum = lsum + p_k;
                }
                psum = psum * acc_scale + lsum;
                row_max = new_max;
                mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                _phase_p_empty_0 ^= 1;
                int k_run = 0;
                unsigned int regs_p[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_vals[_lp*2 + 0], p_vals[_lp*2+1 + 0]));
                    regs_p[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)((k_run / 64 * 32 + my_col) * 128 + (k_run % 64 + tok_base) * 2 ^ ((k_run / 64 * 32 + my_col) * 128 + (k_run % 64 + tok_base) * 2 >> 7 & 7) << 4))), "r"(regs_p[0]), "r"(regs_p[1]), "r"(regs_p[2]), "r"(regs_p[3]) : "memory");
                int k_run_0 = 64;
                unsigned int regs_p_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_vals[_lp*2 + 8], p_vals[_lp*2+1 + 8]));
                    regs_p_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)((k_run_0 / 64 * 32 + my_col) * 128 + (k_run_0 % 64 + tok_base) * 2 ^ ((k_run_0 / 64 * 32 + my_col) * 128 + (k_run_0 % 64 + tok_base) * 2 >> 7 & 7) << 4))), "r"(regs_p_1[0]), "r"(regs_p_1[1]), "r"(regs_p_1[2]), "r"(regs_p_1[3]) : "memory");
                asm volatile("fence.proxy.async;");
                mbarrier_arrive(p_full_addr);
                if (wg != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                sm_stage += 1;
                if (sm_stage == 2) { sm_stage = 0; sm_phase ^= 1; }
            }
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, psum, 1);
            float total = psum + _shfl_xor_3;
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
            total = total + _shfl_xor_4;
            {
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
                total = total + _shfl_xor_5;
            }
            if (quarter == 0) {
                smem_sum[my_col] = total;
                smem_max[my_col] = row_max;
            }
            mbarrier_arrive(stats_full_addr);
            int flat_tid_s = warp * 32 + lane;
            {
                int merge_warp = flat_tid_s / 32;
                int lane_in_warp = flat_tid_s % 32;
                int d0 = lane_in_warp * 8;
                asm volatile("fence.release.gpu;" ::: "memory");
                asm volatile("barrier.sync 10, 384;" ::: "memory");
                if (flat_tid_s == 0) {
                    unsigned int _atomic_old_0;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_old_0) : "l"(&counters[tile_idx * 2]), "r"(static_cast<uint32_t>(1)) : "memory");
                    unsigned int expected_arrivals = NUM_SPLIT;
                    #pragma unroll 1
                    for (int _poll = 0; _poll < 1073741824; _poll++) {
                        unsigned int _atomic_old_1;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(_atomic_old_1) : "l"(&counters[tile_idx * 2]), "r"(static_cast<uint32_t>(0)) : "memory");
                        unsigned int arrived = _atomic_old_1;
                        if (arrived >= expected_arrivals) {
                            break;
                        }
                    }
                }
                asm volatile("barrier.sync 10, 384;" ::: "memory");
                asm volatile("fence.acquire.gpu;" ::: "memory");
                int slot_base = tile_idx * NUM_SPLIT;
                #pragma unroll 1
                for (int r = split_idx; r < 32; r += NUM_SPLIT) {
                    float my_lse = -CAKE_FMHA_INF;
                    if (flat_tid_s < NUM_SPLIT) {
                        my_lse = partial_LSE_ptr[(slot_base + flat_tid_s) * 32 + r];
                    }
                    float _warp_reduce_0 = my_lse;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                    float warp_max = _warp_reduce_0;
                    if (lane_in_warp == 0) {
                        smem_red[merge_warp] = warp_max;
                    }
                    asm volatile("barrier.sync 10, 384;" ::: "memory");
                    float merged_max = smem_red[0];
                    #pragma unroll
                    for (int w = 1; w < 12; w++) {
                        float _max_5 = max_noftz(merged_max, smem_red[w]);
                        merged_max = _max_5;
                    }
                    float my_w = 0.0f;
                    if (flat_tid_s < NUM_SPLIT && my_lse != -CAKE_FMHA_INF) {
                        float _exp2_2 = approx_exp2(my_lse - merged_max);
                        my_w = _exp2_2;
                    }
                    float _warp_reduce_1 = my_w;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                    float warp_sum = _warp_reduce_1;
                    asm volatile("barrier.sync 10, 384;" ::: "memory");
                    if (flat_tid_s < NUM_SPLIT) {
                        smem_w[flat_tid_s] = my_w;
                    }
                    if (lane_in_warp == 0) {
                        smem_red[merge_warp] = warp_sum;
                    }
                    asm volatile("barrier.sync 10, 384;" ::: "memory");
                    float weight_sum = smem_red[0];
                    #pragma unroll
                    for (int w_1 = 1; w_1 < 12; w_1++) {
                        weight_sum = weight_sum + smem_red[w_1];
                    }
                    float acc[8];
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    #pragma unroll 4
                    for (int i_1 = merge_warp; i_1 < NUM_SPLIT; i_1 += 12) {
                        float w_i = smem_w[i_1];
                        float _vec_load_0[8];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_O_ptr + (((slot_base + i_1) * 32 + r) * HEAD_DIM + d0) + 0);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                        #pragma unroll
                        for (int k_1 = 0; k_1 < 8; k_1++) {
                            float _fma_0 = __fmaf_rn(_vec_load_0[k_1], w_i, acc[k_1]);
                            acc[k_1] = _fma_0;
                        }
                    }
                    #pragma unroll
                    for (int k_2 = 0; k_2 < 8; k_2++) {
                        smem_st_a[(merge_warp * 32 + lane_in_warp) * 8 + k_2] = acc[k_2];
                    }
                    asm volatile("barrier.sync 10, 384;" ::: "memory");
                    if (merge_warp == 0) {
                        float _rcp_0 = approx_rcp(weight_sum);
                        float inv_sum = ((weight_sum > 0.0f) ? _rcp_0 : 0.0f);
                        float out[8];
                        #pragma unroll
                        for (int k_3 = 0; k_3 < 8; k_3++) {
                            float total_k = smem_st_a[lane_in_warp * 8 + k_3];
                            #pragma unroll
                            for (int g = 1; g < 12; g++) {
                                total_k = total_k + smem_st_a[(g * 32 + lane_in_warp) * 8 + k_3];
                            }
                            out[k_3] = total_k * inv_sum;
                        }
                        if (r < Q_LEN * GROUP) {
                            int j_row = r / GROUP;
                            int h_row = r % GROUP;
                            int q_row = batch_idx * Q_LEN + j_row;
                            int q_head = kv_head_idx * GROUP + h_row;
                            int o_idx = (q_row * num_q_heads + q_head) * HEAD_DIM + d0;
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(out[0 + 0], out[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(out[0 + 2], out[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(out[0 + 4], out[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(out[0 + 6], out[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + o_idx))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                            if (lane_in_warp == 0) {
                                float merged_lse = -CAKE_FMHA_INF;
                                if (weight_sum > 0.0f) {
                                    float _log2_0;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(weight_sum));
                                    merged_lse = merged_max + _log2_0;
                                }
                                *(reinterpret_cast<float*>(LSE_ptr + (q_row * num_q_heads + q_head)) + (0)) = merged_lse;
                            }
                        }
                    }
                    asm volatile("barrier.sync 10, 384;" ::: "memory");
                }
                if (flat_tid_s == 0) {
                    unsigned int _atomic_old_2 = atomicAdd(&counters[tile_idx * 2 + 1], 1);
                    unsigned int old_done = _atomic_old_2;
                    if (old_done + 1 == (unsigned int)NUM_SPLIT) {
                        *(reinterpret_cast<unsigned int*>(counters + (tile_idx * 2)) + (0)) = 0;
                        *(reinterpret_cast<unsigned int*>(counters + (tile_idx * 2 + 1)) + (0)) = 0;
                    }
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            int work_1 = blockIdx.x;
            int tile_idx_1 = work_1 / NUM_SPLIT;
            int split_idx_1 = work_1 % NUM_SPLIT;
            int batch_idx_1 = tile_idx_1 / num_kv_heads;
            int kv_head_idx_1 = tile_idx_1 % num_kv_heads;
            int kv_len_1 = seq_lens[batch_idx_1];
            int num_blocks_1 = (kv_len_1 + BLOCK_N - 1) / BLOCK_N;
            int base_blocks_1 = num_blocks_1 / NUM_SPLIT;
            int extra_blocks_1 = num_blocks_1 % NUM_SPLIT;
            int cnt_1 = base_blocks_1;
            int start_1 = extra_blocks_1 * (base_blocks_1 + 1) + (split_idx_1 - extra_blocks_1) * base_blocks_1;
            if (split_idx_1 < extra_blocks_1) {
                cnt_1 = base_blocks_1 + 1;
                start_1 = split_idx_1 * (base_blocks_1 + 1);
            }
            int prefix_1 = kv_len_1 - Q_LEN;
            const int warp_in_wg_c = warp % 4;
            const int corr_row = warp_in_wg_c * 32 << 16;
            int d_idx = warp_in_wg_c * 32 + lane;
            unsigned int _phase_corr_scale_0 = 0;
            unsigned int _phase_o_ready_0 = 0;
            #pragma unroll 1
            for (int n_1 = 0; n_1 < cnt_1; n_1++) {
                mbarrier_wait(corr_scale_addr, _phase_corr_scale_0);
                _phase_corr_scale_0 ^= 1;
                if (n_1 > 0) {
                    mbarrier_wait(o_ready_addr, _phase_o_ready_0);
                    _phase_o_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int sc_off = n_1 % NUM_KV_STAGES * 64;
                    int need = 0;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 32; c_1++) {
                        need = need | ((smem_scale[sc_off + c_1] != 1.0f) ? 1 : 0);
                    }
                    int _vote_0 = __any_sync(0xFFFFFFFF, need != 0);
                    if (_vote_0 != 0) {
                        #pragma unroll
                        for (int half = 0; half < 2; half++) {
                            int o_base = taddr + (unsigned int)(((half == 0) ? 128 : 192)) + (unsigned int)corr_row;
                            #pragma unroll
                            for (int chunk = 0; chunk < 1; chunk++) {
                                float _tmem_load_1[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                                    : "r"(o_base + chunk * 32));
                                #pragma unroll
                                for (int c_2 = 0; c_2 < 32; c_2++) {
                                    _tmem_load_1[c_2] = _tmem_load_1[c_2] * smem_scale[sc_off + chunk * 32 + c_2];
                                }
                                tmem_st_x32_f32(o_base + chunk * 32, _tmem_load_1);
                            }
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                mbarrier_arrive(p_full_addr);
            }
            if (cnt_1 > 0) {
                mbarrier_wait(o_ready_addr, _phase_o_ready_0);
                _phase_o_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            unsigned int _phase_stats_full_0 = 0;
            mbarrier_wait(stats_full_addr, _phase_stats_full_0);
            _phase_stats_full_0 ^= 1;
            int slot = work_1;
            #pragma unroll
            for (int half_1 = 0; half_1 < 2; half_1++) {
                int o_base_e = taddr + (unsigned int)(((half_1 == 0) ? 128 : 192)) + (unsigned int)corr_row;
                #pragma unroll
                for (int chunk_1 = 0; chunk_1 < 1; chunk_1++) {
                    float o_epi[32];
                    if (cnt_1 > 0) {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(o_epi[0]), "=f"(o_epi[1]), "=f"(o_epi[2]), "=f"(o_epi[3]), "=f"(o_epi[4]), "=f"(o_epi[5]), "=f"(o_epi[6]), "=f"(o_epi[7]), "=f"(o_epi[8]), "=f"(o_epi[9]), "=f"(o_epi[10]), "=f"(o_epi[11]), "=f"(o_epi[12]), "=f"(o_epi[13]), "=f"(o_epi[14]), "=f"(o_epi[15]), "=f"(o_epi[16]), "=f"(o_epi[17]), "=f"(o_epi[18]), "=f"(o_epi[19]), "=f"(o_epi[20]), "=f"(o_epi[21]), "=f"(o_epi[22]), "=f"(o_epi[23]), "=f"(o_epi[24]), "=f"(o_epi[25]), "=f"(o_epi[26]), "=f"(o_epi[27]), "=f"(o_epi[28]), "=f"(o_epi[29]), "=f"(o_epi[30]), "=f"(o_epi[31])
                            : "r"(o_base_e + chunk_1 * 32));
                    } else {
                        o_epi[0] = 0.0f;
                        o_epi[1] = 0.0f;
                        o_epi[2] = 0.0f;
                        o_epi[3] = 0.0f;
                        o_epi[4] = 0.0f;
                        o_epi[5] = 0.0f;
                        o_epi[6] = 0.0f;
                        o_epi[7] = 0.0f;
                        o_epi[8] = 0.0f;
                        o_epi[9] = 0.0f;
                        o_epi[10] = 0.0f;
                        o_epi[11] = 0.0f;
                        o_epi[12] = 0.0f;
                        o_epi[13] = 0.0f;
                        o_epi[14] = 0.0f;
                        o_epi[15] = 0.0f;
                        o_epi[16] = 0.0f;
                        o_epi[17] = 0.0f;
                        o_epi[18] = 0.0f;
                        o_epi[19] = 0.0f;
                        o_epi[20] = 0.0f;
                        o_epi[21] = 0.0f;
                        o_epi[22] = 0.0f;
                        o_epi[23] = 0.0f;
                        o_epi[24] = 0.0f;
                        o_epi[25] = 0.0f;
                        o_epi[26] = 0.0f;
                        o_epi[27] = 0.0f;
                        o_epi[28] = 0.0f;
                        o_epi[29] = 0.0f;
                        o_epi[30] = 0.0f;
                        o_epi[31] = 0.0f;
                    }
                    #pragma unroll
                    for (int c_3 = 0; c_3 < 32; c_3++) {
                        const int r_c = chunk_1 * 32 + c_3;
                        float row_sum_c = smem_sum[r_c];
                        float _rcp_1 = approx_rcp(row_sum_c);
                        float inv_c = ((row_sum_c > 0.0f) ? _rcp_1 : 0.0f);
                        float val_c = o_epi[c_3] * inv_c;
                        if (cnt_1 == 0) {
                            val_c = 0.0f;
                        }
                        int p_idx = (slot * 32 + r_c) * HEAD_DIM + half_1 * HEAD_DIM_HALF + d_idx;
                        *(reinterpret_cast<__nv_bfloat16*>(partial_O_ptr + p_idx) + (0)) = __float2bfloat16_rn(val_c);
                    }
                }
            }
            if (warp_in_wg_c == 0) {
                #pragma unroll
                for (int chunk_2 = 0; chunk_2 < 1; chunk_2++) {
                    int r_l = chunk_2 * 32 + lane;
                    float m_l = smem_max[r_l];
                    float s_l = smem_sum[r_l];
                    float lse_l = -CAKE_FMHA_INF;
                    if (s_l > 0.0f && m_l > -CAKE_FMHA_INF) {
                        float _log2_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(s_l));
                        lse_l = m_l * softmax_scale_log2 + _log2_1;
                    }
                    *(reinterpret_cast<float*>(partial_LSE_ptr + (slot * 32 + r_l)) + (0)) = lse_l;
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
            int flat_tid_c = 256 + (warp - 8) * 32 + lane;
            {
                int merge_warp_1 = flat_tid_c / 32;
                int lane_in_warp_1 = flat_tid_c % 32;
                int d0_1 = lane_in_warp_1 * 8;
                asm volatile("fence.release.gpu;" ::: "memory");
                asm volatile("barrier.sync 10, 384;" ::: "memory");
                if (flat_tid_c == 0) {
                    unsigned int _atomic_old_3;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(_atomic_old_3) : "l"(&counters[tile_idx_1 * 2]), "r"(static_cast<uint32_t>(1)) : "memory");
                    unsigned int expected_arrivals_1 = NUM_SPLIT;
                    #pragma unroll 1
                    for (int _poll_1 = 0; _poll_1 < 1073741824; _poll_1++) {
                        unsigned int _atomic_old_4;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(_atomic_old_4) : "l"(&counters[tile_idx_1 * 2]), "r"(static_cast<uint32_t>(0)) : "memory");
                        unsigned int arrived_1 = _atomic_old_4;
                        if (arrived_1 >= expected_arrivals_1) {
                            break;
                        }
                    }
                }
                asm volatile("barrier.sync 10, 384;" ::: "memory");
                asm volatile("fence.acquire.gpu;" ::: "memory");
                int slot_base_1 = tile_idx_1 * NUM_SPLIT;
                #pragma unroll 1
                for (int r_1 = split_idx_1; r_1 < 32; r_1 += NUM_SPLIT) {
                    float my_lse_1 = -CAKE_FMHA_INF;
                    if (flat_tid_c < NUM_SPLIT) {
                        my_lse_1 = partial_LSE_ptr[(slot_base_1 + flat_tid_c) * 32 + r_1];
                    }
                    float _warp_reduce_2 = my_lse_1;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_2 = max_noftz(_warp_reduce_2, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset));
                    float warp_max_1 = _warp_reduce_2;
                    if (lane_in_warp_1 == 0) {
                        smem_red[merge_warp_1] = warp_max_1;
                    }
                    asm volatile("barrier.sync 10, 384;" ::: "memory");
                    float merged_max_1 = smem_red[0];
                    #pragma unroll
                    for (int w_2 = 1; w_2 < 12; w_2++) {
                        float _max_6 = max_noftz(merged_max_1, smem_red[w_2]);
                        merged_max_1 = _max_6;
                    }
                    float my_w_1 = 0.0f;
                    if (flat_tid_c < NUM_SPLIT && my_lse_1 != -CAKE_FMHA_INF) {
                        float _exp2_3 = approx_exp2(my_lse_1 - merged_max_1);
                        my_w_1 = _exp2_3;
                    }
                    float _warp_reduce_3 = my_w_1;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
                    float warp_sum_1 = _warp_reduce_3;
                    asm volatile("barrier.sync 10, 384;" ::: "memory");
                    if (flat_tid_c < NUM_SPLIT) {
                        smem_w[flat_tid_c] = my_w_1;
                    }
                    if (lane_in_warp_1 == 0) {
                        smem_red[merge_warp_1] = warp_sum_1;
                    }
                    asm volatile("barrier.sync 10, 384;" ::: "memory");
                    float weight_sum_1 = smem_red[0];
                    #pragma unroll
                    for (int w_3 = 1; w_3 < 12; w_3++) {
                        weight_sum_1 = weight_sum_1 + smem_red[w_3];
                    }
                    float acc_1[8];
                    acc_1[0] = 0.0f;
                    acc_1[1] = 0.0f;
                    acc_1[2] = 0.0f;
                    acc_1[3] = 0.0f;
                    acc_1[4] = 0.0f;
                    acc_1[5] = 0.0f;
                    acc_1[6] = 0.0f;
                    acc_1[7] = 0.0f;
                    #pragma unroll 4
                    for (int i_2 = merge_warp_1; i_2 < NUM_SPLIT; i_2 += 12) {
                        float w_i_1 = smem_w[i_2];
                        float _vec_load_1[8];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_O_ptr + (((slot_base_1 + i_2) * 32 + r_1) * HEAD_DIM + d0_1) + 0);
                            uint4 _vld_0[1];
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                        #pragma unroll
                        for (int k_4 = 0; k_4 < 8; k_4++) {
                            float _fma_1 = __fmaf_rn(_vec_load_1[k_4], w_i_1, acc_1[k_4]);
                            acc_1[k_4] = _fma_1;
                        }
                    }
                    #pragma unroll
                    for (int k_5 = 0; k_5 < 8; k_5++) {
                        smem_st_a[(merge_warp_1 * 32 + lane_in_warp_1) * 8 + k_5] = acc_1[k_5];
                    }
                    asm volatile("barrier.sync 10, 384;" ::: "memory");
                    if (merge_warp_1 == 0) {
                        float _rcp_2 = approx_rcp(weight_sum_1);
                        float inv_sum_1 = ((weight_sum_1 > 0.0f) ? _rcp_2 : 0.0f);
                        float out_1[8];
                        #pragma unroll
                        for (int k_6 = 0; k_6 < 8; k_6++) {
                            float total_k_1 = smem_st_a[lane_in_warp_1 * 8 + k_6];
                            #pragma unroll
                            for (int g_1 = 1; g_1 < 12; g_1++) {
                                total_k_1 = total_k_1 + smem_st_a[(g_1 * 32 + lane_in_warp_1) * 8 + k_6];
                            }
                            out_1[k_6] = total_k_1 * inv_sum_1;
                        }
                        if (r_1 < Q_LEN * GROUP) {
                            int j_row_1 = r_1 / GROUP;
                            int h_row_1 = r_1 % GROUP;
                            int q_row_1 = batch_idx_1 * Q_LEN + j_row_1;
                            int q_head_1 = kv_head_idx_1 * GROUP + h_row_1;
                            int o_idx_1 = (q_row_1 * num_q_heads + q_head_1) * HEAD_DIM + d0_1;
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(out_1[0 + 0], out_1[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(out_1[0 + 2], out_1[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(out_1[0 + 4], out_1[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(out_1[0 + 6], out_1[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + o_idx_1))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                            if (lane_in_warp_1 == 0) {
                                float merged_lse_1 = -CAKE_FMHA_INF;
                                if (weight_sum_1 > 0.0f) {
                                    float _log2_2;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(weight_sum_1));
                                    merged_lse_1 = merged_max_1 + _log2_2;
                                }
                                *(reinterpret_cast<float*>(LSE_ptr + (q_row_1 * num_q_heads + q_head_1)) + (0)) = merged_lse_1;
                            }
                        }
                    }
                    asm volatile("barrier.sync 10, 384;" ::: "memory");
                }
                if (flat_tid_c == 0) {
                    unsigned int _atomic_old_5 = atomicAdd(&counters[tile_idx_1 * 2 + 1], 1);
                    unsigned int old_done_1 = _atomic_old_5;
                    if (old_done_1 + 1 == (unsigned int)NUM_SPLIT) {
                        *(reinterpret_cast<unsigned int*>(counters + (tile_idx_1 * 2)) + (0)) = 0;
                        *(reinterpret_cast<unsigned int*>(counters + (tile_idx_1 * 2 + 1)) + (0)) = 0;
                    }
                }
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            int work_2 = blockIdx.x;
            int tile_idx_2 = work_2 / NUM_SPLIT;
            int split_idx_2 = work_2 % NUM_SPLIT;
            int batch_idx_2 = tile_idx_2 / num_kv_heads;
            int kv_head_idx_2 = tile_idx_2 % num_kv_heads;
            int kv_len_2 = seq_lens[batch_idx_2];
            int num_blocks_2 = (kv_len_2 + BLOCK_N - 1) / BLOCK_N;
            int base_blocks_2 = num_blocks_2 / NUM_SPLIT;
            int extra_blocks_2 = num_blocks_2 % NUM_SPLIT;
            int cnt_2 = base_blocks_2;
            int start_2 = extra_blocks_2 * (base_blocks_2 + 1) + (split_idx_2 - extra_blocks_2) * base_blocks_2;
            if (split_idx_2 < extra_blocks_2) {
                cnt_2 = base_blocks_2 + 1;
                start_2 = split_idx_2 * (base_blocks_2 + 1);
            }
            int prefix_2 = kv_len_2 - Q_LEN;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_p_full_0 = 0;
            if (cnt_2 > 0) {
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                mbarrier_wait(kv_full_addr, 0);
                int _mma_a_lo_0 = make_warp_uniform((((smem_kv_hi_addr) >> 4) & 0x3FFF) + (0) * 2048);
                int _mma_b_lo_0 = make_warp_uniform(((smem_q_hi_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 250U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                    }
                }
                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (0) * 2048);
                int _mma_b_lo_1 = make_warp_uniform(((smem_q_lo_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 250U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                    }
                }
                elect_commit(s_full_addr);
                elect_commit(kv_empty_addr);
                int first_pv = 1;
                #pragma unroll 1
                for (int n_2 = 0; n_2 < cnt_2; n_2++) {
                    int stage = n_2 % NUM_KV_STAGES;
                    int next_n = n_2 + 1;
                    if (next_n < cnt_2) {
                        int nstage = next_n % NUM_KV_STAGES;
                        mbarrier_wait(kv_full_addr + (nstage) * 8, 0);
                        int _mma_a_lo_2 = make_warp_uniform((((smem_kv_hi_addr) >> 4) & 0x3FFF) + (nstage) * 2048);
                        int _mma_b_lo_2 = make_warp_uniform(((smem_q_hi_addr) >> 4) & 0x3FFF);
                        {
                            uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 250U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                            }
                        }
                        int _mma_a_lo_3 = make_warp_uniform((((smem_kv_lo_addr) >> 4) & 0x3FFF) + (nstage) * 2048);
                        int _mma_b_lo_3 = make_warp_uniform(((smem_q_lo_addr) >> 4) & 0x3FFF);
                        {
                            uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                            uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 1018U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 250U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s + (nstage * 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                            }
                        }
                        elect_commit(s_full_addr + (nstage) * 8);
                        elect_commit(kv_empty_addr + (nstage) * 8);
                    }
                    mbarrier_wait(kv_full_addr + (stage) * 8, 1);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int first_pv_flag = first_pv;
                    int _mma_a_lo_4 = make_warp_uniform(((((smem_v_hi_addr) >> 4) & 0x3FFF) | 0x4000000) + (stage) * 2048);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000);
                    {
                        uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, ((first_pv_flag) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 250U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134775952, 1);
                        }
                    }
                    int _mma_a_lo_5 = make_warp_uniform(((((smem_v_lo_addr) >> 4) & 0x3FFF) | 0x4000000) + (stage) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                        uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, ((first_pv_flag) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 250U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134775952, 1);
                        }
                    }
                    elect_commit2(kv_empty_addr + (stage) * 8, o_ready_addr);
                    elect_commit(p_empty_addr);
                    first_pv = 0;
                }
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    }
    // ---- Role: load_warp ----
    if (warp == 13) {
        { // load_warp_main
            int work_3 = blockIdx.x;
            int tile_idx_3 = work_3 / NUM_SPLIT;
            int split_idx_3 = work_3 % NUM_SPLIT;
            int batch_idx_3 = tile_idx_3 / num_kv_heads;
            int kv_head_idx_3 = tile_idx_3 % num_kv_heads;
            int kv_len_3 = seq_lens[batch_idx_3];
            int num_blocks_3 = (kv_len_3 + BLOCK_N - 1) / BLOCK_N;
            int base_blocks_3 = num_blocks_3 / NUM_SPLIT;
            int extra_blocks_3 = num_blocks_3 % NUM_SPLIT;
            int cnt_3 = base_blocks_3;
            int start_3 = extra_blocks_3 * (base_blocks_3 + 1) + (split_idx_3 - extra_blocks_3) * base_blocks_3;
            if (split_idx_3 < extra_blocks_3) {
                cnt_3 = base_blocks_3 + 1;
                start_3 = split_idx_3 * (base_blocks_3 + 1);
            }
            int prefix_3 = kv_len_3 - Q_LEN;
            int max_pg = (kv_len_3 + 32 - 1) / 32 - 1;
            int pt_base = batch_idx_3 * max_pages_per_seq;
            const int kv_tx = BLOCK_N * HEAD_DIM * 2;
            if (cnt_3 > 0) {
                if (elect_sync()) {
                    int q_head0 = kv_head_idx_3 * GROUP;
                    int q_row0 = batch_idx_3 * Q_LEN;
                    mbarrier_arrive_expect_tx(q_full_addr, 32 * HEAD_DIM * 2);
                    tma_4d_gmem2smem(smem_q_hi_addr, Q, 0, q_head0, q_row0, 0, q_full_addr);
                    tma_4d_gmem2smem(smem_q_lo_addr, Q, 0, q_head0, q_row0, 2, q_full_addr);
                    int kv_stage = 0;
                    int kv_phase = 1;
                    int prefill = ((cnt_3 < NUM_KV_STAGES) ? cnt_3 : NUM_KV_STAGES);
                    #pragma unroll 1
                    for (int ni = 0; ni < prefill; ni++) {
                        int n_block = start_3 + cnt_3 - 1 - ni;
                        mbarrier_wait(kv_empty_addr + (kv_stage) * 8, kv_phase);
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, kv_tx);
                        int hdst = smem_kv_hi_addr + (unsigned int)(kv_stage * 32768);
                        int ldst = smem_kv_lo_addr + (unsigned int)(kv_stage * 32768);
                        #pragma unroll
                        for (int pg_i = 0; pg_i < 4; pg_i++) {
                            int raw_p = n_block * 4 + pg_i;
                            int clamp_p = ((raw_p > max_pg) ? max_pg : raw_p);
                            int pg0 = page_table[pt_base + clamp_p];
                            #pragma unroll
                            for (int hg = 0; hg < 2; hg++) {
                                int toff = hg * 16384 + pg_i * 4096;
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(hdst + toff), "l"(K), "r"(0), "r"(0), "r"(hg), "r"(kv_head_idx_3), "r"(pg0),
                                       "r"(kv_full_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            }
                            #pragma unroll
                            for (int hg_1 = 0; hg_1 < 2; hg_1++) {
                                int toff_1 = hg_1 * 16384 + pg_i * 4096;
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(ldst + toff_1), "l"(K), "r"(0), "r"(0), "r"(hg_1 + 2), "r"(kv_head_idx_3), "r"(pg0),
                                       "r"(kv_full_addr + (kv_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            }
                        }
                        kv_stage += 1;
                        if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
                    }
                    #pragma unroll 1
                    for (int ni_1 = 0; ni_1 < cnt_3; ni_1++) {
                        int stage_1 = ni_1 % NUM_KV_STAGES;
                        int n_block_1 = start_3 + cnt_3 - 1 - ni_1;
                        mbarrier_wait(kv_empty_addr + (stage_1) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (stage_1) * 8, kv_tx);
                        int hdst_1 = smem_kv_hi_addr + (unsigned int)(stage_1 * 32768);
                        int ldst_1 = smem_kv_lo_addr + (unsigned int)(stage_1 * 32768);
                        #pragma unroll
                        for (int pg_i_1 = 0; pg_i_1 < 4; pg_i_1++) {
                            int raw_p_1 = n_block_1 * 4 + pg_i_1;
                            int clamp_p_1 = ((raw_p_1 > max_pg) ? max_pg : raw_p_1);
                            int pg0_1 = page_table[pt_base + clamp_p_1];
                            #pragma unroll
                            for (int hg_2 = 0; hg_2 < 2; hg_2++) {
                                int toff_2 = hg_2 * 16384 + pg_i_1 * 4096;
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(hdst_1 + toff_2), "l"(V), "r"(0), "r"(0), "r"(hg_2), "r"(kv_head_idx_3), "r"(pg0_1),
                                       "r"(kv_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            }
                            #pragma unroll
                            for (int hg_3 = 0; hg_3 < 2; hg_3++) {
                                int toff_3 = hg_3 * 16384 + pg_i_1 * 4096;
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(ldst_1 + toff_3), "l"(V), "r"(0), "r"(0), "r"(hg_3 + 2), "r"(kv_head_idx_3), "r"(pg0_1),
                                       "r"(kv_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            }
                        }
                        int next_ni = ni_1 + NUM_KV_STAGES;
                        if (next_ni < cnt_3) {
                            int next_block = start_3 + cnt_3 - 1 - next_ni;
                            mbarrier_wait(kv_empty_addr + (stage_1) * 8, 1);
                            mbarrier_arrive_expect_tx(kv_full_addr + (stage_1) * 8, kv_tx);
                            int hdst_0 = smem_kv_hi_addr + (unsigned int)(stage_1 * 32768);
                            int ldst_1_1 = smem_kv_lo_addr + (unsigned int)(stage_1 * 32768);
                            #pragma unroll
                            for (int pg_i_2 = 0; pg_i_2 < 4; pg_i_2++) {
                                int raw_p_2 = next_block * 4 + pg_i_2;
                                int clamp_p_2 = ((raw_p_2 > max_pg) ? max_pg : raw_p_2);
                                int pg0_2 = page_table[pt_base + clamp_p_2];
                                #pragma unroll
                                for (int hg_4 = 0; hg_4 < 2; hg_4++) {
                                    int toff_4 = hg_4 * 16384 + pg_i_2 * 4096;
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(hdst_0 + toff_4), "l"(K), "r"(0), "r"(0), "r"(hg_4), "r"(kv_head_idx_3), "r"(pg0_2),
                                           "r"(kv_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                                }
                                #pragma unroll
                                for (int hg_5 = 0; hg_5 < 2; hg_5++) {
                                    int toff_5 = hg_5 * 16384 + pg_i_2 * 4096;
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(ldst_1_1 + toff_5), "l"(K), "r"(0), "r"(0), "r"(hg_5 + 2), "r"(kv_head_idx_3), "r"(pg0_2),
                                           "r"(kv_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: pad ----
    if (warp >= 14 && warp <= 15) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
