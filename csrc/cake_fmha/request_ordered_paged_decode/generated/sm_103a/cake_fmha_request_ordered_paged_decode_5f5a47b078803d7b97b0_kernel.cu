// Copyright (c) 2026 FlashInfer contributors.
// SPDX-License-Identifier: Apache-2.0

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
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 48
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_STATS0_OFFSET 16
#define TMEM_TMEM_O_HI_OFFSET 32
#define TMEM_TMEM_O_LO_OFFSET 40
#define NUM_RAW_KV_PIPE_STAGES 4
#define NUM_TRANSFORMED_KV_PIPE_STAGES 2
#define NUM_SM_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 2
#define NUM_THROTTLE_PIPE_STAGES 2
#define NUM_PAGE_PIPE_STAGES 6
#define NUM_CORR_PIPE_STAGES 2
#define NUM_P_PIPE_STAGES 2
#define SMEM_SMEM_CORR_OFF 1920
#define SMEM_SMEM_CORR_STAGE_BYTES 128
#define SMEM_SMEM_CORR_STRIDE 128
#define SMEM_SMEM_EXCH_OFF 1088
#define SMEM_SMEM_EXCH_STAGE_BYTES 256
#define SMEM_SMEM_EXCH_STRIDE 256
#define SMEM_SMEM_EXCH_U32_OFF 1088
#define SMEM_SMEM_EXCH_U32_STAGE_BYTES 256
#define SMEM_SMEM_EXCH_U32_STRIDE 256
#define SMEM_SMEM_QT_HI_OFF 2048
#define SMEM_SMEM_QT_HI_STAGE_BYTES 2048
#define SMEM_SMEM_QT_HI_STRIDE 2048
#define SMEM_SMEM_QT_LO_OFF 4096
#define SMEM_SMEM_QT_LO_STAGE_BYTES 2048
#define SMEM_SMEM_QT_LO_STRIDE 2048
#define SMEM_SMEM_KV_FP8_OFF 71680
#define SMEM_SMEM_KV_FP8_STAGE_BYTES 16384
#define SMEM_SMEM_KV_FP8_STRIDE 16384
#define SMEM_SMEM_KV_OFF 6144
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 6144
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P_OFF 137216
#define SMEM_SMEM_P_STAGE_BYTES 2048
#define SMEM_SMEM_P_STRIDE 2048
#define SMEM_WORK_RESPONSE_VIEW_OFF 1344
#define SMEM_WORK_RESPONSE_VIEW_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_VIEW_STRIDE 16
#define SMEM_SPLIT_REDUCE_FLAG_OFF 141312
#define SMEM_SPLIT_REDUCE_FLAG_STAGE_BYTES 4
#define SMEM_SPLIT_REDUCE_FLAG_STRIDE 4
#define SMEM_SPLIT_WEIGHTS_OFF 1408
#define SMEM_SPLIT_WEIGHTS_STAGE_BYTES 512
#define SMEM_SPLIT_WEIGHTS_STRIDE 512
#define SMEM_SMEM_PAGE_OFFSETS_OFF 141440
#define SMEM_SMEM_PAGE_OFFSETS_STAGE_BYTES 768
#define SMEM_SMEM_PAGE_OFFSETS_STRIDE 768
#define SMEM_TOTAL 142208
#define THREADS 512
#define BLOCK_N 128
#define HEAD_DIM 256
#define HEAD_DIM_HALF 128
#define TILE_Q 8
#define PAGE_SIZE 64
#define NUM_RAW_KV_STAGES 4
#define NUM_TRANSFORMED_KV_STAGES 2
#define Q_LEN 6
#define UNIFORM_KV_LEN 0
#define USE_REQUEST_ORDER 0
#define USE_SCALE_POINTERS 1
#define NUM_SPLIT 1
#define USE_SEGMENTED_CLC 0
#define USE_HIGH_BATCH_TWO_WAVE 0
#define USE_TWO_CTA_REDUCER 0
#define USE_MMA_LOOP_PEEL 1
#define USE_PAGE_OFFSET_CPASYNC 0
#define USE_LEGACY_PAGE_VEC4 0
#define WRITE_LSE 0

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
           "r"(i_desc), "r"(enable_input_d));
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
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %3, p;\n\t"
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


__device__ __forceinline__ void tmem_st_x4_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32"
        " [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
}


__device__ __forceinline__ void tmem_st_x8_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]),
           "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7]));
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


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_cake_fmha_request_ordered_paged_decode_5f5a47b078803d7b97b0(CakeTensorMap const* Qt, CakeTensorMap const* K, CakeTensorMap const* V, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_LSE, unsigned int* __restrict__ split_completion, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, int* __restrict__ request_order, int max_pages_per_seq, int page_table_v_offset, float softmax_scale_log2, float output_scale, float* __restrict__ bmm1_scale_ptr, float* __restrict__ bmm2_scale_ptr, int bmm1_is_log2, int num_q_heads, int num_kv_heads, int batch_size, unsigned int total_tiles)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define raw_kv_full_addr (mbar_base + 16)
    #define raw_kv_empty_addr (mbar_base + 48)
    #define kv_full_addr (mbar_base + 80)
    #define kv_empty_addr (mbar_base + 96)
    #define s_full_0_addr (mbar_base + 112)
    #define s_empty_0_addr (mbar_base + 128)
    #define p_full_0_addr (mbar_base + 144)
    #define corr_scale_0_addr (mbar_base + 160)
    #define corr_empty_0_addr (mbar_base + 176)
    #define o_full_addr (mbar_base + 192)
    #define o_empty_addr (mbar_base + 200)
    #define tmem_dealloc_addr (mbar_base + 208)
    #define work_full_addr (mbar_base + 216)
    #define work_empty_addr (mbar_base + 232)
    #define throttle_full_addr (mbar_base + 248)
    #define throttle_empty_addr (mbar_base + 264)
    #define page_offsets_full_addr (mbar_base + 280)
    #define page_offsets_empty_addr (mbar_base + 328)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Qt)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(V)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* smem_corr = reinterpret_cast<float*>(smem_raw + 1920);
    const int smem_corr_addr = smem + 1920;
    float* smem_exch = reinterpret_cast<float*>(smem_raw + 1088);
    const int smem_exch_addr = smem + 1088;
    unsigned int* smem_exch_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1088);
    const int smem_exch_u32_addr = smem + 1088;
    __nv_bfloat16* smem_qt_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_qt_hi_addr = smem + 2048;
    __nv_bfloat16* smem_qt_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4096);
    const int smem_qt_lo_addr = smem + 4096;
    uint8_t* smem_kv_fp8 = reinterpret_cast<uint8_t*>(smem_raw + 71680);
    const int smem_kv_fp8_addr = smem + 71680;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_kv_addr = smem + 6144;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 6144);
    const int smem_v_addr = smem + 6144;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 137216);
    const int smem_p_addr = smem + 137216;
    unsigned int* work_response_view = reinterpret_cast<unsigned int*>(smem_raw + 1344);
    const int work_response_view_addr = smem + 1344;
    int* split_reduce_flag = reinterpret_cast<int*>(smem_raw + 141312);
    const int split_reduce_flag_addr = smem + 141312;
    float* split_weights = reinterpret_cast<float*>(smem_raw + 1408);
    const int split_weights_addr = smem + 1408;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 141440);
    const int smem_page_offsets_addr = smem + 141440;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Qt)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory");

    // Mbarrier init (20 pipeline groups, 0 ordered-sequence groups, 47 barriers)
    // Mbarriers at smem_raw[0..376)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // raw_kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // raw_kv_empty: 4 barriers, init_count=4
            mbarrier_init(smem + 48, 4);
            mbarrier_init(smem + 56, 4);
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            // kv_full: 2 barriers, init_count=4
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            // kv_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'sm_pipe' ---
            // s_full_0: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // s_empty_0: 2 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            // --- pipeline 'p_pipe' ---
            // p_full_0: 2 barriers, init_count=256
            mbarrier_init(smem + 144, 256);
            mbarrier_init(smem + 152, 256);
            // --- pipeline 'corr_pipe' ---
            // corr_scale_0: 2 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            // corr_empty_0: 2 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // work_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 232, 512);
            mbarrier_init(smem + 240, 512);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 2 barriers, init_count=32
            mbarrier_init(smem + 248, 32);
            mbarrier_init(smem + 256, 32);
            // throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 264, 32);
            mbarrier_init(smem + 272, 32);
            // --- pipeline 'page_pipe' ---
            // page_offsets_full: 6 barriers, init_count=32
            mbarrier_init(smem + 280, 32);
            mbarrier_init(smem + 288, 32);
            mbarrier_init(smem + 296, 32);
            mbarrier_init(smem + 304, 32);
            mbarrier_init(smem + 312, 32);
            mbarrier_init(smem + 320, 32);
            // page_offsets_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 48 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 376);
    if (warp == 0) {
        int _tmem_hold = smem + 376;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_stats0 = taddr + 16;
    const int tmem_tmem_o_hi = taddr + 32;
    const int tmem_tmem_o_lo = taddr + 40;

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            const int tmem_row_base_v = warp * 32;
            int my_tmem_s = taddr;
            int my_tmem_stats = taddr + 16 + (unsigned int)(tmem_row_base_v << 16);
            const int warp_in_wg = warp;
            const int wg_tid = (unsigned int)(warp_in_wg * 32) + lane;
            int col_pair = wg_tid % 4;
            int col_pair_base = col_pair * 2;
            unsigned int work_stage_s = 0;
            int sm_stage = 0;
            int sm_phase = 0;
            int corr_prod_stage = 0;
            int corr_prod_phase = 1;
            float bmm1_scale_log2_s = softmax_scale_log2;
            float bmm1_scale_log2_p = softmax_scale_log2;
            int batch_idx = 0;
            int q_row_idx = 0;
            int kv_head_idx = 0;
            int split_idx = 0;
            int part_count = NUM_SPLIT;
            int bundle_idx = 0;
            int bundle_item_idx = 0;
            {
                q_row_idx = blockIdx.x;
                int head_split_idx = blockIdx.y;
                kv_head_idx = head_split_idx / NUM_SPLIT;
                split_idx = head_split_idx % NUM_SPLIT;
                int schedule_batch_idx = blockIdx.z;
                batch_idx = schedule_batch_idx;
            }
            int batch_idx_s = batch_idx;
            int q_row_idx_s = q_row_idx;
            int kv_head_idx_s = kv_head_idx;
            int split_idx_s = split_idx;
            int part_count_s = part_count;
            int bundle_idx_s = bundle_idx;
            int bundle_item_idx_s = bundle_item_idx;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < total_tiles; _tile_iter_s++) {
                int visible_keys = UNIFORM_KV_LEN - Q_LEN + q_row_idx_s + 1;
                {
                    visible_keys = seq_lens_kv[batch_idx_s] - Q_LEN + q_row_idx_s + 1;
                }
                if (visible_keys < 0) {
                    visible_keys = 0;
                }
                int seqlen_kv_s = visible_keys;
                int num_n_blocks = (seqlen_kv_s + BLOCK_N - 1) / BLOCK_N;
                if (num_n_blocks < 1) {
                    num_n_blocks = 1;
                }
                int total_pairs = (num_n_blocks + 1) / 2;
                int base_pairs = 0;
                int extra_pairs = 0;
                {
                    base_pairs = total_pairs / NUM_SPLIT;
                    extra_pairs = total_pairs % NUM_SPLIT;
                }
                int num_pairs = base_pairs;
                int split_start_pair = extra_pairs * (base_pairs + 1) + (split_idx_s - extra_pairs) * base_pairs;
                if (split_idx_s < extra_pairs) {
                    num_pairs = base_pairs + 1;
                    split_start_pair = split_idx_s * (base_pairs + 1);
                }
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
                    smem_exch_u32[wg_tid] = _amf_enc_0;
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                asm volatile("griddepcontrol.wait;" ::: "memory");
                {
                    bmm1_scale_log2_s = bmm1_scale_ptr[0];
                    if (bmm1_is_log2 == 0) {
                        bmm1_scale_log2_s = bmm1_scale_log2_s * 1.4426950408889634f;
                    }
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                {
                    bmm1_scale_log2_p = bmm1_scale_ptr[0];
                    if (bmm1_is_log2 == 0) {
                        bmm1_scale_log2_p = bmm1_scale_log2_p * 1.4426950408889634f;
                    }
                }
                #pragma unroll 1
                for (int n = 0; n < num_pairs * 2; n++) {
                    mbarrier_wait(s_full_0_addr + (sm_stage) * 8, sm_phase);
                    float sv[8];
                    float sv_lo[4];
                    float sv_hi[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[3]))
                        : "r"(my_tmem_s + sm_stage * 8));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[3]))
                        : "r"(my_tmem_s + sm_stage * 8 + 1048576));
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        sv[c] = sv_lo[c];
                        sv[c + 4] = sv_hi[c];
                    }
                    int my_block = split_start_pair * 2 + n;
                    int ldtm_row_base = (unsigned int)(warp_in_wg * 32) + lane / 4;
                    int kv_pos0 = my_block * BLOCK_N + ldtm_row_base;
                    int kv_pos1 = kv_pos0 + 8;
                    int kv_pos2 = kv_pos0 + 16;
                    int kv_pos3 = kv_pos0 + 24;
                    if (kv_pos0 >= seqlen_kv_s) {
                        sv[0] = -3.4028235e+38f;
                        sv[1] = -3.4028235e+38f;
                    }
                    if (kv_pos1 >= seqlen_kv_s) {
                        sv[2] = -3.4028235e+38f;
                        sv[3] = -3.4028235e+38f;
                    }
                    if (kv_pos2 >= seqlen_kv_s) {
                        sv[4] = -3.4028235e+38f;
                        sv[5] = -3.4028235e+38f;
                    }
                    if (kv_pos3 >= seqlen_kv_s) {
                        sv[6] = -3.4028235e+38f;
                        sv[7] = -3.4028235e+38f;
                    }
                    float pair_max[2];
                    pair_max[0] = -3.4028235e+38f;
                    pair_max[1] = -3.4028235e+38f;
                    float _max_0 = max_noftz(pair_max[0], sv[0]);
                    pair_max[0] = _max_0;
                    float _max_1 = max_noftz(pair_max[0], sv[2]);
                    pair_max[0] = _max_1;
                    float _max_2 = max_noftz(pair_max[1], sv[1]);
                    pair_max[1] = _max_2;
                    float _max_3 = max_noftz(pair_max[1], sv[3]);
                    pair_max[1] = _max_3;
                    float _max_4 = max_noftz(pair_max[0], sv[4]);
                    pair_max[0] = _max_4;
                    float _max_5 = max_noftz(pair_max[0], sv[6]);
                    pair_max[0] = _max_5;
                    float _max_6 = max_noftz(pair_max[1], sv[5]);
                    pair_max[1] = _max_6;
                    float _max_7 = max_noftz(pair_max[1], sv[7]);
                    pair_max[1] = _max_7;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 2; c_1++) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 16);
                        float _max_8 = max_noftz(pair_max[c_1], _shfl_xor_0);
                        pair_max[c_1] = _max_8;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 8);
                        float _max_9 = max_noftz(pair_max[c_1], _shfl_xor_1);
                        pair_max[c_1] = _max_9;
                    }
                    float new_max_pair[2];
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 2; c_2++) {
                        float _max_10 = max_noftz(row_max_pair[c_2], pair_max[c_2]);
                        new_max_pair[c_2] = _max_10;
                    }
                    if (lane < 8) {
                        uint32_t _amf_u_1 = __float_as_uint(new_max_pair[0]);
                        uint32_t _amf_mask_1 = -int32_t(_amf_u_1 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_1 = _amf_u_1 ^ _amf_mask_1;
                        uint32_t _amf_u_2 = __float_as_uint(new_max_pair[1]);
                        uint32_t _amf_mask_2 = -int32_t(_amf_u_2 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_2 = _amf_u_2 ^ _amf_mask_2;
                        atomicMax(&smem_exch_u32[col_pair_base], _amf_enc_1);
                        atomicMax(&smem_exch_u32[col_pair_base + 1], _amf_enc_2);
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    uint32_t _amf_u_3 = smem_exch_u32[col_pair_base];
                    uint32_t _amf_mask_3 = ((_amf_u_3 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_3 ^ _amf_mask_3);
                    new_max_pair[0] = _amf_dec_0;
                    uint32_t _amf_u_4 = smem_exch_u32[col_pair_base + 1];
                    uint32_t _amf_mask_4 = ((_amf_u_4 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_1 = __uint_as_float(_amf_u_4 ^ _amf_mask_4);
                    new_max_pair[1] = _amf_dec_1;
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    float acc_scale_pair[2];
                    #pragma unroll
                    for (int c_3 = 0; c_3 < 2; c_3++) {
                        float delta = bmm1_scale_log2_s * (row_max_pair[c_3] - new_max_pair[c_3]);
                        float _exp2_0 = approx_exp2(delta);
                        acc_scale_pair[c_3] = ((row_max_pair[c_3] > -CAKE_INF) ? _exp2_0 : 1.0f);
                    }
                    mbarrier_wait(corr_empty_0_addr + (corr_prod_stage) * 8, corr_prod_phase);
                    float acc_scale[8];
                    #pragma unroll
                    for (int cp = 0; cp < 4; cp++) {
                        float _shfl_0 = __shfl_sync(0xFFFFFFFF, acc_scale_pair[0], cp);
                        acc_scale[cp * 2] = _shfl_0;
                        float _shfl_1 = __shfl_sync(0xFFFFFFFF, acc_scale_pair[1], cp);
                        acc_scale[cp * 2 + 1] = _shfl_1;
                    }
                    tmem_st_x8_f32(my_tmem_stats + corr_prod_stage * 8, acc_scale);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(corr_scale_0_addr + (corr_prod_stage) * 8);
                    corr_prod_stage += 1;
                    if (corr_prod_stage == 2) { corr_prod_stage = 0; corr_prod_phase ^= 1; }
                    float exp_vals[8];
                    #pragma unroll
                    for (int c_4 = 0; c_4 < 8; c_4++) {
                        const int pair_c = c_4 % 2;
                        float safe_max = ((new_max_pair[pair_c] == -CAKE_INF) ? 0.0f : new_max_pair[pair_c]);
                        float max_scaled = safe_max * bmm1_scale_log2_p;
                        float _exp2_1 = approx_exp2(sv[c_4] * bmm1_scale_log2_p - max_scaled);
                        exp_vals[c_4] = _exp2_1;
                    }
                    #pragma unroll
                    for (int c_5 = 0; c_5 < 2; c_5++) {
                        row_max_pair[c_5] = new_max_pair[c_5];
                    }
                    float pair_sum[2];
                    #pragma unroll
                    for (int c_6 = 0; c_6 < 2; c_6++) {
                        float _fma_0 = __fmaf_rn(row_sum_pair[c_6], acc_scale_pair[c_6], exp_vals[c_6]);
                        pair_sum[c_6] = _fma_0;
                        pair_sum[c_6] = pair_sum[c_6] + exp_vals[c_6 + 2];
                        pair_sum[c_6] = pair_sum[c_6] + exp_vals[c_6 + 4];
                        pair_sum[c_6] = pair_sum[c_6] + exp_vals[c_6 + 6];
                        row_sum_pair[c_6] = pair_sum[c_6];
                    }
                    {
                        #pragma unroll
                        for (int r = 0; r < 4; r++) {
                            #pragma unroll
                            for (int c_7 = 0; c_7 < 2; c_7++) {
                                {
                                    __nv_bfloat16 _bval_5 = __float2bfloat16_rn(exp_vals[r * 2 + c_7]);
                                    uint16_t _bits_5 = *(uint16_t*)&_bval_5;
                                    uint32_t _addr_5 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(sm_stage * 2048) + (unsigned int)((ldtm_row_base + r * 8) / 64 * 1024 + (col_pair_base + c_7) * 128 + (ldtm_row_base + r * 8) % 64 * 2 ^ ((ldtm_row_base + r * 8) / 64 * 1024 + (col_pair_base + c_7) * 128 + (ldtm_row_base + r * 8) % 64 * 2 >> 7 & 7) << 4)));
                                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_5), "h"(_bits_5) : "memory");
                                }
                            }
                        }
                    }
                    asm volatile("fence.proxy.async;");
                    {
                        mbarrier_arrive(p_full_0_addr + (sm_stage) * 8);
                    }
                    mbarrier_arrive(s_empty_0_addr + (sm_stage) * 8);
                    sm_stage += 1;
                    if (sm_stage == 2) { sm_stage = 0; sm_phase ^= 1; }
                }
                mbarrier_wait(corr_empty_0_addr + (corr_prod_stage) * 8, corr_prod_phase);
                float final_stats_pair[4];
                final_stats_pair[0] = row_sum_pair[0];
                final_stats_pair[1] = row_sum_pair[1];
                final_stats_pair[2] = row_max_pair[0];
                final_stats_pair[3] = row_max_pair[1];
                tmem_st_x4_f32(my_tmem_stats + corr_prod_stage * 8, final_stats_pair);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(corr_scale_0_addr + (corr_prod_stage) * 8);
                corr_prod_stage += 1;
                if (corr_prod_stage == 2) { corr_prod_stage = 0; corr_prod_phase ^= 1; }
                {
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            const int warp_in_wg_c = warp % 4;
            const int corr_tid = (unsigned int)(warp_in_wg_c * 32) + lane;
            const int col_pair_c = corr_tid % 4;
            const int col_pair_base_c = col_pair_c * 2;
            unsigned int work_stage_c = 0;
            int corr_cons_stage = 0;
            int corr_cons_phase = 0;
            int p_stage_c = 0;
            int d_idx = warp % 4 * 32 + lane;
            int group_ratio_rt = num_q_heads / num_kv_heads;
            float bmm1_scale_log2_c = softmax_scale_log2;
            float bmm2_scale_c = output_scale;
            int batch_idx_1 = 0;
            int q_row_idx_1 = 0;
            int kv_head_idx_1 = 0;
            int split_idx_1 = 0;
            int part_count_1 = NUM_SPLIT;
            int bundle_idx_1 = 0;
            int bundle_item_idx_1 = 0;
            {
                q_row_idx_1 = blockIdx.x;
                int head_split_idx_1 = blockIdx.y;
                kv_head_idx_1 = head_split_idx_1 / NUM_SPLIT;
                split_idx_1 = head_split_idx_1 % NUM_SPLIT;
                int schedule_batch_idx_1 = blockIdx.z;
                batch_idx_1 = schedule_batch_idx_1;
            }
            int batch_idx_c = batch_idx_1;
            int q_row_idx_c = q_row_idx_1;
            int kv_head_idx_c = kv_head_idx_1;
            int split_idx_c = split_idx_1;
            int part_count_c = part_count_1;
            int bundle_idx_c = bundle_idx_1;
            int bundle_item_idx_c = bundle_item_idx_1;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < total_tiles; _tile_iter_c++) {
                int visible_keys_1 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_c + 1;
                {
                    visible_keys_1 = seq_lens_kv[batch_idx_c] - Q_LEN + q_row_idx_c + 1;
                }
                if (visible_keys_1 < 0) {
                    visible_keys_1 = 0;
                }
                int seqlen_kv_c = visible_keys_1;
                int num_n_blocks_1 = (seqlen_kv_c + BLOCK_N - 1) / BLOCK_N;
                if (num_n_blocks_1 < 1) {
                    num_n_blocks_1 = 1;
                }
                int total_pairs_1 = (num_n_blocks_1 + 1) / 2;
                int base_pairs_1 = 0;
                int extra_pairs_1 = 0;
                {
                    base_pairs_1 = total_pairs_1 / NUM_SPLIT;
                    extra_pairs_1 = total_pairs_1 % NUM_SPLIT;
                }
                int num_pairs_1 = base_pairs_1;
                int split_start_pair_1 = extra_pairs_1 * (base_pairs_1 + 1) + (split_idx_c - extra_pairs_1) * base_pairs_1;
                if (split_idx_c < extra_pairs_1) {
                    num_pairs_1 = base_pairs_1 + 1;
                    split_start_pair_1 = split_idx_c * (base_pairs_1 + 1);
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                {
                    bmm1_scale_log2_c = bmm1_scale_ptr[0];
                    if (bmm1_is_log2 == 0) {
                        bmm1_scale_log2_c = bmm1_scale_log2_c * 1.4426950408889634f;
                    }
                    bmm2_scale_c = bmm2_scale_ptr[0];
                }
                #pragma unroll 1
                for (int _n = 0; _n < num_pairs_1 * 2; _n++) {
                    mbarrier_wait(corr_scale_0_addr + (corr_cons_stage) * 8, corr_cons_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], taddr + 16 + (unsigned int)(corr_cons_stage * 8) + (unsigned int)corr_row);
                    if (_n > 0) {
                        mbarrier_wait(o_full_addr, _phase_o_full_0);
                        _phase_o_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int needs_corr_pred = 0;
                        #pragma unroll
                        for (int h = 0; h < 8; h++) {
                            needs_corr_pred = needs_corr_pred | ((_tmem_load_0[h] != 1.0f) ? 1 : 0);
                        }
                        int _vote_0 = __any_sync(0xFFFFFFFF, needs_corr_pred != 0);
                        if (_vote_0 != 0) {
                            float _tmem_load_1[8];
                            tmem_ld_x8(&_tmem_load_1[0], taddr + 32 + (unsigned int)corr_row);
                            float _tmem_load_2[8];
                            tmem_ld_x8(&_tmem_load_2[0], taddr + 40 + (unsigned int)corr_row);
                            #pragma unroll
                            for (int h_1 = 0; h_1 < 8; h_1++) {
                                _tmem_load_1[h_1] = _tmem_load_1[h_1] * _tmem_load_0[h_1];
                                _tmem_load_2[h_1] = _tmem_load_2[h_1] * _tmem_load_0[h_1];
                            }
                            tmem_st_x8_f32(taddr + 32 + (unsigned int)corr_row, _tmem_load_1);
                            tmem_st_x8_f32(taddr + 40 + (unsigned int)corr_row, _tmem_load_2);
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        mbarrier_arrive(o_empty_addr);
                    }
                    {
                        mbarrier_arrive(p_full_0_addr + (p_stage_c) * 8);
                        p_stage_c += 1;
                        if (p_stage_c == 2) { p_stage_c = 0; }
                    }
                    mbarrier_arrive(corr_empty_0_addr + (corr_cons_stage) * 8);
                    corr_cons_stage += 1;
                    if (corr_cons_stage == 2) { corr_cons_stage = 0; corr_cons_phase ^= 1; }
                }
                mbarrier_wait(corr_scale_0_addr + (corr_cons_stage) * 8, corr_cons_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_3[4];
                tmem_ld_x4(&_tmem_load_3[0], taddr + 16 + (unsigned int)(corr_cons_stage * 8) + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                mbarrier_arrive(corr_empty_0_addr + (corr_cons_stage) * 8);
                corr_cons_stage += 1;
                if (corr_cons_stage == 2) { corr_cons_stage = 0; corr_cons_phase ^= 1; }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float reduced_sum_pair[2];
                #pragma unroll
                for (int c_8 = 0; c_8 < 2; c_8++) {
                    reduced_sum_pair[c_8] = _tmem_load_3[c_8];
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, reduced_sum_pair[c_8], 16);
                    reduced_sum_pair[c_8] = reduced_sum_pair[c_8] + _shfl_xor_2;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, reduced_sum_pair[c_8], 8);
                    reduced_sum_pair[c_8] = reduced_sum_pair[c_8] + _shfl_xor_3;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, reduced_sum_pair[c_8], 4);
                    reduced_sum_pair[c_8] = reduced_sum_pair[c_8] + _shfl_xor_4;
                }
                if (lane < 4) {
                    const int warp_sum_base_c = warp_in_wg_c * 8 + col_pair_base_c;
                    smem_corr[warp_sum_base_c] = reduced_sum_pair[0];
                    smem_corr[warp_sum_base_c + 1] = reduced_sum_pair[1];
                }
                asm volatile("barrier.sync 9, 128;" ::: "memory");
                float total_sum_pair[2];
                #pragma unroll
                for (int c_9 = 0; c_9 < 2; c_9++) {
                    total_sum_pair[c_9] = smem_corr[col_pair_base_c + c_9] + smem_corr[8 + col_pair_base_c + c_9];
                    total_sum_pair[c_9] = total_sum_pair[c_9] + smem_corr[16 + col_pair_base_c + c_9];
                    total_sum_pair[c_9] = total_sum_pair[c_9] + smem_corr[24 + col_pair_base_c + c_9];
                }
                if (corr_tid < 8) {
                    int stats_head_c = col_pair_base_c;
                    float stats_sum_c = total_sum_pair[0];
                    float stats_max_c = _tmem_load_3[2];
                    if (lane >= 4) {
                        stats_head_c = col_pair_base_c + 1;
                        stats_sum_c = total_sum_pair[1];
                        stats_max_c = _tmem_load_3[3];
                    }
                    if (stats_head_c < group_ratio_rt) {
                        int stats_q_head_c = kv_head_idx_c * group_ratio_rt + stats_head_c;
                        int stats_output_row_c = (batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + stats_q_head_c;
                        {
                            {
                            }
                        }
                    }
                }
                float total_sum[8];
                float total_max[8];
                #pragma unroll
                for (int cp_1 = 0; cp_1 < 4; cp_1++) {
                    float _shfl_2 = __shfl_sync(0xFFFFFFFF, total_sum_pair[0], cp_1);
                    total_sum[cp_1 * 2] = _shfl_2;
                    float _shfl_3 = __shfl_sync(0xFFFFFFFF, total_sum_pair[1], cp_1);
                    total_sum[cp_1 * 2 + 1] = _shfl_3;
                    float _shfl_4 = __shfl_sync(0xFFFFFFFF, _tmem_load_3[2], cp_1);
                    total_max[cp_1 * 2] = _shfl_4;
                    float _shfl_5 = __shfl_sync(0xFFFFFFFF, _tmem_load_3[3], cp_1);
                    total_max[cp_1 * 2 + 1] = _shfl_5;
                }
                float _tmem_load_4[8];
                tmem_ld_x8(&_tmem_load_4[0], taddr + 32 + (unsigned int)corr_row);
                float _tmem_load_5[8];
                tmem_ld_x8(&_tmem_load_5[0], taddr + 40 + (unsigned int)corr_row);
                #pragma unroll
                for (int h_2 = 0; h_2 < 8; h_2++) {
                    float _rcp_0 = approx_rcp(total_sum[h_2]);
                    float inv_total = _rcp_0;
                    float final_o_hi = _tmem_load_4[h_2] * inv_total * bmm2_scale_c;
                    float final_o_lo = _tmem_load_5[h_2] * inv_total * bmm2_scale_c;
                    if (group_ratio_rt > h_2) {
                        int q_head = kv_head_idx_c * group_ratio_rt + h_2;
                        int output_row = (batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + q_head;
                        {
                            {
                                int out_idx_hi = output_row * HEAD_DIM + d_idx;
                                int out_idx_lo = out_idx_hi + HEAD_DIM_HALF;
                                *(reinterpret_cast<__nv_bfloat16*>(O + out_idx_hi) + (0)) = __float2bfloat16_rn(final_o_hi);
                                *(reinterpret_cast<__nv_bfloat16*>(O + out_idx_lo) + (0)) = __float2bfloat16_rn(final_o_lo);
                            }
                        }
                    }
                }
                mbarrier_arrive(o_empty_addr);
                if (USE_SEGMENTED_CLC != 0 && part_count_c > 1) {
                    int base_tile_idx_seg = (batch_idx_c * Q_LEN + q_row_idx_c) * num_kv_heads + kv_head_idx_c;
                    __threadfence();
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (d_idx == 0) {
                        uint32_t _atomic_inc_old_0;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_inc_old_0) : "l"(&split_completion[base_tile_idx_seg]), "r"(static_cast<uint32_t>(part_count_c - 1)) : "memory");
                        unsigned int old_count_seg = _atomic_inc_old_0;
                        split_reduce_flag[0] = (((int)old_count_seg + 1 == part_count_c) ? 1 : 0);
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (split_reduce_flag[0] != 0) {
                        __threadfence();
                        int reduce_head_seg = d_idx / 8;
                        int reduce_lane_seg = d_idx % 8;
                        int reduce_head_valid_seg = 0;
                        if (reduce_head_seg < group_ratio_rt) {
                            if (reduce_head_seg < TILE_Q) {
                                reduce_head_valid_seg = 1;
                            }
                        }
                        int reduce_q_head_seg = kv_head_idx_c * group_ratio_rt + reduce_head_seg;
                        int reduce_stat_base_seg = ((batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + reduce_q_head_seg) * 16;
                        int split0_seg = reduce_lane_seg;
                        int split1_seg = reduce_lane_seg + 8;
                        float lse0_seg = -CAKE_INF;
                        float lse1_seg = -CAKE_INF;
                        if (reduce_head_valid_seg != 0) {
                            if (split0_seg < part_count_c) {
                                lse0_seg = partial_LSE[reduce_stat_base_seg + split0_seg];
                            }
                            if (split1_seg < part_count_c) {
                                lse1_seg = partial_LSE[reduce_stat_base_seg + split1_seg];
                            }
                        }
                        float _max_11 = max_noftz(lse0_seg, lse1_seg);
                        float lane_max_seg = _max_11;
                        int subgroup_lane_base_seg = lane / 8 * 8;
                        float merged_max_seg = -CAKE_INF;
                        #pragma unroll
                        for (int source_lane_seg = 0; source_lane_seg < 8; source_lane_seg++) {
                            float _shfl_6 = __shfl_sync(0xFFFFFFFF, lane_max_seg, subgroup_lane_base_seg + source_lane_seg);
                            float source_max_seg = _shfl_6;
                            float _max_12 = max_noftz(merged_max_seg, source_max_seg);
                            merged_max_seg = _max_12;
                        }
                        float weight0_seg = 0.0f;
                        float weight1_seg = 0.0f;
                        if (lse0_seg != -CAKE_INF) {
                            float _exp2_2 = approx_exp2(lse0_seg - merged_max_seg);
                            weight0_seg = _exp2_2;
                        }
                        if (lse1_seg != -CAKE_INF) {
                            float _exp2_3 = approx_exp2(lse1_seg - merged_max_seg);
                            weight1_seg = _exp2_3;
                        }
                        float lane_weight_sum_seg = weight0_seg + weight1_seg;
                        float weight_sum_seg = 0.0f;
                        #pragma unroll
                        for (int source_lane_seg_1 = 0; source_lane_seg_1 < 8; source_lane_seg_1++) {
                            float _shfl_7 = __shfl_sync(0xFFFFFFFF, lane_weight_sum_seg, subgroup_lane_base_seg + source_lane_seg_1);
                            weight_sum_seg = weight_sum_seg + _shfl_7;
                        }
                        float _rcp_1 = approx_rcp(weight_sum_seg);
                        float inv_weight_sum_seg = ((weight_sum_seg > 0.0f) ? _rcp_1 : 0.0f);
                        if (reduce_head_valid_seg != 0) {
                            if (split0_seg < part_count_c) {
                                split_weights[split0_seg * TILE_Q + reduce_head_seg] = weight0_seg * inv_weight_sum_seg;
                            }
                            if (split1_seg < part_count_c) {
                                split_weights[split1_seg * TILE_Q + reduce_head_seg] = weight1_seg * inv_weight_sum_seg;
                            }
                            if (reduce_lane_seg == 0) {
                            }
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        int merge_head_seg = d_idx / 8;
                        int merge_d_base_seg = d_idx % 8 * 32;
                        int merge_head_valid_seg = 0;
                        if (merge_head_seg < group_ratio_rt) {
                            if (merge_head_seg < TILE_Q) {
                                merge_head_valid_seg = 1;
                            }
                        }
                        if (merge_head_valid_seg != 0) {
                            int merge_q_head_seg = kv_head_idx_c * group_ratio_rt + merge_head_seg;
                            #pragma unroll
                            for (int vec_chunk_seg = 0; vec_chunk_seg < 4; vec_chunk_seg++) {
                                int elem_base_seg = merge_d_base_seg + vec_chunk_seg * 8;
                                int partial_o_base_seg = ((batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + merge_q_head_seg) * 16 * HEAD_DIM + elem_base_seg;
                                int final_o_idx_seg = ((batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + merge_q_head_seg) * HEAD_DIM + elem_base_seg;
                                float _vec_load_0[8];
                                {
                                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_O + partial_o_base_seg + 0);
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
                                float merge_weight0_seg = split_weights[merge_head_seg];
                                #pragma unroll
                                for (int elem_seg = 0; elem_seg < 8; elem_seg++) {
                                    _vec_load_0[elem_seg] = _vec_load_0[elem_seg] * merge_weight0_seg;
                                }
                                #pragma unroll 2
                                for (int reduce_part_seg = 1; reduce_part_seg < part_count_c; reduce_part_seg++) {
                                    float _vec_load_1[8];
                                    {
                                        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(partial_O + (partial_o_base_seg + reduce_part_seg * HEAD_DIM) + 0);
                                        uint4 _vld_1[1];
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vld_1[_blk] = _vptr_1[_blk];
                                            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                                    : "r"(_vpairs_1[_pair]));
                                            }
                                        }
                                    }
                                    float reduce_weight_seg = split_weights[reduce_part_seg * TILE_Q + merge_head_seg];
                                    #pragma unroll
                                    for (int elem_seg_1 = 0; elem_seg_1 < 8; elem_seg_1++) {
                                        float _fma_1 = __fmaf_rn(_vec_load_1[elem_seg_1], reduce_weight_seg, _vec_load_0[elem_seg_1]);
                                        _vec_load_0[elem_seg_1] = _fma_1;
                                    }
                                }
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_vec_load_0[0 + 0], _vec_load_0[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_vec_load_0[0 + 2], _vec_load_0[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_vec_load_0[0 + 4], _vec_load_0[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_vec_load_0[0 + 6], _vec_load_0[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + final_o_idx_seg))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    }
                } else if (USE_SEGMENTED_CLC == 0 && NUM_SPLIT > 1) {
                    int base_tile_idx = (batch_idx_c * Q_LEN + q_row_idx_c) * num_kv_heads + kv_head_idx_c;
                    __threadfence();
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (d_idx == 0) {
                        uint32_t _atomic_inc_old_1;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_inc_old_1) : "l"(&split_completion[base_tile_idx]), "r"(static_cast<uint32_t>(NUM_SPLIT - 1)) : "memory");
                        unsigned int old_count = _atomic_inc_old_1;
                        {
                            split_reduce_flag[0] = (((int)old_count + 1 == NUM_SPLIT) ? 1 : 0);
                        }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (split_reduce_flag[0] != 0) {
                        __threadfence();
                        int reduce_head = d_idx / 8;
                        int reduce_lane = d_idx % 8;
                        int reduce_head_valid = 0;
                        {
                            if (reduce_head < group_ratio_rt) {
                                if (reduce_head < TILE_Q) {
                                    reduce_head_valid = 1;
                                }
                            }
                        }
                        int reduce_q_head = kv_head_idx_c * group_ratio_rt + reduce_head;
                        int reduce_stat_base = ((batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + reduce_q_head) * NUM_SPLIT;
                        int split0 = reduce_lane;
                        int split1 = reduce_lane + 8;
                        float lse0 = -CAKE_INF;
                        float lse1 = -CAKE_INF;
                        if (reduce_head_valid != 0) {
                            if (split0 < NUM_SPLIT) {
                                lse0 = partial_LSE[reduce_stat_base + split0];
                            }
                            if (split1 < NUM_SPLIT) {
                                lse1 = partial_LSE[reduce_stat_base + split1];
                            }
                        }
                        float _max_13 = max_noftz(lse0, lse1);
                        float lane_max = _max_13;
                        int subgroup_lane_base = lane / 8 * 8;
                        float merged_max = -CAKE_INF;
                        #pragma unroll
                        for (int source_lane = 0; source_lane < 8; source_lane++) {
                            float _shfl_8 = __shfl_sync(0xFFFFFFFF, lane_max, subgroup_lane_base + source_lane);
                            float source_max = _shfl_8;
                            float _max_14 = max_noftz(merged_max, source_max);
                            merged_max = _max_14;
                        }
                        float weight0 = 0.0f;
                        float weight1 = 0.0f;
                        if (lse0 != -CAKE_INF) {
                            float _exp2_4 = approx_exp2(lse0 - merged_max);
                            weight0 = _exp2_4;
                        }
                        if (lse1 != -CAKE_INF) {
                            float _exp2_5 = approx_exp2(lse1 - merged_max);
                            weight1 = _exp2_5;
                        }
                        float lane_weight_sum = weight0 + weight1;
                        float weight_sum = 0.0f;
                        #pragma unroll
                        for (int source_lane_1 = 0; source_lane_1 < 8; source_lane_1++) {
                            float _shfl_9 = __shfl_sync(0xFFFFFFFF, lane_weight_sum, subgroup_lane_base + source_lane_1);
                            weight_sum = weight_sum + _shfl_9;
                        }
                        float _rcp_2 = approx_rcp(weight_sum);
                        float inv_weight_sum = ((weight_sum > 0.0f) ? _rcp_2 : 0.0f);
                        if (reduce_head_valid != 0) {
                            if (split0 < NUM_SPLIT) {
                                split_weights[split0 * TILE_Q + reduce_head] = weight0 * inv_weight_sum;
                            }
                            if (split1 < NUM_SPLIT) {
                                split_weights[split1 * TILE_Q + reduce_head] = weight1 * inv_weight_sum;
                            }
                            if (reduce_lane == 0) {
                            }
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        int merge_head = d_idx / 8;
                        int merge_d_base = d_idx % 8 * 32;
                        int merge_head_valid = 0;
                        {
                            if (merge_head < group_ratio_rt) {
                                if (merge_head < TILE_Q) {
                                    merge_head_valid = 1;
                                }
                            }
                        }
                        if (merge_head_valid != 0) {
                            int merge_q_head = kv_head_idx_c * group_ratio_rt + merge_head;
                            #pragma unroll
                            for (int vec_chunk = 0; vec_chunk < 4; vec_chunk++) {
                                int elem_base = merge_d_base + vec_chunk * 8;
                                int partial_o_base = ((batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + merge_q_head) * NUM_SPLIT * HEAD_DIM + elem_base;
                                int final_o_idx = ((batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + merge_q_head) * HEAD_DIM + elem_base;
                                float _vec_load_2[8];
                                {
                                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(partial_O + partial_o_base + 0);
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
                                                : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                                : "r"(_vpairs_2[_pair]));
                                        }
                                    }
                                }
                                float merge_weight0 = split_weights[merge_head];
                                #pragma unroll
                                for (int elem = 0; elem < 8; elem++) {
                                    _vec_load_2[elem] = _vec_load_2[elem] * merge_weight0;
                                }
                                #pragma unroll 2
                                for (int reduce_split = 1; reduce_split < NUM_SPLIT; reduce_split++) {
                                    float _vec_load_3[8];
                                    {
                                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(partial_O + (partial_o_base + reduce_split * HEAD_DIM) + 0);
                                        uint4 _vld_3[1];
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vld_3[_blk] = _vptr_3[_blk];
                                            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                                    : "r"(_vpairs_3[_pair]));
                                            }
                                        }
                                    }
                                    float reduce_weight = split_weights[reduce_split * TILE_Q + merge_head];
                                    #pragma unroll
                                    for (int elem_1 = 0; elem_1 < 8; elem_1++) {
                                        float _fma_2 = __fmaf_rn(_vec_load_3[elem_1], reduce_weight, _vec_load_2[elem_1]);
                                        _vec_load_2[elem_1] = _fma_2;
                                    }
                                }
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_vec_load_2[0 + 0], _vec_load_2[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_vec_load_2[0 + 2], _vec_load_2[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_vec_load_2[0 + 4], _vec_load_2[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_vec_load_2[0 + 6], _vec_load_2[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + final_o_idx))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    }
                }
                {
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // mma_warp_main
            const int tmem_s0v = taddr;
            const int tmem_o_hi_v = taddr + 32;
            const int tmem_o_lo_v = taddr + 40;
            unsigned int work_stage_m = 0;
            int sm_stage_1 = 0;
            int transformed_stage = 0;
            int transformed_phase = 0;
            int q_phase_m = 0;
            int sm_empty_phase_m = 1;
            int p_stage_m = 0;
            int p_phase_m = 0;
            mbarrier_wait(s_empty_0_addr, sm_empty_phase_m);
            mbarrier_wait(s_empty_0_addr + 8, sm_empty_phase_m);
            int batch_idx_2 = 0;
            int q_row_idx_2 = 0;
            int kv_head_idx_2 = 0;
            int split_idx_2 = 0;
            int part_count_2 = NUM_SPLIT;
            int bundle_idx_2 = 0;
            int bundle_item_idx_2 = 0;
            {
                q_row_idx_2 = blockIdx.x;
                int head_split_idx_2 = blockIdx.y;
                kv_head_idx_2 = head_split_idx_2 / NUM_SPLIT;
                split_idx_2 = head_split_idx_2 % NUM_SPLIT;
                int schedule_batch_idx_2 = blockIdx.z;
                batch_idx_2 = schedule_batch_idx_2;
            }
            int batch_idx_m = batch_idx_2;
            int q_row_idx_m = q_row_idx_2;
            int kv_head_idx_m = kv_head_idx_2;
            int split_idx_m = split_idx_2;
            int part_count_m = part_count_2;
            int bundle_idx_m = bundle_idx_2;
            int bundle_item_idx_m = bundle_item_idx_2;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_p_full_0_0 = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < total_tiles; _tile_iter_m++) {
                int visible_keys_2 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_m + 1;
                {
                    visible_keys_2 = seq_lens_kv[batch_idx_m] - Q_LEN + q_row_idx_m + 1;
                }
                if (visible_keys_2 < 0) {
                    visible_keys_2 = 0;
                }
                int seqlen_kv_m = visible_keys_2;
                int num_n_blocks_2 = (seqlen_kv_m + BLOCK_N - 1) / BLOCK_N;
                if (num_n_blocks_2 < 1) {
                    num_n_blocks_2 = 1;
                }
                int total_pairs_2 = (num_n_blocks_2 + 1) / 2;
                int base_pairs_2 = 0;
                int extra_pairs_2 = 0;
                {
                    base_pairs_2 = total_pairs_2 / NUM_SPLIT;
                    extra_pairs_2 = total_pairs_2 % NUM_SPLIT;
                }
                int num_pairs_2 = base_pairs_2;
                int split_start_pair_2 = extra_pairs_2 * (base_pairs_2 + 1) + (split_idx_m - extra_pairs_2) * base_pairs_2;
                if (split_idx_m < extra_pairs_2) {
                    num_pairs_2 = base_pairs_2 + 1;
                    split_start_pair_2 = split_idx_m * (base_pairs_2 + 1);
                }
                int first_pv = 1;
                {
                    uint32_t _mbar_token_0 = mbarrier_try_wait(q_full_addr, q_phase_m);
                    mbarrier_wait_token(q_full_addr, q_phase_m, _mbar_token_0);
                    q_phase_m ^= 1;
                    mbarrier_wait(s_empty_0_addr + (sm_stage_1) * 8, sm_empty_phase_m);
                    uint32_t _mbar_token_1 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                    mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_1);
                    int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (transformed_stage) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform(((smem_qt_hi_addr) >> 4) & 0x3FFF);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134349968;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_s0 + (sm_stage_1 * 8))), "r"(0));
                    elect_commit(kv_empty_addr + (transformed_stage) * 8);
                    transformed_stage += 1;
                    if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                    uint32_t _mbar_token_2 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                    mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_2);
                    int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (transformed_stage) * 2048);
                    int _mma_b_lo_1 = make_warp_uniform(((smem_qt_lo_addr) >> 4) & 0x3FFF);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134349968;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_s0 + (sm_stage_1 * 8))), "r"(1));
                    elect_commit(s_full_0_addr + (sm_stage_1) * 8);
                    elect_commit(kv_empty_addr + (transformed_stage) * 8);
                    transformed_stage += 1;
                    if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                    sm_stage_1 += 1;
                    if (sm_stage_1 == 2) { sm_stage_1 = 0; sm_empty_phase_m ^= 1; }
                    #pragma unroll 1
                    for (int _body_n_m = 0; _body_n_m < num_pairs_2 * 2 - 1; _body_n_m++) {
                        mbarrier_wait(s_empty_0_addr + (sm_stage_1) * 8, sm_empty_phase_m);
                        uint32_t _mbar_token_3 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                        mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_3);
                        int _mma_a_lo_2 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (transformed_stage) * 2048);
                        int _mma_b_lo_2 = make_warp_uniform(((smem_qt_hi_addr) >> 4) & 0x3FFF);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134349968;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_s0 + (sm_stage_1 * 8))), "r"(0));
                        elect_commit(kv_empty_addr + (transformed_stage) * 8);
                        transformed_stage += 1;
                        if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                        uint32_t _mbar_token_4 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                        mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_4);
                        int _mma_a_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (transformed_stage) * 2048);
                        int _mma_b_lo_3 = make_warp_uniform(((smem_qt_lo_addr) >> 4) & 0x3FFF);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134349968;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_tmem_s0 + (sm_stage_1 * 8))), "r"(1));
                        elect_commit(s_full_0_addr + (sm_stage_1) * 8);
                        elect_commit(kv_empty_addr + (transformed_stage) * 8);
                        transformed_stage += 1;
                        if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                        sm_stage_1 += 1;
                        if (sm_stage_1 == 2) { sm_stage_1 = 0; sm_empty_phase_m ^= 1; }
                        mbarrier_wait(p_full_0_addr + (p_stage_m) * 8, p_phase_m);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                        _phase_o_empty_0 ^= 1;
                        int first_pv_flag = first_pv;
                        uint32_t _mbar_token_5 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                        mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_5);
                        int _mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (transformed_stage) * 2048);
                        int _mma_b_lo_4 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x400000) + (p_stage_m) * 128);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134382736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_tmem_o_hi), "r"(((first_pv_flag) ? 0 : 1)));
                        elect_commit(kv_empty_addr + (transformed_stage) * 8);
                        transformed_stage += 1;
                        if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                        uint32_t _mbar_token_6 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                        mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_6);
                        int _mma_a_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (transformed_stage) * 2048);
                        int _mma_b_lo_5 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x400000) + (p_stage_m) * 128);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134382736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"(tmem_tmem_o_lo), "r"(((first_pv_flag) ? 0 : 1)));
                        elect_commit(kv_empty_addr + (transformed_stage) * 8);
                        transformed_stage += 1;
                        if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                        elect_commit(o_full_addr);
                        first_pv = 0;
                        p_stage_m += 1;
                        if (p_stage_m == 2) { p_stage_m = 0; p_phase_m ^= 1; }
                    }
                    mbarrier_wait(s_empty_0_addr + (sm_stage_1) * 8, sm_empty_phase_m);
                    mbarrier_wait(s_empty_0_addr + (sm_stage_1 + 1) * 8, sm_empty_phase_m);
                    elect_commit(q_empty_addr);
                    mbarrier_wait(p_full_0_addr + (p_stage_m) * 8, p_phase_m);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                    _phase_o_empty_0 ^= 1;
                    int last_pv_init = first_pv;
                    uint32_t _mbar_token_7 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                    mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_7);
                    int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (transformed_stage) * 2048);
                    int _mma_b_lo_6 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x400000) + (p_stage_m) * 128);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134382736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_tmem_o_hi), "r"(((last_pv_init) ? 0 : 1)));
                    elect_commit(kv_empty_addr + (transformed_stage) * 8);
                    transformed_stage += 1;
                    if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                    uint32_t _mbar_token_8 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                    mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_8);
                    int _mma_a_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (transformed_stage) * 2048);
                    int _mma_b_lo_7 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x400000) + (p_stage_m) * 128);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 134382736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 58;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_tmem_o_lo), "r"(((last_pv_init) ? 0 : 1)));
                    elect_commit(kv_empty_addr + (transformed_stage) * 8);
                    transformed_stage += 1;
                    if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                    elect_commit(o_full_addr);
                    p_stage_m += 1;
                    if (p_stage_m == 2) { p_stage_m = 0; p_phase_m ^= 1; }
                }
                {
                }
            }
            elect_commit(s_full_0_addr + (sm_stage_1) * 8);
            sm_stage_1 += 1;
            if (sm_stage_1 == 2) { sm_stage_1 = 0; sm_empty_phase_m ^= 1; }
            elect_commit(s_full_0_addr + (sm_stage_1) * 8);
            sm_stage_1 += 1;
            if (sm_stage_1 == 2) { sm_stage_1 = 0; sm_empty_phase_m ^= 1; }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(64));
        }
    }
    // ---- Role: page_offsets ----
    if (warp == 9) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // page_offsets_main
            unsigned int work_stage_p = 0;
            int page_prod_stage = 0;
            int page_prod_phase = 1;
            int batch_idx_3 = 0;
            int q_row_idx_3 = 0;
            int kv_head_idx_3 = 0;
            int split_idx_3 = 0;
            int part_count_3 = NUM_SPLIT;
            int bundle_idx_3 = 0;
            int bundle_item_idx_3 = 0;
            {
                q_row_idx_3 = blockIdx.x;
                int head_split_idx_3 = blockIdx.y;
                kv_head_idx_3 = head_split_idx_3 / NUM_SPLIT;
                split_idx_3 = head_split_idx_3 % NUM_SPLIT;
                int schedule_batch_idx_3 = blockIdx.z;
                batch_idx_3 = schedule_batch_idx_3;
            }
            int batch_idx_p = batch_idx_3;
            int q_row_idx_p = q_row_idx_3;
            int kv_head_idx_p = kv_head_idx_3;
            int split_idx_p = split_idx_3;
            int part_count_p = part_count_3;
            int bundle_idx_p = bundle_idx_3;
            int bundle_item_idx_p = bundle_item_idx_3;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < total_tiles; _tile_iter_p++) {
                int visible_keys_3 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_p + 1;
                {
                    visible_keys_3 = seq_lens_kv[batch_idx_p] - Q_LEN + q_row_idx_p + 1;
                }
                if (visible_keys_3 < 0) {
                    visible_keys_3 = 0;
                }
                int seqlen_kv_p = visible_keys_3;
                int num_n_blocks_3 = (seqlen_kv_p + BLOCK_N - 1) / BLOCK_N;
                if (num_n_blocks_3 < 1) {
                    num_n_blocks_3 = 1;
                }
                int total_pairs_3 = (num_n_blocks_3 + 1) / 2;
                int base_pairs_3 = 0;
                int extra_pairs_3 = 0;
                {
                    base_pairs_3 = total_pairs_3 / NUM_SPLIT;
                    extra_pairs_3 = total_pairs_3 % NUM_SPLIT;
                }
                int num_pairs_3 = base_pairs_3;
                int split_start_pair_3 = extra_pairs_3 * (base_pairs_3 + 1) + (split_idx_p - extra_pairs_3) * base_pairs_3;
                if (split_idx_p < extra_pairs_3) {
                    num_pairs_3 = base_pairs_3 + 1;
                    split_start_pair_3 = split_idx_p * (base_pairs_3 + 1);
                }
                int pages_per_seq_p = (seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE;
                int max_page_p = pages_per_seq_p - 1;
                int pt_base_p = batch_idx_p * max_pages_per_seq;
                int pt_base_v_p = pt_base_p + page_table_v_offset;
                {
                    {
                        #pragma unroll 1
                        for (int n_p = 0; n_p < num_pairs_3 * 2; n_p++) {
                            int n_block_p = split_start_pair_3 * 2 + n_p;
                            int logical_page_base_p = n_block_p * 2;
                            mbarrier_wait(page_offsets_empty_addr + (page_prod_stage) * 8, page_prod_phase);
                            if (elect_sync()) {
                                int page_smem_base_p = page_prod_stage * 4;
                                #pragma unroll
                                for (int page_in_block_p = 0; page_in_block_p < 2; page_in_block_p++) {
                                    int logical_page_p = logical_page_base_p + page_in_block_p;
                                    int clamped_page_p = ((logical_page_p > max_page_p) ? max_page_p : logical_page_p);
                                    smem_page_offsets[page_smem_base_p + page_in_block_p] = page_table[pt_base_p + clamped_page_p];
                                    smem_page_offsets[page_smem_base_p + 2 + page_in_block_p] = page_table[pt_base_v_p + clamped_page_p];
                                }
                            }
                            mbarrier_arrive(page_offsets_full_addr + (page_prod_stage) * 8);
                            page_prod_stage += 1;
                            if (page_prod_stage == 6) { page_prod_stage = 0; page_prod_phase ^= 1; }
                        }
                    }
                }
                {
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // scheduler_main
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            {
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp == 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // load_warp_main
            unsigned int work_stage_l = 0;
            unsigned int throttle_stage_l = 0;
            int raw_stage = 0;
            int raw_phase = 1;
            int page_cons_stage = 0;
            int page_cons_phase = 0;
            int page_release_stage = 0;
            int page_release_phase = 0;
            int batch_idx_4 = 0;
            int q_row_idx_4 = 0;
            int kv_head_idx_4 = 0;
            int split_idx_4 = 0;
            int part_count_4 = NUM_SPLIT;
            int bundle_idx_4 = 0;
            int bundle_item_idx_4 = 0;
            {
                q_row_idx_4 = blockIdx.x;
                int head_split_idx_4 = blockIdx.y;
                kv_head_idx_4 = head_split_idx_4 / NUM_SPLIT;
                split_idx_4 = head_split_idx_4 % NUM_SPLIT;
                int schedule_batch_idx_4 = blockIdx.z;
                batch_idx_4 = schedule_batch_idx_4;
            }
            int batch_idx_0 = batch_idx_4;
            int q_row_idx_1_1 = q_row_idx_4;
            int kv_head_idx_2_1 = kv_head_idx_4;
            int split_idx_l = split_idx_4;
            int part_count_l = part_count_4;
            int bundle_idx_l = bundle_idx_4;
            int bundle_item_idx_l = bundle_item_idx_4;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < total_tiles; _tile_iter_l++) {
                {
                }
                int visible_keys_4 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_1_1 + 1;
                {
                    visible_keys_4 = seq_lens_kv[batch_idx_0] - Q_LEN + q_row_idx_1_1 + 1;
                }
                if (visible_keys_4 < 0) {
                    visible_keys_4 = 0;
                }
                int seqlen_kv = visible_keys_4;
                int num_n_blocks_4 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                if (num_n_blocks_4 < 1) {
                    num_n_blocks_4 = 1;
                }
                int total_pairs_4 = (num_n_blocks_4 + 1) / 2;
                int base_pairs_4 = 0;
                int extra_pairs_4 = 0;
                {
                    base_pairs_4 = total_pairs_4 / NUM_SPLIT;
                    extra_pairs_4 = total_pairs_4 % NUM_SPLIT;
                }
                int num_pairs_4 = base_pairs_4;
                int split_start_pair_4 = extra_pairs_4 * (base_pairs_4 + 1) + (split_idx_l - extra_pairs_4) * base_pairs_4;
                if (split_idx_l < extra_pairs_4) {
                    num_pairs_4 = base_pairs_4 + 1;
                    split_start_pair_4 = split_idx_l * (base_pairs_4 + 1);
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    int group_ratio_l = num_q_heads / num_kv_heads;
                    int off_qt = (batch_idx_0 * Q_LEN + q_row_idx_1_1) * num_q_heads + kv_head_idx_2_1 * group_ratio_l;
                    mbarrier_arrive_expect_tx(q_full_addr, TILE_Q * HEAD_DIM * 2);
                    tma_3d_gmem2smem(smem_qt_hi_addr, Qt, 0, off_qt, 0, q_full_addr);
                    tma_3d_gmem2smem(smem_qt_lo_addr, Qt, 0, off_qt, 2, q_full_addr);
                    {
                        {
                            mbarrier_wait(page_offsets_full_addr + (page_cons_stage) * 8, page_cons_phase);
                            int page_smem_base = page_cons_stage * 4;
                            #pragma unroll
                            for (int dim_half = 0; dim_half < 2; dim_half++) {
                                mbarrier_wait(raw_kv_empty_addr + (raw_stage) * 8, raw_phase);
                                mbarrier_arrive_expect_tx(raw_kv_full_addr + (raw_stage) * 8, BLOCK_N * HEAD_DIM_HALF);
                                int raw_dst = smem_kv_fp8_addr + (unsigned int)(raw_stage * 16384);
                                #pragma unroll
                                for (int page_in_block = 0; page_in_block < 2; page_in_block++) {
                                    int physical_page = smem_page_offsets[page_smem_base + page_in_block];
                                    int page_dst = raw_dst + page_in_block * PAGE_SIZE * HEAD_DIM_HALF;
                                    {
                                        tma_5d_gmem2smem(page_dst, K, 0, 0, dim_half, kv_head_idx_2_1, physical_page, raw_kv_full_addr + (raw_stage) * 8);
                                    }
                                }
                                raw_stage += 1;
                                if (raw_stage == 4) { raw_stage = 0; raw_phase ^= 1; }
                            }
                            page_cons_stage += 1;
                            if (page_cons_stage == 6) { page_cons_stage = 0; page_cons_phase ^= 1; }
                            #pragma unroll 1
                            for (int _body_n_l = 0; _body_n_l < num_pairs_4 * 2 - 1; _body_n_l++) {
                                mbarrier_wait(page_offsets_full_addr + (page_cons_stage) * 8, page_cons_phase);
                                int page_smem_base_0 = page_cons_stage * 4;
                                #pragma unroll
                                for (int dim_half_1 = 0; dim_half_1 < 2; dim_half_1++) {
                                    mbarrier_wait(raw_kv_empty_addr + (raw_stage) * 8, raw_phase);
                                    mbarrier_arrive_expect_tx(raw_kv_full_addr + (raw_stage) * 8, BLOCK_N * HEAD_DIM_HALF);
                                    int raw_dst_1 = smem_kv_fp8_addr + (unsigned int)(raw_stage * 16384);
                                    #pragma unroll
                                    for (int page_in_block_1 = 0; page_in_block_1 < 2; page_in_block_1++) {
                                        int physical_page_1 = smem_page_offsets[page_smem_base_0 + page_in_block_1];
                                        int page_dst_1 = raw_dst_1 + page_in_block_1 * PAGE_SIZE * HEAD_DIM_HALF;
                                        {
                                            tma_5d_gmem2smem(page_dst_1, K, 0, 0, dim_half_1, kv_head_idx_2_1, physical_page_1, raw_kv_full_addr + (raw_stage) * 8);
                                        }
                                    }
                                    raw_stage += 1;
                                    if (raw_stage == 4) { raw_stage = 0; raw_phase ^= 1; }
                                }
                                page_cons_stage += 1;
                                if (page_cons_stage == 6) { page_cons_stage = 0; page_cons_phase ^= 1; }
                                int page_smem_base_1 = page_release_stage * 4;
                                #pragma unroll
                                for (int dim_half_2 = 0; dim_half_2 < 2; dim_half_2++) {
                                    mbarrier_wait(raw_kv_empty_addr + (raw_stage) * 8, raw_phase);
                                    mbarrier_arrive_expect_tx(raw_kv_full_addr + (raw_stage) * 8, BLOCK_N * HEAD_DIM_HALF);
                                    int raw_dst_2 = smem_kv_fp8_addr + (unsigned int)(raw_stage * 16384);
                                    #pragma unroll
                                    for (int page_in_block_2 = 0; page_in_block_2 < 2; page_in_block_2++) {
                                        int physical_page_2 = smem_page_offsets[page_smem_base_1 + 2 + page_in_block_2];
                                        int page_dst_2 = raw_dst_2 + page_in_block_2 * PAGE_SIZE * HEAD_DIM_HALF;
                                        {
                                            tma_5d_gmem2smem(page_dst_2, V, 0, 0, dim_half_2, kv_head_idx_2_1, physical_page_2, raw_kv_full_addr + (raw_stage) * 8);
                                        }
                                    }
                                    raw_stage += 1;
                                    if (raw_stage == 4) { raw_stage = 0; raw_phase ^= 1; }
                                }
                                mbarrier_arrive(page_offsets_empty_addr + (page_release_stage) * 8);
                                page_release_stage += 1;
                                if (page_release_stage == 6) { page_release_stage = 0; page_release_phase ^= 1; }
                            }
                            int page_smem_base_0_1 = page_release_stage * 4;
                            #pragma unroll
                            for (int dim_half_3 = 0; dim_half_3 < 2; dim_half_3++) {
                                mbarrier_wait(raw_kv_empty_addr + (raw_stage) * 8, raw_phase);
                                mbarrier_arrive_expect_tx(raw_kv_full_addr + (raw_stage) * 8, BLOCK_N * HEAD_DIM_HALF);
                                int raw_dst_3 = smem_kv_fp8_addr + (unsigned int)(raw_stage * 16384);
                                #pragma unroll
                                for (int page_in_block_3 = 0; page_in_block_3 < 2; page_in_block_3++) {
                                    int physical_page_3 = smem_page_offsets[page_smem_base_0_1 + 2 + page_in_block_3];
                                    int page_dst_3 = raw_dst_3 + page_in_block_3 * PAGE_SIZE * HEAD_DIM_HALF;
                                    {
                                        tma_5d_gmem2smem(page_dst_3, V, 0, 0, dim_half_3, kv_head_idx_2_1, physical_page_3, raw_kv_full_addr + (raw_stage) * 8);
                                    }
                                }
                                raw_stage += 1;
                                if (raw_stage == 4) { raw_stage = 0; raw_phase ^= 1; }
                            }
                            mbarrier_arrive(page_offsets_empty_addr + (page_release_stage) * 8);
                            page_release_stage += 1;
                            if (page_release_stage == 6) { page_release_stage = 0; page_release_phase ^= 1; }
                        }
                    }
                }
                {
                }
            }
        }
    }
    // ---- Role: transform ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // transform_main
            unsigned int work_stage_t = 0;
            int raw_stage_t = 0;
            int raw_phase_t = 0;
            int transformed_stage_t = 0;
            int transformed_phase_t = 1;
            int batch_idx_5 = 0;
            int q_row_idx_5 = 0;
            int kv_head_idx_5 = 0;
            int split_idx_5 = 0;
            int part_count_5 = NUM_SPLIT;
            int bundle_idx_5 = 0;
            int bundle_item_idx_5 = 0;
            {
                q_row_idx_5 = blockIdx.x;
                int head_split_idx_5 = blockIdx.y;
                kv_head_idx_5 = head_split_idx_5 / NUM_SPLIT;
                split_idx_5 = head_split_idx_5 % NUM_SPLIT;
                int schedule_batch_idx_5 = blockIdx.z;
                batch_idx_5 = schedule_batch_idx_5;
            }
            int batch_idx_t = batch_idx_5;
            int q_row_idx_t = q_row_idx_5;
            int kv_head_idx_t = kv_head_idx_5;
            int split_idx_t = split_idx_5;
            int part_count_t = part_count_5;
            int bundle_idx_t = bundle_idx_5;
            int bundle_item_idx_t = bundle_item_idx_5;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_t = 0; _tile_iter_t < total_tiles; _tile_iter_t++) {
                int visible_keys_5 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_t + 1;
                {
                    visible_keys_5 = seq_lens_kv[batch_idx_t] - Q_LEN + q_row_idx_t + 1;
                }
                if (visible_keys_5 < 0) {
                    visible_keys_5 = 0;
                }
                int seqlen_kv_t = visible_keys_5;
                int num_n_blocks_5 = (seqlen_kv_t + BLOCK_N - 1) / BLOCK_N;
                if (num_n_blocks_5 < 1) {
                    num_n_blocks_5 = 1;
                }
                int total_pairs_5 = (num_n_blocks_5 + 1) / 2;
                int base_pairs_5 = 0;
                int extra_pairs_5 = 0;
                {
                    base_pairs_5 = total_pairs_5 / NUM_SPLIT;
                    extra_pairs_5 = total_pairs_5 % NUM_SPLIT;
                }
                int num_pairs_5 = base_pairs_5;
                int split_start_pair_5 = extra_pairs_5 * (base_pairs_5 + 1) + (split_idx_t - extra_pairs_5) * base_pairs_5;
                if (split_idx_t < extra_pairs_5) {
                    num_pairs_5 = base_pairs_5 + 1;
                    split_start_pair_5 = split_idx_t * (base_pairs_5 + 1);
                }
                int total_half_items = num_pairs_5 * 2 * 4;
                #pragma unroll 1
                for (int _item = 0; _item < total_half_items; _item++) {
                    mbarrier_wait(raw_kv_full_addr + (raw_stage_t) * 8, raw_phase_t);
                    mbarrier_wait(kv_empty_addr + (transformed_stage_t) * 8, transformed_phase_t);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        const char* _src_ptr = smem_raw + (smem_kv_fp8_addr + (unsigned int)(raw_stage_t * 16384) - smem);
                        char* _dst_ptr = smem_raw + (smem_kv_addr + (unsigned int)(transformed_stage_t * 32768) - smem);
                        const int _tid = (int)threadIdx.x - (12) * 32;
                        uint64_t _src_buf[16];
                        #pragma unroll
                        for (int _outer = 0; _outer < 2; ++_outer) {
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                                int _off = _base + _tid;
                                _src_buf[_base >> 7] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                            }
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                                int _off = _base + _tid;
                                uint64_t _src64 = _src_buf[_base >> 7];
                                uint32_t _out_x16x2[4];
                                #pragma unroll
                                for (int _cv = 0; _cv < 4; ++_cv) {
                                    uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_out_x16x2[_cv]) : "h"(_e4m3x2));
                                    #else
                                    uint32_t _f16x2;
                                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                    float _f0;
                                    float _f1;
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                                    #endif
                                }
                                uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                                int _elt = _off * 8;
                                int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                                int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                                int _swz_off = _byte_off ^ ((_row % 8) * 16);
                                *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(raw_kv_empty_addr + (raw_stage_t) * 8);
                        mbarrier_arrive(kv_full_addr + (transformed_stage_t) * 8);
                    }
                    raw_stage_t += 1;
                    if (raw_stage_t == 4) { raw_stage_t = 0; raw_phase_t ^= 1; }
                    transformed_stage_t += 1;
                    if (transformed_stage_t == 2) { transformed_stage_t = 0; transformed_phase_t ^= 1; }
                }
                {
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
