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
#define Q_LEN 1
#define UNIFORM_KV_LEN 0
#define USE_REQUEST_ORDER 1
#define USE_SCALE_POINTERS 1
#define NUM_SPLIT 2
#define USE_SEGMENTED_CLC 1
#define USE_HIGH_BATCH_TWO_WAVE 1
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
kernel_cake_fmha_request_ordered_paged_decode_1c19b4be7cbb9ff51e92(CakeTensorMap const* Qt, CakeTensorMap const* K, CakeTensorMap const* V, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_LSE, unsigned int* __restrict__ split_completion, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, int* __restrict__ request_order, int max_pages_per_seq, int page_table_v_offset, float softmax_scale_log2, float output_scale, float* __restrict__ bmm1_scale_ptr, float* __restrict__ bmm2_scale_ptr, int bmm1_is_log2, int num_q_heads, int num_kv_heads, int batch_size, unsigned int total_tiles)
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
                int flat_tile_idx = blockIdx.x;
                int schedule_batch_idx = 0;
                {
                    q_row_idx = blockIdx.x;
                    bundle_idx = blockIdx.z;
                    int tile_rank = 0;
                    {
                        int starter_count = batch_size - 152;
                        int starter_rank = 74;
                        int tail_rank = 158;
                        if (batch_size == 160) {
                            starter_rank = 8;
                            tail_rank = 68;
                        }
                        if (batch_size == 192) {
                            starter_rank = 56;
                            tail_rank = 188;
                        }
                        tile_rank = starter_rank + bundle_idx;
                        if (bundle_idx >= starter_count) {
                            if (bundle_idx < 152) {
                                int local_bundle_count = 152 - starter_count;
                                int remaining_idx = bundle_idx - starter_count;
                                if (bundle_item_idx != 0) {
                                    remaining_idx = 2 * local_bundle_count - 1 - remaining_idx;
                                }
                                tile_rank = remaining_idx;
                                if (remaining_idx >= starter_rank) {
                                    tile_rank = remaining_idx + starter_count;
                                }
                                if (remaining_idx >= tail_rank - starter_count) {
                                    tile_rank = remaining_idx + 2 * starter_count;
                                }
                            } else {
                                tile_rank = tail_rank + (starter_count - 1 - (bundle_idx - 152));
                            }
                        }
                    }
                    int tile_rank_0 = tile_rank;
                    int schedule_batch_idx_1 = 0;
                    int split_idx_2 = 0;
                    int split_requests = 304 - batch_size;
                    {
                        if (batch_size == 160) {
                            schedule_batch_idx_1 = tile_rank_0 - 294;
                            split_idx_2 = 0;
                            schedule_batch_idx_1 = ((tile_rank_0 < 294) ? tile_rank_0 - 284 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 294) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 284) ? 10 + tile_rank_0 - 274 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 284) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 274) ? 10 + tile_rank_0 - 264 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 274) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 264) ? 20 + tile_rank_0 - 254 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 264) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 254) ? 20 + tile_rank_0 - 244 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 254) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 244) ? 30 + tile_rank_0 - 234 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 244) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 234) ? 30 + tile_rank_0 - 224 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 234) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 224) ? 40 + tile_rank_0 - 214 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 224) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 214) ? 40 + tile_rank_0 - 204 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 214) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 204) ? 50 + tile_rank_0 - 194 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 204) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 194) ? 50 + tile_rank_0 - 184 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 194) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 184) ? 60 + tile_rank_0 - 174 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 184) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 174) ? 60 + tile_rank_0 - 164 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 174) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 164) ? 70 + tile_rank_0 - 154 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 164) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 154) ? 70 + tile_rank_0 - 144 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 154) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 144) ? 80 + tile_rank_0 - 134 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 144) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 134) ? 80 + tile_rank_0 - 124 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 134) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 124) ? 90 + tile_rank_0 - 114 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 124) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 114) ? 144 + tile_rank_0 - 108 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 114) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 108) ? 90 + tile_rank_0 - 98 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 108) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 98) ? 100 + tile_rank_0 - 88 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 98) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 88) ? 100 + tile_rank_0 - 78 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 88) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 78) ? 110 + tile_rank_0 - 68 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 78) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 68) ? 110 + tile_rank_0 - 58 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 68) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 58) ? 120 + tile_rank_0 - 48 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 58) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 48) ? 120 + tile_rank_0 - 38 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 48) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 38) ? 130 + tile_rank_0 - 28 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 38) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 28) ? 130 + tile_rank_0 - 18 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 28) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 18) ? 140 + tile_rank_0 - 14 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 18) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 14) ? 140 + tile_rank_0 - 10 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 14) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 10) ? 150 + tile_rank_0 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 10) ? 0 : split_idx_2);
                        } else if (batch_size == 192) {
                            schedule_batch_idx_1 = tile_rank_0 - 292;
                            split_idx_2 = 0;
                            schedule_batch_idx_1 = ((tile_rank_0 < 292) ? tile_rank_0 - 280 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 292) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 280) ? 12 + tile_rank_0 - 268 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 280) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 268) ? 112 + tile_rank_0 - 260 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 268) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 260) ? 12 + tile_rank_0 - 248 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 260) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 248) ? 24 + tile_rank_0 - 236 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 248) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 236) ? 120 + tile_rank_0 - 224 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 236) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 224) ? 24 + tile_rank_0 - 212 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 224) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 212) ? 36 + tile_rank_0 - 200 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 212) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 200) ? 132 + tile_rank_0 - 188 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 200) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 188) ? 36 + tile_rank_0 - 176 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 188) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 176) ? 48 + tile_rank_0 - 164 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 176) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 164) ? 48 + tile_rank_0 - 152 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 164) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 152) ? 60 + tile_rank_0 - 140 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 152) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 140) ? 144 + tile_rank_0 - 128 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 140) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 128) ? 60 + tile_rank_0 - 116 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 128) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 116) ? 72 + tile_rank_0 - 104 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 116) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 104) ? 156 + tile_rank_0 - 92 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 104) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 92) ? 72 + tile_rank_0 - 80 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 92) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 80) ? 84 + tile_rank_0 - 68 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 80) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 68) ? 84 + tile_rank_0 - 56 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 68) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 56) ? 96 + tile_rank_0 - 44 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 56) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 44) ? 96 + tile_rank_0 - 32 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 44) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 32) ? 108 + tile_rank_0 - 28 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 32) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 28) ? 168 + tile_rank_0 - 16 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 28) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 16) ? 108 + tile_rank_0 - 12 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 16) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 12) ? 180 + tile_rank_0 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 12) ? 0 : split_idx_2);
                        } else {
                            schedule_batch_idx_1 = 80 + tile_rank_0 - 300;
                            split_idx_2 = 0;
                            schedule_batch_idx_1 = ((tile_rank_0 < 300) ? 84 + tile_rank_0 - 286 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 300) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 286) ? 98 + tile_rank_0 - 272 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 286) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 272) ? 112 + tile_rank_0 - 258 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 272) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 258) ? tile_rank_0 - 244 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 258) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 244) ? tile_rank_0 - 230 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 244) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 230) ? 14 + tile_rank_0 - 216 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 230) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 216) ? 126 + tile_rank_0 - 202 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 216) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 202) ? 14 + tile_rank_0 - 188 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 202) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 188) ? 28 + tile_rank_0 - 174 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 188) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 174) ? 140 + tile_rank_0 - 160 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 174) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 160) ? 28 + tile_rank_0 - 146 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 160) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 146) ? 42 + tile_rank_0 - 132 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 146) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 132) ? 154 + tile_rank_0 - 118 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 132) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 118) ? 42 + tile_rank_0 - 104 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 118) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 104) ? 56 + tile_rank_0 - 90 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 104) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 90) ? 56 + tile_rank_0 - 76 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 90) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 76) ? 70 + tile_rank_0 - 66 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 76) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 66) ? 168 + tile_rank_0 - 52 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 66) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 52) ? 70 + tile_rank_0 - 42 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 52) ? 1 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 42) ? 182 + tile_rank_0 - 28 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 42) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 28) ? 196 + tile_rank_0 - 14 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 28) ? 0 : split_idx_2);
                            schedule_batch_idx_1 = ((tile_rank_0 < 14) ? 210 + tile_rank_0 : schedule_batch_idx_1);
                            split_idx_2 = ((tile_rank_0 < 14) ? 0 : split_idx_2);
                        }
                    }
                    int part_count_3 = 1;
                    if (schedule_batch_idx_1 < split_requests) {
                        part_count_3 = 2;
                    }
                    schedule_batch_idx = schedule_batch_idx_1;
                    split_idx = split_idx_2;
                    part_count = part_count_3;
                }
                batch_idx = schedule_batch_idx;
                {
                    {
                        batch_idx = request_order[schedule_batch_idx];
                    }
                }
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
                    base_pairs = total_pairs / part_count_s;
                    extra_pairs = total_pairs % part_count_s;
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
                    int has_local = 0;
                    {
                        int item_count = 1;
                        {
                            int starter_count_1 = batch_size - 152;
                            if (bundle_idx_s >= starter_count_1) {
                                if (bundle_idx_s < 152) {
                                    item_count = 2;
                                }
                            }
                        }
                        if (item_count > bundle_item_idx_s + 1) {
                            has_local = 1;
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
                    unsigned int valid = 1;
                    unsigned int flat_or_q = 0;
                    unsigned int default_part = 0;
                    unsigned int default_batch = 0;
                    if (has_local == 0) {
                        uint32_t _clc_valid_5 = 0;
                        uint32_t _clc_ctaid_x_5;
                        uint32_t _clc_ctaid_y_5;
                        uint32_t _clc_ctaid_z_5;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%4];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %3, 1, 0, p1;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_x_5), "=r"(_clc_ctaid_y_5), "=r"(_clc_ctaid_z_5), "=r"(_clc_valid_5)
                            : "r"(work_response_view_addr + work_stage_s * 16 + 0 * 16)
                            : "memory");
                        valid = _clc_valid_5;
                        flat_or_q = _clc_ctaid_x_5;
                        default_part = _clc_ctaid_y_5;
                        default_batch = _clc_ctaid_z_5;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
                    work_stage_s += 1;
                    if (work_stage_s == 2) { work_stage_s = 0; _phase_work_full ^= 1; }
                    unsigned int next_q_row = 0;
                    unsigned int next_part = 0;
                    unsigned int schedule_batch = 0;
                    int kv_head_idx_0 = 0;
                    int split_idx_1 = 0;
                    int part_count_2 = NUM_SPLIT;
                    int next_bundle_idx = 0;
                    int next_bundle_item_idx = 0;
                    {
                        {
                            if (has_local != 0) {
                                next_q_row = (unsigned int)q_row_idx_s;
                                next_bundle_idx = bundle_idx_s;
                                next_bundle_item_idx = bundle_item_idx_s + 1;
                            } else {
                                next_q_row = flat_or_q;
                                next_bundle_idx = (int)default_batch;
                                next_bundle_item_idx = 0;
                            }
                            int tile_rank_1 = 0;
                            {
                                int starter_count_2 = batch_size - 152;
                                int starter_rank_1 = 74;
                                int tail_rank_1 = 158;
                                if (batch_size == 160) {
                                    starter_rank_1 = 8;
                                    tail_rank_1 = 68;
                                }
                                if (batch_size == 192) {
                                    starter_rank_1 = 56;
                                    tail_rank_1 = 188;
                                }
                                tile_rank_1 = starter_rank_1 + next_bundle_idx;
                                if (next_bundle_idx >= starter_count_2) {
                                    if (next_bundle_idx < 152) {
                                        int local_bundle_count_1 = 152 - starter_count_2;
                                        int remaining_idx_1 = next_bundle_idx - starter_count_2;
                                        if (next_bundle_item_idx != 0) {
                                            remaining_idx_1 = 2 * local_bundle_count_1 - 1 - remaining_idx_1;
                                        }
                                        tile_rank_1 = remaining_idx_1;
                                        if (remaining_idx_1 >= starter_rank_1) {
                                            tile_rank_1 = remaining_idx_1 + starter_count_2;
                                        }
                                        if (remaining_idx_1 >= tail_rank_1 - starter_count_2) {
                                            tile_rank_1 = remaining_idx_1 + 2 * starter_count_2;
                                        }
                                    } else {
                                        tile_rank_1 = tail_rank_1 + (starter_count_2 - 1 - (next_bundle_idx - 152));
                                    }
                                }
                            }
                            int tile_rank_0_1 = tile_rank_1;
                            int schedule_batch_idx_2 = 0;
                            int split_idx_2_1 = 0;
                            int split_requests_1 = 304 - batch_size;
                            {
                                if (batch_size == 160) {
                                    schedule_batch_idx_2 = tile_rank_0_1 - 294;
                                    split_idx_2_1 = 0;
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 294) ? tile_rank_0_1 - 284 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 294) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 284) ? 10 + tile_rank_0_1 - 274 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 284) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 274) ? 10 + tile_rank_0_1 - 264 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 274) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 264) ? 20 + tile_rank_0_1 - 254 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 264) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 254) ? 20 + tile_rank_0_1 - 244 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 254) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 244) ? 30 + tile_rank_0_1 - 234 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 244) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 234) ? 30 + tile_rank_0_1 - 224 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 234) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 224) ? 40 + tile_rank_0_1 - 214 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 224) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 214) ? 40 + tile_rank_0_1 - 204 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 214) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 204) ? 50 + tile_rank_0_1 - 194 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 204) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 194) ? 50 + tile_rank_0_1 - 184 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 194) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 184) ? 60 + tile_rank_0_1 - 174 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 184) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 174) ? 60 + tile_rank_0_1 - 164 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 174) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 164) ? 70 + tile_rank_0_1 - 154 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 164) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 154) ? 70 + tile_rank_0_1 - 144 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 154) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 144) ? 80 + tile_rank_0_1 - 134 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 144) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 134) ? 80 + tile_rank_0_1 - 124 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 134) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 124) ? 90 + tile_rank_0_1 - 114 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 124) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 114) ? 144 + tile_rank_0_1 - 108 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 114) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 108) ? 90 + tile_rank_0_1 - 98 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 108) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 98) ? 100 + tile_rank_0_1 - 88 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 98) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 88) ? 100 + tile_rank_0_1 - 78 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 88) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 78) ? 110 + tile_rank_0_1 - 68 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 78) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 68) ? 110 + tile_rank_0_1 - 58 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 68) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 58) ? 120 + tile_rank_0_1 - 48 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 58) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 48) ? 120 + tile_rank_0_1 - 38 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 48) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 38) ? 130 + tile_rank_0_1 - 28 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 38) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 28) ? 130 + tile_rank_0_1 - 18 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 28) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 18) ? 140 + tile_rank_0_1 - 14 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 18) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 14) ? 140 + tile_rank_0_1 - 10 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 14) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 10) ? 150 + tile_rank_0_1 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 10) ? 0 : split_idx_2_1);
                                } else if (batch_size == 192) {
                                    schedule_batch_idx_2 = tile_rank_0_1 - 292;
                                    split_idx_2_1 = 0;
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 292) ? tile_rank_0_1 - 280 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 292) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 280) ? 12 + tile_rank_0_1 - 268 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 280) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 268) ? 112 + tile_rank_0_1 - 260 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 268) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 260) ? 12 + tile_rank_0_1 - 248 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 260) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 248) ? 24 + tile_rank_0_1 - 236 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 248) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 236) ? 120 + tile_rank_0_1 - 224 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 236) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 224) ? 24 + tile_rank_0_1 - 212 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 224) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 212) ? 36 + tile_rank_0_1 - 200 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 212) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 200) ? 132 + tile_rank_0_1 - 188 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 200) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 188) ? 36 + tile_rank_0_1 - 176 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 188) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 176) ? 48 + tile_rank_0_1 - 164 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 176) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 164) ? 48 + tile_rank_0_1 - 152 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 164) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 152) ? 60 + tile_rank_0_1 - 140 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 152) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 140) ? 144 + tile_rank_0_1 - 128 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 140) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 128) ? 60 + tile_rank_0_1 - 116 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 128) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 116) ? 72 + tile_rank_0_1 - 104 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 116) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 104) ? 156 + tile_rank_0_1 - 92 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 104) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 92) ? 72 + tile_rank_0_1 - 80 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 92) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 80) ? 84 + tile_rank_0_1 - 68 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 80) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 68) ? 84 + tile_rank_0_1 - 56 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 68) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 56) ? 96 + tile_rank_0_1 - 44 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 56) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 44) ? 96 + tile_rank_0_1 - 32 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 44) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 32) ? 108 + tile_rank_0_1 - 28 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 32) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 28) ? 168 + tile_rank_0_1 - 16 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 28) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 16) ? 108 + tile_rank_0_1 - 12 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 16) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 12) ? 180 + tile_rank_0_1 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 12) ? 0 : split_idx_2_1);
                                } else {
                                    schedule_batch_idx_2 = 80 + tile_rank_0_1 - 300;
                                    split_idx_2_1 = 0;
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 300) ? 84 + tile_rank_0_1 - 286 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 300) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 286) ? 98 + tile_rank_0_1 - 272 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 286) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 272) ? 112 + tile_rank_0_1 - 258 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 272) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 258) ? tile_rank_0_1 - 244 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 258) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 244) ? tile_rank_0_1 - 230 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 244) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 230) ? 14 + tile_rank_0_1 - 216 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 230) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 216) ? 126 + tile_rank_0_1 - 202 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 216) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 202) ? 14 + tile_rank_0_1 - 188 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 202) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 188) ? 28 + tile_rank_0_1 - 174 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 188) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 174) ? 140 + tile_rank_0_1 - 160 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 174) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 160) ? 28 + tile_rank_0_1 - 146 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 160) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 146) ? 42 + tile_rank_0_1 - 132 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 146) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 132) ? 154 + tile_rank_0_1 - 118 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 132) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 118) ? 42 + tile_rank_0_1 - 104 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 118) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 104) ? 56 + tile_rank_0_1 - 90 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 104) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 90) ? 56 + tile_rank_0_1 - 76 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 90) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 76) ? 70 + tile_rank_0_1 - 66 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 76) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 66) ? 168 + tile_rank_0_1 - 52 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 66) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 52) ? 70 + tile_rank_0_1 - 42 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 52) ? 1 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 42) ? 182 + tile_rank_0_1 - 28 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 42) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 28) ? 196 + tile_rank_0_1 - 14 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 28) ? 0 : split_idx_2_1);
                                    schedule_batch_idx_2 = ((tile_rank_0_1 < 14) ? 210 + tile_rank_0_1 : schedule_batch_idx_2);
                                    split_idx_2_1 = ((tile_rank_0_1 < 14) ? 0 : split_idx_2_1);
                                }
                            }
                            int part_count_3_1 = 1;
                            if (schedule_batch_idx_2 < split_requests_1) {
                                part_count_3_1 = 2;
                            }
                            schedule_batch = (unsigned int)schedule_batch_idx_2;
                            next_part = (unsigned int)(part_count_3_1 << 16 | split_idx_2_1);
                        }
                    }
                    unsigned int next_batch = schedule_batch;
                    {
                        if (valid != 0) {
                            {
                                {
                                    next_batch = request_order[schedule_batch];
                                }
                            }
                        }
                    }
                    {
                        int packed_part = (int)next_part;
                        split_idx_1 = packed_part % 65536;
                        part_count_2 = packed_part / 65536;
                    }
                    unsigned int valid_s = valid;
                    batch_idx_s = (int)next_batch;
                    q_row_idx_s = (int)next_q_row;
                    kv_head_idx_s = kv_head_idx_0;
                    split_idx_s = split_idx_1;
                    part_count_s = part_count_2;
                    bundle_idx_s = next_bundle_idx;
                    bundle_item_idx_s = next_bundle_item_idx;
                    if (valid_s == 0) {
                        break;
                    }
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
            int split_idx_3 = 0;
            int part_count_1 = NUM_SPLIT;
            int bundle_idx_1 = 0;
            int bundle_item_idx_1 = 0;
            {
                int flat_tile_idx_1 = blockIdx.x;
                int schedule_batch_idx_3 = 0;
                {
                    q_row_idx_1 = blockIdx.x;
                    bundle_idx_1 = blockIdx.z;
                    int tile_rank_2 = 0;
                    {
                        int starter_count_3 = batch_size - 152;
                        int starter_rank_2 = 74;
                        int tail_rank_2 = 158;
                        if (batch_size == 160) {
                            starter_rank_2 = 8;
                            tail_rank_2 = 68;
                        }
                        if (batch_size == 192) {
                            starter_rank_2 = 56;
                            tail_rank_2 = 188;
                        }
                        tile_rank_2 = starter_rank_2 + bundle_idx_1;
                        if (bundle_idx_1 >= starter_count_3) {
                            if (bundle_idx_1 < 152) {
                                int local_bundle_count_2 = 152 - starter_count_3;
                                int remaining_idx_2 = bundle_idx_1 - starter_count_3;
                                if (bundle_item_idx_1 != 0) {
                                    remaining_idx_2 = 2 * local_bundle_count_2 - 1 - remaining_idx_2;
                                }
                                tile_rank_2 = remaining_idx_2;
                                if (remaining_idx_2 >= starter_rank_2) {
                                    tile_rank_2 = remaining_idx_2 + starter_count_3;
                                }
                                if (remaining_idx_2 >= tail_rank_2 - starter_count_3) {
                                    tile_rank_2 = remaining_idx_2 + 2 * starter_count_3;
                                }
                            } else {
                                tile_rank_2 = tail_rank_2 + (starter_count_3 - 1 - (bundle_idx_1 - 152));
                            }
                        }
                    }
                    int tile_rank_0_2 = tile_rank_2;
                    int schedule_batch_idx_1_1 = 0;
                    int split_idx_2_2 = 0;
                    int split_requests_2 = 304 - batch_size;
                    {
                        if (batch_size == 160) {
                            schedule_batch_idx_1_1 = tile_rank_0_2 - 294;
                            split_idx_2_2 = 0;
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 294) ? tile_rank_0_2 - 284 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 294) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 284) ? 10 + tile_rank_0_2 - 274 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 284) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 274) ? 10 + tile_rank_0_2 - 264 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 274) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 264) ? 20 + tile_rank_0_2 - 254 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 264) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 254) ? 20 + tile_rank_0_2 - 244 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 254) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 244) ? 30 + tile_rank_0_2 - 234 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 244) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 234) ? 30 + tile_rank_0_2 - 224 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 234) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 224) ? 40 + tile_rank_0_2 - 214 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 224) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 214) ? 40 + tile_rank_0_2 - 204 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 214) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 204) ? 50 + tile_rank_0_2 - 194 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 204) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 194) ? 50 + tile_rank_0_2 - 184 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 194) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 184) ? 60 + tile_rank_0_2 - 174 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 184) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 174) ? 60 + tile_rank_0_2 - 164 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 174) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 164) ? 70 + tile_rank_0_2 - 154 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 164) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 154) ? 70 + tile_rank_0_2 - 144 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 154) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 144) ? 80 + tile_rank_0_2 - 134 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 144) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 134) ? 80 + tile_rank_0_2 - 124 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 134) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 124) ? 90 + tile_rank_0_2 - 114 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 124) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 114) ? 144 + tile_rank_0_2 - 108 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 114) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 108) ? 90 + tile_rank_0_2 - 98 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 108) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 98) ? 100 + tile_rank_0_2 - 88 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 98) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 88) ? 100 + tile_rank_0_2 - 78 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 88) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 78) ? 110 + tile_rank_0_2 - 68 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 78) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 68) ? 110 + tile_rank_0_2 - 58 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 68) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 58) ? 120 + tile_rank_0_2 - 48 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 58) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 48) ? 120 + tile_rank_0_2 - 38 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 48) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 38) ? 130 + tile_rank_0_2 - 28 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 38) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 28) ? 130 + tile_rank_0_2 - 18 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 28) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 18) ? 140 + tile_rank_0_2 - 14 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 18) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 14) ? 140 + tile_rank_0_2 - 10 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 14) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 10) ? 150 + tile_rank_0_2 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 10) ? 0 : split_idx_2_2);
                        } else if (batch_size == 192) {
                            schedule_batch_idx_1_1 = tile_rank_0_2 - 292;
                            split_idx_2_2 = 0;
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 292) ? tile_rank_0_2 - 280 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 292) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 280) ? 12 + tile_rank_0_2 - 268 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 280) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 268) ? 112 + tile_rank_0_2 - 260 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 268) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 260) ? 12 + tile_rank_0_2 - 248 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 260) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 248) ? 24 + tile_rank_0_2 - 236 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 248) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 236) ? 120 + tile_rank_0_2 - 224 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 236) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 224) ? 24 + tile_rank_0_2 - 212 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 224) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 212) ? 36 + tile_rank_0_2 - 200 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 212) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 200) ? 132 + tile_rank_0_2 - 188 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 200) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 188) ? 36 + tile_rank_0_2 - 176 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 188) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 176) ? 48 + tile_rank_0_2 - 164 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 176) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 164) ? 48 + tile_rank_0_2 - 152 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 164) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 152) ? 60 + tile_rank_0_2 - 140 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 152) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 140) ? 144 + tile_rank_0_2 - 128 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 140) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 128) ? 60 + tile_rank_0_2 - 116 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 128) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 116) ? 72 + tile_rank_0_2 - 104 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 116) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 104) ? 156 + tile_rank_0_2 - 92 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 104) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 92) ? 72 + tile_rank_0_2 - 80 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 92) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 80) ? 84 + tile_rank_0_2 - 68 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 80) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 68) ? 84 + tile_rank_0_2 - 56 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 68) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 56) ? 96 + tile_rank_0_2 - 44 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 56) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 44) ? 96 + tile_rank_0_2 - 32 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 44) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 32) ? 108 + tile_rank_0_2 - 28 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 32) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 28) ? 168 + tile_rank_0_2 - 16 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 28) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 16) ? 108 + tile_rank_0_2 - 12 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 16) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 12) ? 180 + tile_rank_0_2 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 12) ? 0 : split_idx_2_2);
                        } else {
                            schedule_batch_idx_1_1 = 80 + tile_rank_0_2 - 300;
                            split_idx_2_2 = 0;
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 300) ? 84 + tile_rank_0_2 - 286 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 300) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 286) ? 98 + tile_rank_0_2 - 272 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 286) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 272) ? 112 + tile_rank_0_2 - 258 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 272) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 258) ? tile_rank_0_2 - 244 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 258) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 244) ? tile_rank_0_2 - 230 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 244) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 230) ? 14 + tile_rank_0_2 - 216 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 230) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 216) ? 126 + tile_rank_0_2 - 202 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 216) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 202) ? 14 + tile_rank_0_2 - 188 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 202) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 188) ? 28 + tile_rank_0_2 - 174 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 188) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 174) ? 140 + tile_rank_0_2 - 160 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 174) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 160) ? 28 + tile_rank_0_2 - 146 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 160) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 146) ? 42 + tile_rank_0_2 - 132 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 146) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 132) ? 154 + tile_rank_0_2 - 118 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 132) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 118) ? 42 + tile_rank_0_2 - 104 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 118) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 104) ? 56 + tile_rank_0_2 - 90 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 104) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 90) ? 56 + tile_rank_0_2 - 76 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 90) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 76) ? 70 + tile_rank_0_2 - 66 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 76) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 66) ? 168 + tile_rank_0_2 - 52 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 66) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 52) ? 70 + tile_rank_0_2 - 42 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 52) ? 1 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 42) ? 182 + tile_rank_0_2 - 28 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 42) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 28) ? 196 + tile_rank_0_2 - 14 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 28) ? 0 : split_idx_2_2);
                            schedule_batch_idx_1_1 = ((tile_rank_0_2 < 14) ? 210 + tile_rank_0_2 : schedule_batch_idx_1_1);
                            split_idx_2_2 = ((tile_rank_0_2 < 14) ? 0 : split_idx_2_2);
                        }
                    }
                    int part_count_3_2 = 1;
                    if (schedule_batch_idx_1_1 < split_requests_2) {
                        part_count_3_2 = 2;
                    }
                    schedule_batch_idx_3 = schedule_batch_idx_1_1;
                    split_idx_3 = split_idx_2_2;
                    part_count_1 = part_count_3_2;
                }
                batch_idx_1 = schedule_batch_idx_3;
                {
                    {
                        batch_idx_1 = request_order[schedule_batch_idx_3];
                    }
                }
            }
            int batch_idx_c = batch_idx_1;
            int q_row_idx_c = q_row_idx_1;
            int kv_head_idx_c = kv_head_idx_1;
            int split_idx_c = split_idx_3;
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
                    base_pairs_1 = total_pairs_1 / part_count_c;
                    extra_pairs_1 = total_pairs_1 % part_count_c;
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
                            if (part_count_c == 1) {
                            } else {
                                float _log2_1;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(stats_sum_c));
                                float lse_value_partial_seg = _log2_1 + stats_max_c * bmm1_scale_log2_c;
                                int partial_lse_idx = stats_output_row_c * 16 + split_idx_c;
                                *(reinterpret_cast<float*>(partial_LSE + partial_lse_idx) + (0)) = lse_value_partial_seg;
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
                            if (part_count_c == 1) {
                                int direct_out_idx_hi_seg = output_row * HEAD_DIM + d_idx;
                                int direct_out_idx_lo_seg = direct_out_idx_hi_seg + HEAD_DIM_HALF;
                                *(reinterpret_cast<__nv_bfloat16*>(O + direct_out_idx_hi_seg) + (0)) = __float2bfloat16_rn(final_o_hi);
                                *(reinterpret_cast<__nv_bfloat16*>(O + direct_out_idx_lo_seg) + (0)) = __float2bfloat16_rn(final_o_lo);
                            } else {
                                int partial_row = output_row * 16 + split_idx_c;
                                int partial_idx_hi = partial_row * HEAD_DIM + d_idx;
                                int partial_idx_lo = partial_idx_hi + HEAD_DIM_HALF;
                                *(reinterpret_cast<__nv_bfloat16*>(partial_O + partial_idx_hi) + (0)) = __float2bfloat16_rn(final_o_hi);
                                *(reinterpret_cast<__nv_bfloat16*>(partial_O + partial_idx_lo) + (0)) = __float2bfloat16_rn(final_o_lo);
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
                    int has_local_1 = 0;
                    {
                        int item_count_1 = 1;
                        {
                            int starter_count_4 = batch_size - 152;
                            if (bundle_idx_c >= starter_count_4) {
                                if (bundle_idx_c < 152) {
                                    item_count_1 = 2;
                                }
                            }
                        }
                        if (item_count_1 > bundle_item_idx_c + 1) {
                            has_local_1 = 1;
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
                    unsigned int valid_1 = 1;
                    unsigned int flat_or_q_1 = 0;
                    unsigned int default_part_1 = 0;
                    unsigned int default_batch_1 = 0;
                    if (has_local_1 == 0) {
                        uint32_t _clc_valid_6 = 0;
                        uint32_t _clc_ctaid_x_6;
                        uint32_t _clc_ctaid_y_6;
                        uint32_t _clc_ctaid_z_6;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%4];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %3, 1, 0, p1;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_x_6), "=r"(_clc_ctaid_y_6), "=r"(_clc_ctaid_z_6), "=r"(_clc_valid_6)
                            : "r"(work_response_view_addr + work_stage_c * 16 + 0 * 16)
                            : "memory");
                        valid_1 = _clc_valid_6;
                        flat_or_q_1 = _clc_ctaid_x_6;
                        default_part_1 = _clc_ctaid_y_6;
                        default_batch_1 = _clc_ctaid_z_6;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                    work_stage_c += 1;
                    if (work_stage_c == 2) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
                    unsigned int next_q_row_1 = 0;
                    unsigned int next_part_1 = 0;
                    unsigned int schedule_batch_1 = 0;
                    int kv_head_idx_0_1 = 0;
                    int split_idx_1_1 = 0;
                    int part_count_2_1 = NUM_SPLIT;
                    int next_bundle_idx_1 = 0;
                    int next_bundle_item_idx_1 = 0;
                    {
                        {
                            if (has_local_1 != 0) {
                                next_q_row_1 = (unsigned int)q_row_idx_c;
                                next_bundle_idx_1 = bundle_idx_c;
                                next_bundle_item_idx_1 = bundle_item_idx_c + 1;
                            } else {
                                next_q_row_1 = flat_or_q_1;
                                next_bundle_idx_1 = (int)default_batch_1;
                                next_bundle_item_idx_1 = 0;
                            }
                            int tile_rank_3 = 0;
                            {
                                int starter_count_5 = batch_size - 152;
                                int starter_rank_3 = 74;
                                int tail_rank_3 = 158;
                                if (batch_size == 160) {
                                    starter_rank_3 = 8;
                                    tail_rank_3 = 68;
                                }
                                if (batch_size == 192) {
                                    starter_rank_3 = 56;
                                    tail_rank_3 = 188;
                                }
                                tile_rank_3 = starter_rank_3 + next_bundle_idx_1;
                                if (next_bundle_idx_1 >= starter_count_5) {
                                    if (next_bundle_idx_1 < 152) {
                                        int local_bundle_count_3 = 152 - starter_count_5;
                                        int remaining_idx_3 = next_bundle_idx_1 - starter_count_5;
                                        if (next_bundle_item_idx_1 != 0) {
                                            remaining_idx_3 = 2 * local_bundle_count_3 - 1 - remaining_idx_3;
                                        }
                                        tile_rank_3 = remaining_idx_3;
                                        if (remaining_idx_3 >= starter_rank_3) {
                                            tile_rank_3 = remaining_idx_3 + starter_count_5;
                                        }
                                        if (remaining_idx_3 >= tail_rank_3 - starter_count_5) {
                                            tile_rank_3 = remaining_idx_3 + 2 * starter_count_5;
                                        }
                                    } else {
                                        tile_rank_3 = tail_rank_3 + (starter_count_5 - 1 - (next_bundle_idx_1 - 152));
                                    }
                                }
                            }
                            int tile_rank_0_3 = tile_rank_3;
                            int schedule_batch_idx_4 = 0;
                            int split_idx_2_3 = 0;
                            int split_requests_3 = 304 - batch_size;
                            {
                                if (batch_size == 160) {
                                    schedule_batch_idx_4 = tile_rank_0_3 - 294;
                                    split_idx_2_3 = 0;
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 294) ? tile_rank_0_3 - 284 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 294) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 284) ? 10 + tile_rank_0_3 - 274 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 284) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 274) ? 10 + tile_rank_0_3 - 264 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 274) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 264) ? 20 + tile_rank_0_3 - 254 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 264) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 254) ? 20 + tile_rank_0_3 - 244 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 254) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 244) ? 30 + tile_rank_0_3 - 234 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 244) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 234) ? 30 + tile_rank_0_3 - 224 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 234) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 224) ? 40 + tile_rank_0_3 - 214 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 224) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 214) ? 40 + tile_rank_0_3 - 204 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 214) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 204) ? 50 + tile_rank_0_3 - 194 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 204) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 194) ? 50 + tile_rank_0_3 - 184 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 194) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 184) ? 60 + tile_rank_0_3 - 174 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 184) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 174) ? 60 + tile_rank_0_3 - 164 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 174) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 164) ? 70 + tile_rank_0_3 - 154 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 164) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 154) ? 70 + tile_rank_0_3 - 144 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 154) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 144) ? 80 + tile_rank_0_3 - 134 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 144) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 134) ? 80 + tile_rank_0_3 - 124 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 134) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 124) ? 90 + tile_rank_0_3 - 114 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 124) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 114) ? 144 + tile_rank_0_3 - 108 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 114) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 108) ? 90 + tile_rank_0_3 - 98 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 108) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 98) ? 100 + tile_rank_0_3 - 88 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 98) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 88) ? 100 + tile_rank_0_3 - 78 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 88) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 78) ? 110 + tile_rank_0_3 - 68 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 78) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 68) ? 110 + tile_rank_0_3 - 58 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 68) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 58) ? 120 + tile_rank_0_3 - 48 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 58) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 48) ? 120 + tile_rank_0_3 - 38 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 48) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 38) ? 130 + tile_rank_0_3 - 28 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 38) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 28) ? 130 + tile_rank_0_3 - 18 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 28) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 18) ? 140 + tile_rank_0_3 - 14 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 18) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 14) ? 140 + tile_rank_0_3 - 10 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 14) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 10) ? 150 + tile_rank_0_3 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 10) ? 0 : split_idx_2_3);
                                } else if (batch_size == 192) {
                                    schedule_batch_idx_4 = tile_rank_0_3 - 292;
                                    split_idx_2_3 = 0;
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 292) ? tile_rank_0_3 - 280 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 292) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 280) ? 12 + tile_rank_0_3 - 268 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 280) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 268) ? 112 + tile_rank_0_3 - 260 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 268) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 260) ? 12 + tile_rank_0_3 - 248 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 260) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 248) ? 24 + tile_rank_0_3 - 236 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 248) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 236) ? 120 + tile_rank_0_3 - 224 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 236) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 224) ? 24 + tile_rank_0_3 - 212 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 224) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 212) ? 36 + tile_rank_0_3 - 200 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 212) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 200) ? 132 + tile_rank_0_3 - 188 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 200) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 188) ? 36 + tile_rank_0_3 - 176 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 188) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 176) ? 48 + tile_rank_0_3 - 164 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 176) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 164) ? 48 + tile_rank_0_3 - 152 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 164) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 152) ? 60 + tile_rank_0_3 - 140 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 152) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 140) ? 144 + tile_rank_0_3 - 128 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 140) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 128) ? 60 + tile_rank_0_3 - 116 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 128) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 116) ? 72 + tile_rank_0_3 - 104 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 116) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 104) ? 156 + tile_rank_0_3 - 92 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 104) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 92) ? 72 + tile_rank_0_3 - 80 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 92) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 80) ? 84 + tile_rank_0_3 - 68 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 80) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 68) ? 84 + tile_rank_0_3 - 56 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 68) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 56) ? 96 + tile_rank_0_3 - 44 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 56) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 44) ? 96 + tile_rank_0_3 - 32 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 44) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 32) ? 108 + tile_rank_0_3 - 28 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 32) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 28) ? 168 + tile_rank_0_3 - 16 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 28) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 16) ? 108 + tile_rank_0_3 - 12 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 16) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 12) ? 180 + tile_rank_0_3 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 12) ? 0 : split_idx_2_3);
                                } else {
                                    schedule_batch_idx_4 = 80 + tile_rank_0_3 - 300;
                                    split_idx_2_3 = 0;
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 300) ? 84 + tile_rank_0_3 - 286 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 300) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 286) ? 98 + tile_rank_0_3 - 272 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 286) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 272) ? 112 + tile_rank_0_3 - 258 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 272) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 258) ? tile_rank_0_3 - 244 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 258) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 244) ? tile_rank_0_3 - 230 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 244) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 230) ? 14 + tile_rank_0_3 - 216 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 230) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 216) ? 126 + tile_rank_0_3 - 202 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 216) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 202) ? 14 + tile_rank_0_3 - 188 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 202) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 188) ? 28 + tile_rank_0_3 - 174 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 188) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 174) ? 140 + tile_rank_0_3 - 160 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 174) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 160) ? 28 + tile_rank_0_3 - 146 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 160) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 146) ? 42 + tile_rank_0_3 - 132 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 146) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 132) ? 154 + tile_rank_0_3 - 118 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 132) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 118) ? 42 + tile_rank_0_3 - 104 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 118) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 104) ? 56 + tile_rank_0_3 - 90 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 104) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 90) ? 56 + tile_rank_0_3 - 76 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 90) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 76) ? 70 + tile_rank_0_3 - 66 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 76) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 66) ? 168 + tile_rank_0_3 - 52 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 66) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 52) ? 70 + tile_rank_0_3 - 42 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 52) ? 1 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 42) ? 182 + tile_rank_0_3 - 28 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 42) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 28) ? 196 + tile_rank_0_3 - 14 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 28) ? 0 : split_idx_2_3);
                                    schedule_batch_idx_4 = ((tile_rank_0_3 < 14) ? 210 + tile_rank_0_3 : schedule_batch_idx_4);
                                    split_idx_2_3 = ((tile_rank_0_3 < 14) ? 0 : split_idx_2_3);
                                }
                            }
                            int part_count_3_3 = 1;
                            if (schedule_batch_idx_4 < split_requests_3) {
                                part_count_3_3 = 2;
                            }
                            schedule_batch_1 = (unsigned int)schedule_batch_idx_4;
                            next_part_1 = (unsigned int)(part_count_3_3 << 16 | split_idx_2_3);
                        }
                    }
                    unsigned int next_batch_1 = schedule_batch_1;
                    {
                        if (valid_1 != 0) {
                            {
                                {
                                    next_batch_1 = request_order[schedule_batch_1];
                                }
                            }
                        }
                    }
                    {
                        int packed_part_1 = (int)next_part_1;
                        split_idx_1_1 = packed_part_1 % 65536;
                        part_count_2_1 = packed_part_1 / 65536;
                    }
                    unsigned int valid_c = valid_1;
                    batch_idx_c = (int)next_batch_1;
                    q_row_idx_c = (int)next_q_row_1;
                    kv_head_idx_c = kv_head_idx_0_1;
                    split_idx_c = split_idx_1_1;
                    part_count_c = part_count_2_1;
                    bundle_idx_c = next_bundle_idx_1;
                    bundle_item_idx_c = next_bundle_item_idx_1;
                    if (valid_c == 0) {
                        break;
                    }
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
            int split_idx_4 = 0;
            int part_count_4 = NUM_SPLIT;
            int bundle_idx_2 = 0;
            int bundle_item_idx_2 = 0;
            {
                int flat_tile_idx_2 = blockIdx.x;
                int schedule_batch_idx_5 = 0;
                {
                    q_row_idx_2 = blockIdx.x;
                    bundle_idx_2 = blockIdx.z;
                    int tile_rank_4 = 0;
                    {
                        int starter_count_6 = batch_size - 152;
                        int starter_rank_4 = 74;
                        int tail_rank_4 = 158;
                        if (batch_size == 160) {
                            starter_rank_4 = 8;
                            tail_rank_4 = 68;
                        }
                        if (batch_size == 192) {
                            starter_rank_4 = 56;
                            tail_rank_4 = 188;
                        }
                        tile_rank_4 = starter_rank_4 + bundle_idx_2;
                        if (bundle_idx_2 >= starter_count_6) {
                            if (bundle_idx_2 < 152) {
                                int local_bundle_count_4 = 152 - starter_count_6;
                                int remaining_idx_4 = bundle_idx_2 - starter_count_6;
                                if (bundle_item_idx_2 != 0) {
                                    remaining_idx_4 = 2 * local_bundle_count_4 - 1 - remaining_idx_4;
                                }
                                tile_rank_4 = remaining_idx_4;
                                if (remaining_idx_4 >= starter_rank_4) {
                                    tile_rank_4 = remaining_idx_4 + starter_count_6;
                                }
                                if (remaining_idx_4 >= tail_rank_4 - starter_count_6) {
                                    tile_rank_4 = remaining_idx_4 + 2 * starter_count_6;
                                }
                            } else {
                                tile_rank_4 = tail_rank_4 + (starter_count_6 - 1 - (bundle_idx_2 - 152));
                            }
                        }
                    }
                    int tile_rank_0_4 = tile_rank_4;
                    int schedule_batch_idx_1_2 = 0;
                    int split_idx_2_4 = 0;
                    int split_requests_4 = 304 - batch_size;
                    {
                        if (batch_size == 160) {
                            schedule_batch_idx_1_2 = tile_rank_0_4 - 294;
                            split_idx_2_4 = 0;
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 294) ? tile_rank_0_4 - 284 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 294) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 284) ? 10 + tile_rank_0_4 - 274 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 284) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 274) ? 10 + tile_rank_0_4 - 264 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 274) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 264) ? 20 + tile_rank_0_4 - 254 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 264) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 254) ? 20 + tile_rank_0_4 - 244 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 254) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 244) ? 30 + tile_rank_0_4 - 234 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 244) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 234) ? 30 + tile_rank_0_4 - 224 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 234) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 224) ? 40 + tile_rank_0_4 - 214 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 224) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 214) ? 40 + tile_rank_0_4 - 204 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 214) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 204) ? 50 + tile_rank_0_4 - 194 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 204) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 194) ? 50 + tile_rank_0_4 - 184 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 194) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 184) ? 60 + tile_rank_0_4 - 174 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 184) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 174) ? 60 + tile_rank_0_4 - 164 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 174) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 164) ? 70 + tile_rank_0_4 - 154 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 164) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 154) ? 70 + tile_rank_0_4 - 144 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 154) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 144) ? 80 + tile_rank_0_4 - 134 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 144) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 134) ? 80 + tile_rank_0_4 - 124 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 134) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 124) ? 90 + tile_rank_0_4 - 114 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 124) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 114) ? 144 + tile_rank_0_4 - 108 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 114) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 108) ? 90 + tile_rank_0_4 - 98 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 108) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 98) ? 100 + tile_rank_0_4 - 88 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 98) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 88) ? 100 + tile_rank_0_4 - 78 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 88) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 78) ? 110 + tile_rank_0_4 - 68 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 78) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 68) ? 110 + tile_rank_0_4 - 58 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 68) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 58) ? 120 + tile_rank_0_4 - 48 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 58) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 48) ? 120 + tile_rank_0_4 - 38 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 48) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 38) ? 130 + tile_rank_0_4 - 28 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 38) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 28) ? 130 + tile_rank_0_4 - 18 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 28) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 18) ? 140 + tile_rank_0_4 - 14 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 18) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 14) ? 140 + tile_rank_0_4 - 10 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 14) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 10) ? 150 + tile_rank_0_4 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 10) ? 0 : split_idx_2_4);
                        } else if (batch_size == 192) {
                            schedule_batch_idx_1_2 = tile_rank_0_4 - 292;
                            split_idx_2_4 = 0;
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 292) ? tile_rank_0_4 - 280 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 292) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 280) ? 12 + tile_rank_0_4 - 268 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 280) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 268) ? 112 + tile_rank_0_4 - 260 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 268) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 260) ? 12 + tile_rank_0_4 - 248 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 260) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 248) ? 24 + tile_rank_0_4 - 236 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 248) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 236) ? 120 + tile_rank_0_4 - 224 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 236) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 224) ? 24 + tile_rank_0_4 - 212 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 224) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 212) ? 36 + tile_rank_0_4 - 200 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 212) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 200) ? 132 + tile_rank_0_4 - 188 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 200) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 188) ? 36 + tile_rank_0_4 - 176 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 188) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 176) ? 48 + tile_rank_0_4 - 164 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 176) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 164) ? 48 + tile_rank_0_4 - 152 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 164) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 152) ? 60 + tile_rank_0_4 - 140 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 152) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 140) ? 144 + tile_rank_0_4 - 128 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 140) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 128) ? 60 + tile_rank_0_4 - 116 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 128) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 116) ? 72 + tile_rank_0_4 - 104 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 116) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 104) ? 156 + tile_rank_0_4 - 92 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 104) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 92) ? 72 + tile_rank_0_4 - 80 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 92) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 80) ? 84 + tile_rank_0_4 - 68 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 80) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 68) ? 84 + tile_rank_0_4 - 56 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 68) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 56) ? 96 + tile_rank_0_4 - 44 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 56) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 44) ? 96 + tile_rank_0_4 - 32 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 44) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 32) ? 108 + tile_rank_0_4 - 28 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 32) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 28) ? 168 + tile_rank_0_4 - 16 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 28) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 16) ? 108 + tile_rank_0_4 - 12 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 16) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 12) ? 180 + tile_rank_0_4 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 12) ? 0 : split_idx_2_4);
                        } else {
                            schedule_batch_idx_1_2 = 80 + tile_rank_0_4 - 300;
                            split_idx_2_4 = 0;
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 300) ? 84 + tile_rank_0_4 - 286 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 300) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 286) ? 98 + tile_rank_0_4 - 272 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 286) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 272) ? 112 + tile_rank_0_4 - 258 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 272) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 258) ? tile_rank_0_4 - 244 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 258) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 244) ? tile_rank_0_4 - 230 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 244) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 230) ? 14 + tile_rank_0_4 - 216 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 230) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 216) ? 126 + tile_rank_0_4 - 202 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 216) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 202) ? 14 + tile_rank_0_4 - 188 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 202) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 188) ? 28 + tile_rank_0_4 - 174 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 188) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 174) ? 140 + tile_rank_0_4 - 160 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 174) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 160) ? 28 + tile_rank_0_4 - 146 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 160) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 146) ? 42 + tile_rank_0_4 - 132 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 146) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 132) ? 154 + tile_rank_0_4 - 118 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 132) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 118) ? 42 + tile_rank_0_4 - 104 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 118) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 104) ? 56 + tile_rank_0_4 - 90 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 104) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 90) ? 56 + tile_rank_0_4 - 76 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 90) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 76) ? 70 + tile_rank_0_4 - 66 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 76) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 66) ? 168 + tile_rank_0_4 - 52 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 66) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 52) ? 70 + tile_rank_0_4 - 42 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 52) ? 1 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 42) ? 182 + tile_rank_0_4 - 28 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 42) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 28) ? 196 + tile_rank_0_4 - 14 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 28) ? 0 : split_idx_2_4);
                            schedule_batch_idx_1_2 = ((tile_rank_0_4 < 14) ? 210 + tile_rank_0_4 : schedule_batch_idx_1_2);
                            split_idx_2_4 = ((tile_rank_0_4 < 14) ? 0 : split_idx_2_4);
                        }
                    }
                    int part_count_3_4 = 1;
                    if (schedule_batch_idx_1_2 < split_requests_4) {
                        part_count_3_4 = 2;
                    }
                    schedule_batch_idx_5 = schedule_batch_idx_1_2;
                    split_idx_4 = split_idx_2_4;
                    part_count_4 = part_count_3_4;
                }
                batch_idx_2 = schedule_batch_idx_5;
                {
                    {
                        batch_idx_2 = request_order[schedule_batch_idx_5];
                    }
                }
            }
            int batch_idx_m = batch_idx_2;
            int q_row_idx_m = q_row_idx_2;
            int kv_head_idx_m = kv_head_idx_2;
            int split_idx_m = split_idx_4;
            int part_count_m = part_count_4;
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
                    base_pairs_2 = total_pairs_2 / part_count_m;
                    extra_pairs_2 = total_pairs_2 % part_count_m;
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
                    int has_local_2 = 0;
                    {
                        int item_count_2 = 1;
                        {
                            int starter_count_7 = batch_size - 152;
                            if (bundle_idx_m >= starter_count_7) {
                                if (bundle_idx_m < 152) {
                                    item_count_2 = 2;
                                }
                            }
                        }
                        if (item_count_2 > bundle_item_idx_m + 1) {
                            has_local_2 = 1;
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
                    unsigned int valid_2 = 1;
                    unsigned int flat_or_q_2 = 0;
                    unsigned int default_part_2 = 0;
                    unsigned int default_batch_2 = 0;
                    if (has_local_2 == 0) {
                        uint32_t _clc_valid_4 = 0;
                        uint32_t _clc_ctaid_x_4;
                        uint32_t _clc_ctaid_y_4;
                        uint32_t _clc_ctaid_z_4;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%4];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %3, 1, 0, p1;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_x_4), "=r"(_clc_ctaid_y_4), "=r"(_clc_ctaid_z_4), "=r"(_clc_valid_4)
                            : "r"(work_response_view_addr + work_stage_m * 16 + 0 * 16)
                            : "memory");
                        valid_2 = _clc_valid_4;
                        flat_or_q_2 = _clc_ctaid_x_4;
                        default_part_2 = _clc_ctaid_y_4;
                        default_batch_2 = _clc_ctaid_z_4;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                    work_stage_m += 1;
                    if (work_stage_m == 2) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
                    unsigned int next_q_row_2 = 0;
                    unsigned int next_part_2 = 0;
                    unsigned int schedule_batch_2 = 0;
                    int kv_head_idx_0_2 = 0;
                    int split_idx_1_2 = 0;
                    int part_count_2_2 = NUM_SPLIT;
                    int next_bundle_idx_2 = 0;
                    int next_bundle_item_idx_2 = 0;
                    {
                        {
                            if (has_local_2 != 0) {
                                next_q_row_2 = (unsigned int)q_row_idx_m;
                                next_bundle_idx_2 = bundle_idx_m;
                                next_bundle_item_idx_2 = bundle_item_idx_m + 1;
                            } else {
                                next_q_row_2 = flat_or_q_2;
                                next_bundle_idx_2 = (int)default_batch_2;
                                next_bundle_item_idx_2 = 0;
                            }
                            int tile_rank_5 = 0;
                            {
                                int starter_count_8 = batch_size - 152;
                                int starter_rank_5 = 74;
                                int tail_rank_5 = 158;
                                if (batch_size == 160) {
                                    starter_rank_5 = 8;
                                    tail_rank_5 = 68;
                                }
                                if (batch_size == 192) {
                                    starter_rank_5 = 56;
                                    tail_rank_5 = 188;
                                }
                                tile_rank_5 = starter_rank_5 + next_bundle_idx_2;
                                if (next_bundle_idx_2 >= starter_count_8) {
                                    if (next_bundle_idx_2 < 152) {
                                        int local_bundle_count_5 = 152 - starter_count_8;
                                        int remaining_idx_5 = next_bundle_idx_2 - starter_count_8;
                                        if (next_bundle_item_idx_2 != 0) {
                                            remaining_idx_5 = 2 * local_bundle_count_5 - 1 - remaining_idx_5;
                                        }
                                        tile_rank_5 = remaining_idx_5;
                                        if (remaining_idx_5 >= starter_rank_5) {
                                            tile_rank_5 = remaining_idx_5 + starter_count_8;
                                        }
                                        if (remaining_idx_5 >= tail_rank_5 - starter_count_8) {
                                            tile_rank_5 = remaining_idx_5 + 2 * starter_count_8;
                                        }
                                    } else {
                                        tile_rank_5 = tail_rank_5 + (starter_count_8 - 1 - (next_bundle_idx_2 - 152));
                                    }
                                }
                            }
                            int tile_rank_0_5 = tile_rank_5;
                            int schedule_batch_idx_6 = 0;
                            int split_idx_2_5 = 0;
                            int split_requests_5 = 304 - batch_size;
                            {
                                if (batch_size == 160) {
                                    schedule_batch_idx_6 = tile_rank_0_5 - 294;
                                    split_idx_2_5 = 0;
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 294) ? tile_rank_0_5 - 284 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 294) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 284) ? 10 + tile_rank_0_5 - 274 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 284) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 274) ? 10 + tile_rank_0_5 - 264 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 274) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 264) ? 20 + tile_rank_0_5 - 254 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 264) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 254) ? 20 + tile_rank_0_5 - 244 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 254) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 244) ? 30 + tile_rank_0_5 - 234 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 244) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 234) ? 30 + tile_rank_0_5 - 224 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 234) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 224) ? 40 + tile_rank_0_5 - 214 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 224) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 214) ? 40 + tile_rank_0_5 - 204 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 214) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 204) ? 50 + tile_rank_0_5 - 194 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 204) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 194) ? 50 + tile_rank_0_5 - 184 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 194) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 184) ? 60 + tile_rank_0_5 - 174 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 184) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 174) ? 60 + tile_rank_0_5 - 164 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 174) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 164) ? 70 + tile_rank_0_5 - 154 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 164) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 154) ? 70 + tile_rank_0_5 - 144 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 154) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 144) ? 80 + tile_rank_0_5 - 134 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 144) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 134) ? 80 + tile_rank_0_5 - 124 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 134) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 124) ? 90 + tile_rank_0_5 - 114 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 124) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 114) ? 144 + tile_rank_0_5 - 108 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 114) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 108) ? 90 + tile_rank_0_5 - 98 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 108) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 98) ? 100 + tile_rank_0_5 - 88 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 98) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 88) ? 100 + tile_rank_0_5 - 78 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 88) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 78) ? 110 + tile_rank_0_5 - 68 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 78) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 68) ? 110 + tile_rank_0_5 - 58 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 68) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 58) ? 120 + tile_rank_0_5 - 48 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 58) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 48) ? 120 + tile_rank_0_5 - 38 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 48) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 38) ? 130 + tile_rank_0_5 - 28 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 38) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 28) ? 130 + tile_rank_0_5 - 18 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 28) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 18) ? 140 + tile_rank_0_5 - 14 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 18) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 14) ? 140 + tile_rank_0_5 - 10 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 14) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 10) ? 150 + tile_rank_0_5 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 10) ? 0 : split_idx_2_5);
                                } else if (batch_size == 192) {
                                    schedule_batch_idx_6 = tile_rank_0_5 - 292;
                                    split_idx_2_5 = 0;
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 292) ? tile_rank_0_5 - 280 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 292) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 280) ? 12 + tile_rank_0_5 - 268 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 280) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 268) ? 112 + tile_rank_0_5 - 260 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 268) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 260) ? 12 + tile_rank_0_5 - 248 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 260) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 248) ? 24 + tile_rank_0_5 - 236 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 248) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 236) ? 120 + tile_rank_0_5 - 224 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 236) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 224) ? 24 + tile_rank_0_5 - 212 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 224) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 212) ? 36 + tile_rank_0_5 - 200 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 212) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 200) ? 132 + tile_rank_0_5 - 188 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 200) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 188) ? 36 + tile_rank_0_5 - 176 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 188) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 176) ? 48 + tile_rank_0_5 - 164 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 176) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 164) ? 48 + tile_rank_0_5 - 152 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 164) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 152) ? 60 + tile_rank_0_5 - 140 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 152) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 140) ? 144 + tile_rank_0_5 - 128 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 140) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 128) ? 60 + tile_rank_0_5 - 116 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 128) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 116) ? 72 + tile_rank_0_5 - 104 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 116) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 104) ? 156 + tile_rank_0_5 - 92 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 104) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 92) ? 72 + tile_rank_0_5 - 80 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 92) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 80) ? 84 + tile_rank_0_5 - 68 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 80) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 68) ? 84 + tile_rank_0_5 - 56 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 68) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 56) ? 96 + tile_rank_0_5 - 44 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 56) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 44) ? 96 + tile_rank_0_5 - 32 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 44) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 32) ? 108 + tile_rank_0_5 - 28 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 32) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 28) ? 168 + tile_rank_0_5 - 16 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 28) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 16) ? 108 + tile_rank_0_5 - 12 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 16) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 12) ? 180 + tile_rank_0_5 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 12) ? 0 : split_idx_2_5);
                                } else {
                                    schedule_batch_idx_6 = 80 + tile_rank_0_5 - 300;
                                    split_idx_2_5 = 0;
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 300) ? 84 + tile_rank_0_5 - 286 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 300) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 286) ? 98 + tile_rank_0_5 - 272 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 286) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 272) ? 112 + tile_rank_0_5 - 258 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 272) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 258) ? tile_rank_0_5 - 244 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 258) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 244) ? tile_rank_0_5 - 230 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 244) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 230) ? 14 + tile_rank_0_5 - 216 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 230) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 216) ? 126 + tile_rank_0_5 - 202 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 216) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 202) ? 14 + tile_rank_0_5 - 188 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 202) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 188) ? 28 + tile_rank_0_5 - 174 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 188) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 174) ? 140 + tile_rank_0_5 - 160 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 174) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 160) ? 28 + tile_rank_0_5 - 146 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 160) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 146) ? 42 + tile_rank_0_5 - 132 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 146) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 132) ? 154 + tile_rank_0_5 - 118 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 132) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 118) ? 42 + tile_rank_0_5 - 104 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 118) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 104) ? 56 + tile_rank_0_5 - 90 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 104) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 90) ? 56 + tile_rank_0_5 - 76 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 90) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 76) ? 70 + tile_rank_0_5 - 66 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 76) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 66) ? 168 + tile_rank_0_5 - 52 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 66) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 52) ? 70 + tile_rank_0_5 - 42 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 52) ? 1 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 42) ? 182 + tile_rank_0_5 - 28 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 42) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 28) ? 196 + tile_rank_0_5 - 14 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 28) ? 0 : split_idx_2_5);
                                    schedule_batch_idx_6 = ((tile_rank_0_5 < 14) ? 210 + tile_rank_0_5 : schedule_batch_idx_6);
                                    split_idx_2_5 = ((tile_rank_0_5 < 14) ? 0 : split_idx_2_5);
                                }
                            }
                            int part_count_3_5 = 1;
                            if (schedule_batch_idx_6 < split_requests_5) {
                                part_count_3_5 = 2;
                            }
                            schedule_batch_2 = (unsigned int)schedule_batch_idx_6;
                            next_part_2 = (unsigned int)(part_count_3_5 << 16 | split_idx_2_5);
                        }
                    }
                    unsigned int next_batch_2 = schedule_batch_2;
                    {
                        if (valid_2 != 0) {
                            {
                                {
                                    next_batch_2 = request_order[schedule_batch_2];
                                }
                            }
                        }
                    }
                    {
                        int packed_part_2 = (int)next_part_2;
                        split_idx_1_2 = packed_part_2 % 65536;
                        part_count_2_2 = packed_part_2 / 65536;
                    }
                    unsigned int valid_m = valid_2;
                    batch_idx_m = (int)next_batch_2;
                    q_row_idx_m = (int)next_q_row_2;
                    kv_head_idx_m = kv_head_idx_0_2;
                    split_idx_m = split_idx_1_2;
                    part_count_m = part_count_2_2;
                    bundle_idx_m = next_bundle_idx_2;
                    bundle_item_idx_m = next_bundle_item_idx_2;
                    if (valid_m == 0) {
                        break;
                    }
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
            int split_idx_5 = 0;
            int part_count_5 = NUM_SPLIT;
            int bundle_idx_3 = 0;
            int bundle_item_idx_3 = 0;
            {
                int flat_tile_idx_3 = blockIdx.x;
                int schedule_batch_idx_7 = 0;
                {
                    q_row_idx_3 = blockIdx.x;
                    bundle_idx_3 = blockIdx.z;
                    int tile_rank_6 = 0;
                    {
                        int starter_count_9 = batch_size - 152;
                        int starter_rank_6 = 74;
                        int tail_rank_6 = 158;
                        if (batch_size == 160) {
                            starter_rank_6 = 8;
                            tail_rank_6 = 68;
                        }
                        if (batch_size == 192) {
                            starter_rank_6 = 56;
                            tail_rank_6 = 188;
                        }
                        tile_rank_6 = starter_rank_6 + bundle_idx_3;
                        if (bundle_idx_3 >= starter_count_9) {
                            if (bundle_idx_3 < 152) {
                                int local_bundle_count_6 = 152 - starter_count_9;
                                int remaining_idx_6 = bundle_idx_3 - starter_count_9;
                                if (bundle_item_idx_3 != 0) {
                                    remaining_idx_6 = 2 * local_bundle_count_6 - 1 - remaining_idx_6;
                                }
                                tile_rank_6 = remaining_idx_6;
                                if (remaining_idx_6 >= starter_rank_6) {
                                    tile_rank_6 = remaining_idx_6 + starter_count_9;
                                }
                                if (remaining_idx_6 >= tail_rank_6 - starter_count_9) {
                                    tile_rank_6 = remaining_idx_6 + 2 * starter_count_9;
                                }
                            } else {
                                tile_rank_6 = tail_rank_6 + (starter_count_9 - 1 - (bundle_idx_3 - 152));
                            }
                        }
                    }
                    int tile_rank_0_6 = tile_rank_6;
                    int schedule_batch_idx_1_3 = 0;
                    int split_idx_2_6 = 0;
                    int split_requests_6 = 304 - batch_size;
                    {
                        if (batch_size == 160) {
                            schedule_batch_idx_1_3 = tile_rank_0_6 - 294;
                            split_idx_2_6 = 0;
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 294) ? tile_rank_0_6 - 284 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 294) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 284) ? 10 + tile_rank_0_6 - 274 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 284) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 274) ? 10 + tile_rank_0_6 - 264 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 274) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 264) ? 20 + tile_rank_0_6 - 254 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 264) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 254) ? 20 + tile_rank_0_6 - 244 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 254) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 244) ? 30 + tile_rank_0_6 - 234 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 244) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 234) ? 30 + tile_rank_0_6 - 224 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 234) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 224) ? 40 + tile_rank_0_6 - 214 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 224) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 214) ? 40 + tile_rank_0_6 - 204 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 214) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 204) ? 50 + tile_rank_0_6 - 194 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 204) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 194) ? 50 + tile_rank_0_6 - 184 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 194) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 184) ? 60 + tile_rank_0_6 - 174 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 184) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 174) ? 60 + tile_rank_0_6 - 164 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 174) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 164) ? 70 + tile_rank_0_6 - 154 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 164) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 154) ? 70 + tile_rank_0_6 - 144 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 154) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 144) ? 80 + tile_rank_0_6 - 134 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 144) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 134) ? 80 + tile_rank_0_6 - 124 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 134) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 124) ? 90 + tile_rank_0_6 - 114 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 124) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 114) ? 144 + tile_rank_0_6 - 108 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 114) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 108) ? 90 + tile_rank_0_6 - 98 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 108) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 98) ? 100 + tile_rank_0_6 - 88 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 98) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 88) ? 100 + tile_rank_0_6 - 78 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 88) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 78) ? 110 + tile_rank_0_6 - 68 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 78) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 68) ? 110 + tile_rank_0_6 - 58 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 68) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 58) ? 120 + tile_rank_0_6 - 48 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 58) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 48) ? 120 + tile_rank_0_6 - 38 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 48) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 38) ? 130 + tile_rank_0_6 - 28 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 38) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 28) ? 130 + tile_rank_0_6 - 18 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 28) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 18) ? 140 + tile_rank_0_6 - 14 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 18) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 14) ? 140 + tile_rank_0_6 - 10 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 14) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 10) ? 150 + tile_rank_0_6 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 10) ? 0 : split_idx_2_6);
                        } else if (batch_size == 192) {
                            schedule_batch_idx_1_3 = tile_rank_0_6 - 292;
                            split_idx_2_6 = 0;
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 292) ? tile_rank_0_6 - 280 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 292) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 280) ? 12 + tile_rank_0_6 - 268 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 280) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 268) ? 112 + tile_rank_0_6 - 260 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 268) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 260) ? 12 + tile_rank_0_6 - 248 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 260) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 248) ? 24 + tile_rank_0_6 - 236 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 248) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 236) ? 120 + tile_rank_0_6 - 224 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 236) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 224) ? 24 + tile_rank_0_6 - 212 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 224) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 212) ? 36 + tile_rank_0_6 - 200 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 212) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 200) ? 132 + tile_rank_0_6 - 188 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 200) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 188) ? 36 + tile_rank_0_6 - 176 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 188) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 176) ? 48 + tile_rank_0_6 - 164 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 176) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 164) ? 48 + tile_rank_0_6 - 152 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 164) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 152) ? 60 + tile_rank_0_6 - 140 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 152) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 140) ? 144 + tile_rank_0_6 - 128 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 140) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 128) ? 60 + tile_rank_0_6 - 116 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 128) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 116) ? 72 + tile_rank_0_6 - 104 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 116) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 104) ? 156 + tile_rank_0_6 - 92 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 104) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 92) ? 72 + tile_rank_0_6 - 80 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 92) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 80) ? 84 + tile_rank_0_6 - 68 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 80) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 68) ? 84 + tile_rank_0_6 - 56 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 68) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 56) ? 96 + tile_rank_0_6 - 44 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 56) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 44) ? 96 + tile_rank_0_6 - 32 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 44) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 32) ? 108 + tile_rank_0_6 - 28 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 32) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 28) ? 168 + tile_rank_0_6 - 16 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 28) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 16) ? 108 + tile_rank_0_6 - 12 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 16) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 12) ? 180 + tile_rank_0_6 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 12) ? 0 : split_idx_2_6);
                        } else {
                            schedule_batch_idx_1_3 = 80 + tile_rank_0_6 - 300;
                            split_idx_2_6 = 0;
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 300) ? 84 + tile_rank_0_6 - 286 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 300) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 286) ? 98 + tile_rank_0_6 - 272 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 286) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 272) ? 112 + tile_rank_0_6 - 258 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 272) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 258) ? tile_rank_0_6 - 244 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 258) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 244) ? tile_rank_0_6 - 230 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 244) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 230) ? 14 + tile_rank_0_6 - 216 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 230) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 216) ? 126 + tile_rank_0_6 - 202 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 216) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 202) ? 14 + tile_rank_0_6 - 188 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 202) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 188) ? 28 + tile_rank_0_6 - 174 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 188) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 174) ? 140 + tile_rank_0_6 - 160 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 174) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 160) ? 28 + tile_rank_0_6 - 146 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 160) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 146) ? 42 + tile_rank_0_6 - 132 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 146) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 132) ? 154 + tile_rank_0_6 - 118 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 132) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 118) ? 42 + tile_rank_0_6 - 104 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 118) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 104) ? 56 + tile_rank_0_6 - 90 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 104) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 90) ? 56 + tile_rank_0_6 - 76 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 90) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 76) ? 70 + tile_rank_0_6 - 66 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 76) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 66) ? 168 + tile_rank_0_6 - 52 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 66) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 52) ? 70 + tile_rank_0_6 - 42 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 52) ? 1 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 42) ? 182 + tile_rank_0_6 - 28 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 42) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 28) ? 196 + tile_rank_0_6 - 14 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 28) ? 0 : split_idx_2_6);
                            schedule_batch_idx_1_3 = ((tile_rank_0_6 < 14) ? 210 + tile_rank_0_6 : schedule_batch_idx_1_3);
                            split_idx_2_6 = ((tile_rank_0_6 < 14) ? 0 : split_idx_2_6);
                        }
                    }
                    int part_count_3_6 = 1;
                    if (schedule_batch_idx_1_3 < split_requests_6) {
                        part_count_3_6 = 2;
                    }
                    schedule_batch_idx_7 = schedule_batch_idx_1_3;
                    split_idx_5 = split_idx_2_6;
                    part_count_5 = part_count_3_6;
                }
                batch_idx_3 = schedule_batch_idx_7;
                {
                    {
                        batch_idx_3 = request_order[schedule_batch_idx_7];
                    }
                }
            }
            int batch_idx_p = batch_idx_3;
            int q_row_idx_p = q_row_idx_3;
            int kv_head_idx_p = kv_head_idx_3;
            int split_idx_p = split_idx_5;
            int part_count_p = part_count_5;
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
                    base_pairs_3 = total_pairs_3 / part_count_p;
                    extra_pairs_3 = total_pairs_3 % part_count_p;
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
                    int has_local_3 = 0;
                    {
                        int item_count_3 = 1;
                        {
                            int starter_count_10 = batch_size - 152;
                            if (bundle_idx_p >= starter_count_10) {
                                if (bundle_idx_p < 152) {
                                    item_count_3 = 2;
                                }
                            }
                        }
                        if (item_count_3 > bundle_item_idx_p + 1) {
                            has_local_3 = 1;
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
                    unsigned int valid_3 = 1;
                    unsigned int flat_or_q_3 = 0;
                    unsigned int default_part_3 = 0;
                    unsigned int default_batch_3 = 0;
                    if (has_local_3 == 0) {
                        uint32_t _clc_valid_0 = 0;
                        uint32_t _clc_ctaid_x_0;
                        uint32_t _clc_ctaid_y_0;
                        uint32_t _clc_ctaid_z_0;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%4];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %3, 1, 0, p1;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_x_0), "=r"(_clc_ctaid_y_0), "=r"(_clc_ctaid_z_0), "=r"(_clc_valid_0)
                            : "r"(work_response_view_addr + work_stage_p * 16 + 0 * 16)
                            : "memory");
                        valid_3 = _clc_valid_0;
                        flat_or_q_3 = _clc_ctaid_x_0;
                        default_part_3 = _clc_ctaid_y_0;
                        default_batch_3 = _clc_ctaid_z_0;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
                    work_stage_p += 1;
                    if (work_stage_p == 2) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
                    unsigned int next_q_row_3 = 0;
                    unsigned int next_part_3 = 0;
                    unsigned int schedule_batch_3 = 0;
                    int kv_head_idx_0_3 = 0;
                    int split_idx_1_3 = 0;
                    int part_count_2_3 = NUM_SPLIT;
                    int next_bundle_idx_3 = 0;
                    int next_bundle_item_idx_3 = 0;
                    {
                        {
                            if (has_local_3 != 0) {
                                next_q_row_3 = (unsigned int)q_row_idx_p;
                                next_bundle_idx_3 = bundle_idx_p;
                                next_bundle_item_idx_3 = bundle_item_idx_p + 1;
                            } else {
                                next_q_row_3 = flat_or_q_3;
                                next_bundle_idx_3 = (int)default_batch_3;
                                next_bundle_item_idx_3 = 0;
                            }
                            int tile_rank_7 = 0;
                            {
                                int starter_count_11 = batch_size - 152;
                                int starter_rank_7 = 74;
                                int tail_rank_7 = 158;
                                if (batch_size == 160) {
                                    starter_rank_7 = 8;
                                    tail_rank_7 = 68;
                                }
                                if (batch_size == 192) {
                                    starter_rank_7 = 56;
                                    tail_rank_7 = 188;
                                }
                                tile_rank_7 = starter_rank_7 + next_bundle_idx_3;
                                if (next_bundle_idx_3 >= starter_count_11) {
                                    if (next_bundle_idx_3 < 152) {
                                        int local_bundle_count_7 = 152 - starter_count_11;
                                        int remaining_idx_7 = next_bundle_idx_3 - starter_count_11;
                                        if (next_bundle_item_idx_3 != 0) {
                                            remaining_idx_7 = 2 * local_bundle_count_7 - 1 - remaining_idx_7;
                                        }
                                        tile_rank_7 = remaining_idx_7;
                                        if (remaining_idx_7 >= starter_rank_7) {
                                            tile_rank_7 = remaining_idx_7 + starter_count_11;
                                        }
                                        if (remaining_idx_7 >= tail_rank_7 - starter_count_11) {
                                            tile_rank_7 = remaining_idx_7 + 2 * starter_count_11;
                                        }
                                    } else {
                                        tile_rank_7 = tail_rank_7 + (starter_count_11 - 1 - (next_bundle_idx_3 - 152));
                                    }
                                }
                            }
                            int tile_rank_0_7 = tile_rank_7;
                            int schedule_batch_idx_8 = 0;
                            int split_idx_2_7 = 0;
                            int split_requests_7 = 304 - batch_size;
                            {
                                if (batch_size == 160) {
                                    schedule_batch_idx_8 = tile_rank_0_7 - 294;
                                    split_idx_2_7 = 0;
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 294) ? tile_rank_0_7 - 284 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 294) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 284) ? 10 + tile_rank_0_7 - 274 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 284) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 274) ? 10 + tile_rank_0_7 - 264 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 274) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 264) ? 20 + tile_rank_0_7 - 254 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 264) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 254) ? 20 + tile_rank_0_7 - 244 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 254) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 244) ? 30 + tile_rank_0_7 - 234 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 244) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 234) ? 30 + tile_rank_0_7 - 224 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 234) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 224) ? 40 + tile_rank_0_7 - 214 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 224) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 214) ? 40 + tile_rank_0_7 - 204 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 214) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 204) ? 50 + tile_rank_0_7 - 194 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 204) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 194) ? 50 + tile_rank_0_7 - 184 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 194) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 184) ? 60 + tile_rank_0_7 - 174 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 184) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 174) ? 60 + tile_rank_0_7 - 164 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 174) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 164) ? 70 + tile_rank_0_7 - 154 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 164) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 154) ? 70 + tile_rank_0_7 - 144 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 154) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 144) ? 80 + tile_rank_0_7 - 134 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 144) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 134) ? 80 + tile_rank_0_7 - 124 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 134) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 124) ? 90 + tile_rank_0_7 - 114 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 124) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 114) ? 144 + tile_rank_0_7 - 108 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 114) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 108) ? 90 + tile_rank_0_7 - 98 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 108) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 98) ? 100 + tile_rank_0_7 - 88 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 98) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 88) ? 100 + tile_rank_0_7 - 78 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 88) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 78) ? 110 + tile_rank_0_7 - 68 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 78) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 68) ? 110 + tile_rank_0_7 - 58 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 68) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 58) ? 120 + tile_rank_0_7 - 48 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 58) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 48) ? 120 + tile_rank_0_7 - 38 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 48) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 38) ? 130 + tile_rank_0_7 - 28 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 38) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 28) ? 130 + tile_rank_0_7 - 18 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 28) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 18) ? 140 + tile_rank_0_7 - 14 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 18) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 14) ? 140 + tile_rank_0_7 - 10 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 14) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 10) ? 150 + tile_rank_0_7 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 10) ? 0 : split_idx_2_7);
                                } else if (batch_size == 192) {
                                    schedule_batch_idx_8 = tile_rank_0_7 - 292;
                                    split_idx_2_7 = 0;
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 292) ? tile_rank_0_7 - 280 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 292) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 280) ? 12 + tile_rank_0_7 - 268 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 280) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 268) ? 112 + tile_rank_0_7 - 260 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 268) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 260) ? 12 + tile_rank_0_7 - 248 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 260) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 248) ? 24 + tile_rank_0_7 - 236 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 248) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 236) ? 120 + tile_rank_0_7 - 224 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 236) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 224) ? 24 + tile_rank_0_7 - 212 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 224) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 212) ? 36 + tile_rank_0_7 - 200 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 212) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 200) ? 132 + tile_rank_0_7 - 188 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 200) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 188) ? 36 + tile_rank_0_7 - 176 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 188) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 176) ? 48 + tile_rank_0_7 - 164 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 176) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 164) ? 48 + tile_rank_0_7 - 152 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 164) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 152) ? 60 + tile_rank_0_7 - 140 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 152) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 140) ? 144 + tile_rank_0_7 - 128 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 140) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 128) ? 60 + tile_rank_0_7 - 116 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 128) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 116) ? 72 + tile_rank_0_7 - 104 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 116) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 104) ? 156 + tile_rank_0_7 - 92 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 104) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 92) ? 72 + tile_rank_0_7 - 80 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 92) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 80) ? 84 + tile_rank_0_7 - 68 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 80) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 68) ? 84 + tile_rank_0_7 - 56 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 68) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 56) ? 96 + tile_rank_0_7 - 44 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 56) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 44) ? 96 + tile_rank_0_7 - 32 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 44) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 32) ? 108 + tile_rank_0_7 - 28 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 32) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 28) ? 168 + tile_rank_0_7 - 16 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 28) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 16) ? 108 + tile_rank_0_7 - 12 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 16) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 12) ? 180 + tile_rank_0_7 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 12) ? 0 : split_idx_2_7);
                                } else {
                                    schedule_batch_idx_8 = 80 + tile_rank_0_7 - 300;
                                    split_idx_2_7 = 0;
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 300) ? 84 + tile_rank_0_7 - 286 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 300) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 286) ? 98 + tile_rank_0_7 - 272 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 286) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 272) ? 112 + tile_rank_0_7 - 258 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 272) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 258) ? tile_rank_0_7 - 244 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 258) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 244) ? tile_rank_0_7 - 230 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 244) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 230) ? 14 + tile_rank_0_7 - 216 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 230) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 216) ? 126 + tile_rank_0_7 - 202 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 216) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 202) ? 14 + tile_rank_0_7 - 188 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 202) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 188) ? 28 + tile_rank_0_7 - 174 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 188) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 174) ? 140 + tile_rank_0_7 - 160 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 174) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 160) ? 28 + tile_rank_0_7 - 146 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 160) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 146) ? 42 + tile_rank_0_7 - 132 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 146) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 132) ? 154 + tile_rank_0_7 - 118 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 132) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 118) ? 42 + tile_rank_0_7 - 104 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 118) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 104) ? 56 + tile_rank_0_7 - 90 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 104) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 90) ? 56 + tile_rank_0_7 - 76 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 90) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 76) ? 70 + tile_rank_0_7 - 66 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 76) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 66) ? 168 + tile_rank_0_7 - 52 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 66) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 52) ? 70 + tile_rank_0_7 - 42 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 52) ? 1 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 42) ? 182 + tile_rank_0_7 - 28 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 42) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 28) ? 196 + tile_rank_0_7 - 14 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 28) ? 0 : split_idx_2_7);
                                    schedule_batch_idx_8 = ((tile_rank_0_7 < 14) ? 210 + tile_rank_0_7 : schedule_batch_idx_8);
                                    split_idx_2_7 = ((tile_rank_0_7 < 14) ? 0 : split_idx_2_7);
                                }
                            }
                            int part_count_3_7 = 1;
                            if (schedule_batch_idx_8 < split_requests_7) {
                                part_count_3_7 = 2;
                            }
                            schedule_batch_3 = (unsigned int)schedule_batch_idx_8;
                            next_part_3 = (unsigned int)(part_count_3_7 << 16 | split_idx_2_7);
                        }
                    }
                    unsigned int next_batch_3 = schedule_batch_3;
                    {
                        if (valid_3 != 0) {
                            {
                                {
                                    next_batch_3 = request_order[schedule_batch_3];
                                }
                            }
                        }
                    }
                    {
                        int packed_part_3 = (int)next_part_3;
                        split_idx_1_3 = packed_part_3 % 65536;
                        part_count_2_3 = packed_part_3 / 65536;
                    }
                    unsigned int valid_p = valid_3;
                    batch_idx_p = (int)next_batch_3;
                    q_row_idx_p = (int)next_q_row_3;
                    kv_head_idx_p = kv_head_idx_0_3;
                    split_idx_p = split_idx_1_3;
                    part_count_p = part_count_2_3;
                    bundle_idx_p = next_bundle_idx_3;
                    bundle_item_idx_p = next_bundle_item_idx_3;
                    if (valid_p == 0) {
                        break;
                    }
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
                unsigned int work_stage_sched = 0;
                unsigned int throttle_stage_sched = 0;
                int batch_idx_4 = 0;
                int q_row_idx_4 = 0;
                int kv_head_idx_4 = 0;
                int split_idx_6 = 0;
                int part_count_6 = NUM_SPLIT;
                int bundle_idx_4 = 0;
                int bundle_item_idx_4 = 0;
                {
                    int flat_tile_idx_4 = blockIdx.x;
                    int schedule_batch_idx_9 = 0;
                    {
                        q_row_idx_4 = blockIdx.x;
                        bundle_idx_4 = blockIdx.z;
                        int tile_rank_8 = 0;
                        {
                            int starter_count_12 = batch_size - 152;
                            int starter_rank_8 = 74;
                            int tail_rank_8 = 158;
                            if (batch_size == 160) {
                                starter_rank_8 = 8;
                                tail_rank_8 = 68;
                            }
                            if (batch_size == 192) {
                                starter_rank_8 = 56;
                                tail_rank_8 = 188;
                            }
                            tile_rank_8 = starter_rank_8 + bundle_idx_4;
                            if (bundle_idx_4 >= starter_count_12) {
                                if (bundle_idx_4 < 152) {
                                    int local_bundle_count_8 = 152 - starter_count_12;
                                    int remaining_idx_8 = bundle_idx_4 - starter_count_12;
                                    if (bundle_item_idx_4 != 0) {
                                        remaining_idx_8 = 2 * local_bundle_count_8 - 1 - remaining_idx_8;
                                    }
                                    tile_rank_8 = remaining_idx_8;
                                    if (remaining_idx_8 >= starter_rank_8) {
                                        tile_rank_8 = remaining_idx_8 + starter_count_12;
                                    }
                                    if (remaining_idx_8 >= tail_rank_8 - starter_count_12) {
                                        tile_rank_8 = remaining_idx_8 + 2 * starter_count_12;
                                    }
                                } else {
                                    tile_rank_8 = tail_rank_8 + (starter_count_12 - 1 - (bundle_idx_4 - 152));
                                }
                            }
                        }
                        int tile_rank_0_8 = tile_rank_8;
                        int schedule_batch_idx_1_4 = 0;
                        int split_idx_2_8 = 0;
                        int split_requests_8 = 304 - batch_size;
                        {
                            if (batch_size == 160) {
                                schedule_batch_idx_1_4 = tile_rank_0_8 - 294;
                                split_idx_2_8 = 0;
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 294) ? tile_rank_0_8 - 284 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 294) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 284) ? 10 + tile_rank_0_8 - 274 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 284) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 274) ? 10 + tile_rank_0_8 - 264 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 274) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 264) ? 20 + tile_rank_0_8 - 254 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 264) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 254) ? 20 + tile_rank_0_8 - 244 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 254) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 244) ? 30 + tile_rank_0_8 - 234 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 244) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 234) ? 30 + tile_rank_0_8 - 224 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 234) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 224) ? 40 + tile_rank_0_8 - 214 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 224) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 214) ? 40 + tile_rank_0_8 - 204 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 214) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 204) ? 50 + tile_rank_0_8 - 194 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 204) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 194) ? 50 + tile_rank_0_8 - 184 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 194) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 184) ? 60 + tile_rank_0_8 - 174 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 184) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 174) ? 60 + tile_rank_0_8 - 164 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 174) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 164) ? 70 + tile_rank_0_8 - 154 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 164) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 154) ? 70 + tile_rank_0_8 - 144 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 154) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 144) ? 80 + tile_rank_0_8 - 134 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 144) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 134) ? 80 + tile_rank_0_8 - 124 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 134) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 124) ? 90 + tile_rank_0_8 - 114 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 124) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 114) ? 144 + tile_rank_0_8 - 108 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 114) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 108) ? 90 + tile_rank_0_8 - 98 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 108) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 98) ? 100 + tile_rank_0_8 - 88 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 98) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 88) ? 100 + tile_rank_0_8 - 78 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 88) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 78) ? 110 + tile_rank_0_8 - 68 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 78) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 68) ? 110 + tile_rank_0_8 - 58 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 68) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 58) ? 120 + tile_rank_0_8 - 48 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 58) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 48) ? 120 + tile_rank_0_8 - 38 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 48) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 38) ? 130 + tile_rank_0_8 - 28 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 38) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 28) ? 130 + tile_rank_0_8 - 18 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 28) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 18) ? 140 + tile_rank_0_8 - 14 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 18) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 14) ? 140 + tile_rank_0_8 - 10 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 14) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 10) ? 150 + tile_rank_0_8 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 10) ? 0 : split_idx_2_8);
                            } else if (batch_size == 192) {
                                schedule_batch_idx_1_4 = tile_rank_0_8 - 292;
                                split_idx_2_8 = 0;
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 292) ? tile_rank_0_8 - 280 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 292) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 280) ? 12 + tile_rank_0_8 - 268 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 280) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 268) ? 112 + tile_rank_0_8 - 260 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 268) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 260) ? 12 + tile_rank_0_8 - 248 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 260) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 248) ? 24 + tile_rank_0_8 - 236 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 248) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 236) ? 120 + tile_rank_0_8 - 224 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 236) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 224) ? 24 + tile_rank_0_8 - 212 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 224) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 212) ? 36 + tile_rank_0_8 - 200 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 212) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 200) ? 132 + tile_rank_0_8 - 188 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 200) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 188) ? 36 + tile_rank_0_8 - 176 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 188) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 176) ? 48 + tile_rank_0_8 - 164 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 176) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 164) ? 48 + tile_rank_0_8 - 152 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 164) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 152) ? 60 + tile_rank_0_8 - 140 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 152) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 140) ? 144 + tile_rank_0_8 - 128 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 140) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 128) ? 60 + tile_rank_0_8 - 116 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 128) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 116) ? 72 + tile_rank_0_8 - 104 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 116) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 104) ? 156 + tile_rank_0_8 - 92 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 104) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 92) ? 72 + tile_rank_0_8 - 80 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 92) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 80) ? 84 + tile_rank_0_8 - 68 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 80) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 68) ? 84 + tile_rank_0_8 - 56 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 68) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 56) ? 96 + tile_rank_0_8 - 44 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 56) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 44) ? 96 + tile_rank_0_8 - 32 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 44) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 32) ? 108 + tile_rank_0_8 - 28 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 32) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 28) ? 168 + tile_rank_0_8 - 16 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 28) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 16) ? 108 + tile_rank_0_8 - 12 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 16) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 12) ? 180 + tile_rank_0_8 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 12) ? 0 : split_idx_2_8);
                            } else {
                                schedule_batch_idx_1_4 = 80 + tile_rank_0_8 - 300;
                                split_idx_2_8 = 0;
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 300) ? 84 + tile_rank_0_8 - 286 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 300) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 286) ? 98 + tile_rank_0_8 - 272 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 286) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 272) ? 112 + tile_rank_0_8 - 258 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 272) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 258) ? tile_rank_0_8 - 244 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 258) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 244) ? tile_rank_0_8 - 230 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 244) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 230) ? 14 + tile_rank_0_8 - 216 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 230) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 216) ? 126 + tile_rank_0_8 - 202 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 216) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 202) ? 14 + tile_rank_0_8 - 188 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 202) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 188) ? 28 + tile_rank_0_8 - 174 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 188) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 174) ? 140 + tile_rank_0_8 - 160 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 174) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 160) ? 28 + tile_rank_0_8 - 146 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 160) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 146) ? 42 + tile_rank_0_8 - 132 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 146) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 132) ? 154 + tile_rank_0_8 - 118 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 132) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 118) ? 42 + tile_rank_0_8 - 104 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 118) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 104) ? 56 + tile_rank_0_8 - 90 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 104) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 90) ? 56 + tile_rank_0_8 - 76 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 90) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 76) ? 70 + tile_rank_0_8 - 66 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 76) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 66) ? 168 + tile_rank_0_8 - 52 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 66) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 52) ? 70 + tile_rank_0_8 - 42 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 52) ? 1 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 42) ? 182 + tile_rank_0_8 - 28 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 42) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 28) ? 196 + tile_rank_0_8 - 14 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 28) ? 0 : split_idx_2_8);
                                schedule_batch_idx_1_4 = ((tile_rank_0_8 < 14) ? 210 + tile_rank_0_8 : schedule_batch_idx_1_4);
                                split_idx_2_8 = ((tile_rank_0_8 < 14) ? 0 : split_idx_2_8);
                            }
                        }
                        int part_count_3_8 = 1;
                        if (schedule_batch_idx_1_4 < split_requests_8) {
                            part_count_3_8 = 2;
                        }
                        schedule_batch_idx_9 = schedule_batch_idx_1_4;
                        split_idx_6 = split_idx_2_8;
                        part_count_6 = part_count_3_8;
                    }
                    batch_idx_4 = schedule_batch_idx_9;
                    {
                        {
                            batch_idx_4 = request_order[schedule_batch_idx_9];
                        }
                    }
                }
                int batch_idx_sched = batch_idx_4;
                int q_row_idx_sched = q_row_idx_4;
                int kv_head_idx_sched = kv_head_idx_4;
                int split_idx_sched = split_idx_6;
                int part_count_sched = part_count_6;
                int bundle_idx_sched = bundle_idx_4;
                int bundle_item_idx_sched = bundle_item_idx_4;
                #pragma unroll 1
                for (unsigned int _tile_iter_sched = 0; _tile_iter_sched < total_tiles; _tile_iter_sched++) {
                    mbarrier_wait(throttle_full_addr + (throttle_stage_sched) * 8, _phase_throttle_full);
                    mbarrier_arrive(throttle_empty_addr + (throttle_stage_sched) * 8);
                    throttle_stage_sched += 1;
                    if (throttle_stage_sched == 2) { throttle_stage_sched = 0; _phase_throttle_full ^= 1; }
                    int has_local_sched = 0;
                    {
                        int item_count_4 = 1;
                        {
                            int starter_count_13 = batch_size - 152;
                            if (bundle_idx_sched >= starter_count_13) {
                                if (bundle_idx_sched < 152) {
                                    item_count_4 = 2;
                                }
                            }
                        }
                        if (item_count_4 > bundle_item_idx_sched + 1) {
                            has_local_sched = 1;
                        }
                    }
                    if (elect_sync()) {
                        mbarrier_wait(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty);
                        if (has_local_sched != 0) {
                            mbarrier_arrive(work_full_addr + (work_stage_sched) * 8);
                        } else {
                            mbarrier_arrive_expect_tx(work_full_addr + (work_stage_sched) * 8, 16);
                            asm volatile(
                                "fence.proxy.async.shared::cta;\n\t"
                                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                    " [%0], [%1];"
                                :: "r"(work_response_view_addr + work_stage_sched * 16 + 0 * 16), "r"(work_full_addr + work_stage_sched * 8)
                                : "memory");
                        }
                    }
                    int has_local_4 = 0;
                    {
                        int item_count_5 = 1;
                        {
                            int starter_count_14 = batch_size - 152;
                            if (bundle_idx_sched >= starter_count_14) {
                                if (bundle_idx_sched < 152) {
                                    item_count_5 = 2;
                                }
                            }
                        }
                        if (item_count_5 > bundle_item_idx_sched + 1) {
                            has_local_4 = 1;
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_sched) * 8, _phase_work_full_4);
                    unsigned int valid_4 = 1;
                    unsigned int _flat_or_q = 0;
                    unsigned int _default_part = 0;
                    unsigned int default_batch_4 = 0;
                    int next_bundle_idx_4 = bundle_idx_sched;
                    int next_bundle_item_idx_4 = bundle_item_idx_sched + 1;
                    if (has_local_4 == 0) {
                        uint32_t _clc_valid_1 = 0;
                        uint32_t _clc_ctaid_x_1;
                        uint32_t _clc_ctaid_y_1;
                        uint32_t _clc_ctaid_z_1;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%4];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %3, 1, 0, p1;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_x_1), "=r"(_clc_ctaid_y_1), "=r"(_clc_ctaid_z_1), "=r"(_clc_valid_1)
                            : "r"(work_response_view_addr + work_stage_sched * 16 + 0 * 16)
                            : "memory");
                        valid_4 = _clc_valid_1;
                        _flat_or_q = _clc_ctaid_x_1;
                        _default_part = _clc_ctaid_y_1;
                        default_batch_4 = _clc_ctaid_z_1;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        next_bundle_idx_4 = (int)default_batch_4;
                        next_bundle_item_idx_4 = 0;
                    }
                    mbarrier_arrive(work_empty_addr + (work_stage_sched) * 8);
                    work_stage_sched += 1;
                    if (work_stage_sched == 2) { work_stage_sched = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                    unsigned int valid_sched = valid_4;
                    bundle_idx_sched = next_bundle_idx_4;
                    bundle_item_idx_sched = next_bundle_item_idx_4;
                    if (valid_sched == 0) {
                        break;
                    }
                }
                #pragma unroll
                for (int _tail_sched = 0; _tail_sched < 2; _tail_sched++) {
                    mbarrier_wait(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty);
                    work_stage_sched += 1;
                    if (work_stage_sched == 2) { work_stage_sched = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                }
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
            int batch_idx_5 = 0;
            int q_row_idx_5 = 0;
            int kv_head_idx_5 = 0;
            int split_idx_7 = 0;
            int part_count_7 = NUM_SPLIT;
            int bundle_idx_5 = 0;
            int bundle_item_idx_5 = 0;
            {
                int flat_tile_idx_5 = blockIdx.x;
                int schedule_batch_idx_10 = 0;
                {
                    q_row_idx_5 = blockIdx.x;
                    bundle_idx_5 = blockIdx.z;
                    int tile_rank_9 = 0;
                    {
                        int starter_count_15 = batch_size - 152;
                        int starter_rank_9 = 74;
                        int tail_rank_9 = 158;
                        if (batch_size == 160) {
                            starter_rank_9 = 8;
                            tail_rank_9 = 68;
                        }
                        if (batch_size == 192) {
                            starter_rank_9 = 56;
                            tail_rank_9 = 188;
                        }
                        tile_rank_9 = starter_rank_9 + bundle_idx_5;
                        if (bundle_idx_5 >= starter_count_15) {
                            if (bundle_idx_5 < 152) {
                                int local_bundle_count_9 = 152 - starter_count_15;
                                int remaining_idx_9 = bundle_idx_5 - starter_count_15;
                                if (bundle_item_idx_5 != 0) {
                                    remaining_idx_9 = 2 * local_bundle_count_9 - 1 - remaining_idx_9;
                                }
                                tile_rank_9 = remaining_idx_9;
                                if (remaining_idx_9 >= starter_rank_9) {
                                    tile_rank_9 = remaining_idx_9 + starter_count_15;
                                }
                                if (remaining_idx_9 >= tail_rank_9 - starter_count_15) {
                                    tile_rank_9 = remaining_idx_9 + 2 * starter_count_15;
                                }
                            } else {
                                tile_rank_9 = tail_rank_9 + (starter_count_15 - 1 - (bundle_idx_5 - 152));
                            }
                        }
                    }
                    int tile_rank_0_9 = tile_rank_9;
                    int schedule_batch_idx_1_5 = 0;
                    int split_idx_2_9 = 0;
                    int split_requests_9 = 304 - batch_size;
                    {
                        if (batch_size == 160) {
                            schedule_batch_idx_1_5 = tile_rank_0_9 - 294;
                            split_idx_2_9 = 0;
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 294) ? tile_rank_0_9 - 284 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 294) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 284) ? 10 + tile_rank_0_9 - 274 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 284) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 274) ? 10 + tile_rank_0_9 - 264 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 274) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 264) ? 20 + tile_rank_0_9 - 254 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 264) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 254) ? 20 + tile_rank_0_9 - 244 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 254) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 244) ? 30 + tile_rank_0_9 - 234 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 244) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 234) ? 30 + tile_rank_0_9 - 224 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 234) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 224) ? 40 + tile_rank_0_9 - 214 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 224) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 214) ? 40 + tile_rank_0_9 - 204 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 214) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 204) ? 50 + tile_rank_0_9 - 194 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 204) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 194) ? 50 + tile_rank_0_9 - 184 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 194) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 184) ? 60 + tile_rank_0_9 - 174 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 184) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 174) ? 60 + tile_rank_0_9 - 164 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 174) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 164) ? 70 + tile_rank_0_9 - 154 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 164) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 154) ? 70 + tile_rank_0_9 - 144 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 154) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 144) ? 80 + tile_rank_0_9 - 134 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 144) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 134) ? 80 + tile_rank_0_9 - 124 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 134) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 124) ? 90 + tile_rank_0_9 - 114 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 124) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 114) ? 144 + tile_rank_0_9 - 108 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 114) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 108) ? 90 + tile_rank_0_9 - 98 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 108) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 98) ? 100 + tile_rank_0_9 - 88 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 98) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 88) ? 100 + tile_rank_0_9 - 78 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 88) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 78) ? 110 + tile_rank_0_9 - 68 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 78) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 68) ? 110 + tile_rank_0_9 - 58 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 68) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 58) ? 120 + tile_rank_0_9 - 48 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 58) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 48) ? 120 + tile_rank_0_9 - 38 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 48) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 38) ? 130 + tile_rank_0_9 - 28 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 38) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 28) ? 130 + tile_rank_0_9 - 18 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 28) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 18) ? 140 + tile_rank_0_9 - 14 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 18) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 14) ? 140 + tile_rank_0_9 - 10 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 14) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 10) ? 150 + tile_rank_0_9 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 10) ? 0 : split_idx_2_9);
                        } else if (batch_size == 192) {
                            schedule_batch_idx_1_5 = tile_rank_0_9 - 292;
                            split_idx_2_9 = 0;
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 292) ? tile_rank_0_9 - 280 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 292) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 280) ? 12 + tile_rank_0_9 - 268 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 280) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 268) ? 112 + tile_rank_0_9 - 260 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 268) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 260) ? 12 + tile_rank_0_9 - 248 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 260) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 248) ? 24 + tile_rank_0_9 - 236 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 248) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 236) ? 120 + tile_rank_0_9 - 224 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 236) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 224) ? 24 + tile_rank_0_9 - 212 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 224) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 212) ? 36 + tile_rank_0_9 - 200 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 212) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 200) ? 132 + tile_rank_0_9 - 188 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 200) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 188) ? 36 + tile_rank_0_9 - 176 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 188) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 176) ? 48 + tile_rank_0_9 - 164 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 176) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 164) ? 48 + tile_rank_0_9 - 152 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 164) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 152) ? 60 + tile_rank_0_9 - 140 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 152) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 140) ? 144 + tile_rank_0_9 - 128 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 140) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 128) ? 60 + tile_rank_0_9 - 116 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 128) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 116) ? 72 + tile_rank_0_9 - 104 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 116) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 104) ? 156 + tile_rank_0_9 - 92 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 104) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 92) ? 72 + tile_rank_0_9 - 80 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 92) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 80) ? 84 + tile_rank_0_9 - 68 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 80) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 68) ? 84 + tile_rank_0_9 - 56 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 68) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 56) ? 96 + tile_rank_0_9 - 44 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 56) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 44) ? 96 + tile_rank_0_9 - 32 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 44) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 32) ? 108 + tile_rank_0_9 - 28 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 32) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 28) ? 168 + tile_rank_0_9 - 16 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 28) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 16) ? 108 + tile_rank_0_9 - 12 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 16) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 12) ? 180 + tile_rank_0_9 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 12) ? 0 : split_idx_2_9);
                        } else {
                            schedule_batch_idx_1_5 = 80 + tile_rank_0_9 - 300;
                            split_idx_2_9 = 0;
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 300) ? 84 + tile_rank_0_9 - 286 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 300) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 286) ? 98 + tile_rank_0_9 - 272 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 286) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 272) ? 112 + tile_rank_0_9 - 258 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 272) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 258) ? tile_rank_0_9 - 244 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 258) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 244) ? tile_rank_0_9 - 230 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 244) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 230) ? 14 + tile_rank_0_9 - 216 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 230) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 216) ? 126 + tile_rank_0_9 - 202 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 216) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 202) ? 14 + tile_rank_0_9 - 188 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 202) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 188) ? 28 + tile_rank_0_9 - 174 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 188) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 174) ? 140 + tile_rank_0_9 - 160 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 174) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 160) ? 28 + tile_rank_0_9 - 146 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 160) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 146) ? 42 + tile_rank_0_9 - 132 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 146) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 132) ? 154 + tile_rank_0_9 - 118 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 132) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 118) ? 42 + tile_rank_0_9 - 104 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 118) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 104) ? 56 + tile_rank_0_9 - 90 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 104) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 90) ? 56 + tile_rank_0_9 - 76 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 90) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 76) ? 70 + tile_rank_0_9 - 66 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 76) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 66) ? 168 + tile_rank_0_9 - 52 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 66) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 52) ? 70 + tile_rank_0_9 - 42 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 52) ? 1 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 42) ? 182 + tile_rank_0_9 - 28 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 42) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 28) ? 196 + tile_rank_0_9 - 14 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 28) ? 0 : split_idx_2_9);
                            schedule_batch_idx_1_5 = ((tile_rank_0_9 < 14) ? 210 + tile_rank_0_9 : schedule_batch_idx_1_5);
                            split_idx_2_9 = ((tile_rank_0_9 < 14) ? 0 : split_idx_2_9);
                        }
                    }
                    int part_count_3_9 = 1;
                    if (schedule_batch_idx_1_5 < split_requests_9) {
                        part_count_3_9 = 2;
                    }
                    schedule_batch_idx_10 = schedule_batch_idx_1_5;
                    split_idx_7 = split_idx_2_9;
                    part_count_7 = part_count_3_9;
                }
                batch_idx_5 = schedule_batch_idx_10;
                {
                    {
                        batch_idx_5 = request_order[schedule_batch_idx_10];
                    }
                }
            }
            int batch_idx_0 = batch_idx_5;
            int q_row_idx_1_1 = q_row_idx_5;
            int kv_head_idx_2_1 = kv_head_idx_5;
            int split_idx_l = split_idx_7;
            int part_count_l = part_count_7;
            int bundle_idx_l = bundle_idx_5;
            int bundle_item_idx_l = bundle_item_idx_5;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < total_tiles; _tile_iter_l++) {
                {
                    mbarrier_wait(throttle_empty_addr + (throttle_stage_l) * 8, _phase_throttle_empty);
                    mbarrier_arrive(throttle_full_addr + (throttle_stage_l) * 8);
                    throttle_stage_l += 1;
                    if (throttle_stage_l == 2) { throttle_stage_l = 0; _phase_throttle_empty ^= 1; }
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
                    base_pairs_4 = total_pairs_4 / part_count_l;
                    extra_pairs_4 = total_pairs_4 % part_count_l;
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
                    int has_local_5 = 0;
                    {
                        int item_count_6 = 1;
                        {
                            int starter_count_16 = batch_size - 152;
                            if (bundle_idx_l >= starter_count_16) {
                                if (bundle_idx_l < 152) {
                                    item_count_6 = 2;
                                }
                            }
                        }
                        if (item_count_6 > bundle_item_idx_l + 1) {
                            has_local_5 = 1;
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_5);
                    unsigned int valid_5 = 1;
                    unsigned int flat_or_q_4 = 0;
                    unsigned int default_part_4 = 0;
                    unsigned int default_batch_5 = 0;
                    if (has_local_5 == 0) {
                        uint32_t _clc_valid_2 = 0;
                        uint32_t _clc_ctaid_x_2;
                        uint32_t _clc_ctaid_y_2;
                        uint32_t _clc_ctaid_z_2;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%4];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %3, 1, 0, p1;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_x_2), "=r"(_clc_ctaid_y_2), "=r"(_clc_ctaid_z_2), "=r"(_clc_valid_2)
                            : "r"(work_response_view_addr + work_stage_l * 16 + 0 * 16)
                            : "memory");
                        valid_5 = _clc_valid_2;
                        flat_or_q_4 = _clc_ctaid_x_2;
                        default_part_4 = _clc_ctaid_y_2;
                        default_batch_5 = _clc_ctaid_z_2;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                    work_stage_l += 1;
                    if (work_stage_l == 2) { work_stage_l = 0; _phase_work_full_5 ^= 1; }
                    unsigned int next_q_row_4 = 0;
                    unsigned int next_part_4 = 0;
                    unsigned int schedule_batch_4 = 0;
                    int kv_head_idx_0_4 = 0;
                    int split_idx_1_4 = 0;
                    int part_count_2_4 = NUM_SPLIT;
                    int next_bundle_idx_5 = 0;
                    int next_bundle_item_idx_5 = 0;
                    {
                        {
                            if (has_local_5 != 0) {
                                next_q_row_4 = (unsigned int)q_row_idx_1_1;
                                next_bundle_idx_5 = bundle_idx_l;
                                next_bundle_item_idx_5 = bundle_item_idx_l + 1;
                            } else {
                                next_q_row_4 = flat_or_q_4;
                                next_bundle_idx_5 = (int)default_batch_5;
                                next_bundle_item_idx_5 = 0;
                            }
                            int tile_rank_10 = 0;
                            {
                                int starter_count_17 = batch_size - 152;
                                int starter_rank_10 = 74;
                                int tail_rank_10 = 158;
                                if (batch_size == 160) {
                                    starter_rank_10 = 8;
                                    tail_rank_10 = 68;
                                }
                                if (batch_size == 192) {
                                    starter_rank_10 = 56;
                                    tail_rank_10 = 188;
                                }
                                tile_rank_10 = starter_rank_10 + next_bundle_idx_5;
                                if (next_bundle_idx_5 >= starter_count_17) {
                                    if (next_bundle_idx_5 < 152) {
                                        int local_bundle_count_10 = 152 - starter_count_17;
                                        int remaining_idx_10 = next_bundle_idx_5 - starter_count_17;
                                        if (next_bundle_item_idx_5 != 0) {
                                            remaining_idx_10 = 2 * local_bundle_count_10 - 1 - remaining_idx_10;
                                        }
                                        tile_rank_10 = remaining_idx_10;
                                        if (remaining_idx_10 >= starter_rank_10) {
                                            tile_rank_10 = remaining_idx_10 + starter_count_17;
                                        }
                                        if (remaining_idx_10 >= tail_rank_10 - starter_count_17) {
                                            tile_rank_10 = remaining_idx_10 + 2 * starter_count_17;
                                        }
                                    } else {
                                        tile_rank_10 = tail_rank_10 + (starter_count_17 - 1 - (next_bundle_idx_5 - 152));
                                    }
                                }
                            }
                            int tile_rank_0_10 = tile_rank_10;
                            int schedule_batch_idx_11 = 0;
                            int split_idx_2_10 = 0;
                            int split_requests_10 = 304 - batch_size;
                            {
                                if (batch_size == 160) {
                                    schedule_batch_idx_11 = tile_rank_0_10 - 294;
                                    split_idx_2_10 = 0;
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 294) ? tile_rank_0_10 - 284 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 294) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 284) ? 10 + tile_rank_0_10 - 274 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 284) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 274) ? 10 + tile_rank_0_10 - 264 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 274) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 264) ? 20 + tile_rank_0_10 - 254 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 264) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 254) ? 20 + tile_rank_0_10 - 244 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 254) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 244) ? 30 + tile_rank_0_10 - 234 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 244) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 234) ? 30 + tile_rank_0_10 - 224 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 234) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 224) ? 40 + tile_rank_0_10 - 214 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 224) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 214) ? 40 + tile_rank_0_10 - 204 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 214) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 204) ? 50 + tile_rank_0_10 - 194 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 204) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 194) ? 50 + tile_rank_0_10 - 184 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 194) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 184) ? 60 + tile_rank_0_10 - 174 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 184) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 174) ? 60 + tile_rank_0_10 - 164 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 174) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 164) ? 70 + tile_rank_0_10 - 154 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 164) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 154) ? 70 + tile_rank_0_10 - 144 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 154) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 144) ? 80 + tile_rank_0_10 - 134 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 144) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 134) ? 80 + tile_rank_0_10 - 124 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 134) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 124) ? 90 + tile_rank_0_10 - 114 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 124) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 114) ? 144 + tile_rank_0_10 - 108 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 114) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 108) ? 90 + tile_rank_0_10 - 98 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 108) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 98) ? 100 + tile_rank_0_10 - 88 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 98) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 88) ? 100 + tile_rank_0_10 - 78 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 88) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 78) ? 110 + tile_rank_0_10 - 68 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 78) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 68) ? 110 + tile_rank_0_10 - 58 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 68) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 58) ? 120 + tile_rank_0_10 - 48 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 58) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 48) ? 120 + tile_rank_0_10 - 38 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 48) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 38) ? 130 + tile_rank_0_10 - 28 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 38) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 28) ? 130 + tile_rank_0_10 - 18 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 28) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 18) ? 140 + tile_rank_0_10 - 14 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 18) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 14) ? 140 + tile_rank_0_10 - 10 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 14) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 10) ? 150 + tile_rank_0_10 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 10) ? 0 : split_idx_2_10);
                                } else if (batch_size == 192) {
                                    schedule_batch_idx_11 = tile_rank_0_10 - 292;
                                    split_idx_2_10 = 0;
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 292) ? tile_rank_0_10 - 280 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 292) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 280) ? 12 + tile_rank_0_10 - 268 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 280) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 268) ? 112 + tile_rank_0_10 - 260 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 268) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 260) ? 12 + tile_rank_0_10 - 248 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 260) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 248) ? 24 + tile_rank_0_10 - 236 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 248) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 236) ? 120 + tile_rank_0_10 - 224 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 236) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 224) ? 24 + tile_rank_0_10 - 212 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 224) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 212) ? 36 + tile_rank_0_10 - 200 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 212) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 200) ? 132 + tile_rank_0_10 - 188 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 200) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 188) ? 36 + tile_rank_0_10 - 176 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 188) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 176) ? 48 + tile_rank_0_10 - 164 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 176) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 164) ? 48 + tile_rank_0_10 - 152 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 164) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 152) ? 60 + tile_rank_0_10 - 140 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 152) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 140) ? 144 + tile_rank_0_10 - 128 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 140) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 128) ? 60 + tile_rank_0_10 - 116 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 128) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 116) ? 72 + tile_rank_0_10 - 104 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 116) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 104) ? 156 + tile_rank_0_10 - 92 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 104) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 92) ? 72 + tile_rank_0_10 - 80 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 92) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 80) ? 84 + tile_rank_0_10 - 68 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 80) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 68) ? 84 + tile_rank_0_10 - 56 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 68) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 56) ? 96 + tile_rank_0_10 - 44 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 56) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 44) ? 96 + tile_rank_0_10 - 32 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 44) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 32) ? 108 + tile_rank_0_10 - 28 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 32) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 28) ? 168 + tile_rank_0_10 - 16 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 28) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 16) ? 108 + tile_rank_0_10 - 12 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 16) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 12) ? 180 + tile_rank_0_10 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 12) ? 0 : split_idx_2_10);
                                } else {
                                    schedule_batch_idx_11 = 80 + tile_rank_0_10 - 300;
                                    split_idx_2_10 = 0;
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 300) ? 84 + tile_rank_0_10 - 286 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 300) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 286) ? 98 + tile_rank_0_10 - 272 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 286) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 272) ? 112 + tile_rank_0_10 - 258 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 272) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 258) ? tile_rank_0_10 - 244 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 258) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 244) ? tile_rank_0_10 - 230 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 244) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 230) ? 14 + tile_rank_0_10 - 216 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 230) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 216) ? 126 + tile_rank_0_10 - 202 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 216) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 202) ? 14 + tile_rank_0_10 - 188 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 202) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 188) ? 28 + tile_rank_0_10 - 174 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 188) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 174) ? 140 + tile_rank_0_10 - 160 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 174) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 160) ? 28 + tile_rank_0_10 - 146 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 160) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 146) ? 42 + tile_rank_0_10 - 132 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 146) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 132) ? 154 + tile_rank_0_10 - 118 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 132) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 118) ? 42 + tile_rank_0_10 - 104 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 118) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 104) ? 56 + tile_rank_0_10 - 90 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 104) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 90) ? 56 + tile_rank_0_10 - 76 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 90) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 76) ? 70 + tile_rank_0_10 - 66 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 76) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 66) ? 168 + tile_rank_0_10 - 52 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 66) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 52) ? 70 + tile_rank_0_10 - 42 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 52) ? 1 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 42) ? 182 + tile_rank_0_10 - 28 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 42) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 28) ? 196 + tile_rank_0_10 - 14 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 28) ? 0 : split_idx_2_10);
                                    schedule_batch_idx_11 = ((tile_rank_0_10 < 14) ? 210 + tile_rank_0_10 : schedule_batch_idx_11);
                                    split_idx_2_10 = ((tile_rank_0_10 < 14) ? 0 : split_idx_2_10);
                                }
                            }
                            int part_count_3_10 = 1;
                            if (schedule_batch_idx_11 < split_requests_10) {
                                part_count_3_10 = 2;
                            }
                            schedule_batch_4 = (unsigned int)schedule_batch_idx_11;
                            next_part_4 = (unsigned int)(part_count_3_10 << 16 | split_idx_2_10);
                        }
                    }
                    unsigned int next_batch_4 = schedule_batch_4;
                    {
                        if (valid_5 != 0) {
                            {
                                {
                                    next_batch_4 = request_order[schedule_batch_4];
                                }
                            }
                        }
                    }
                    {
                        int packed_part_4 = (int)next_part_4;
                        split_idx_1_4 = packed_part_4 % 65536;
                        part_count_2_4 = packed_part_4 / 65536;
                    }
                    unsigned int valid_l = valid_5;
                    batch_idx_0 = (int)next_batch_4;
                    q_row_idx_1_1 = (int)next_q_row_4;
                    kv_head_idx_2_1 = kv_head_idx_0_4;
                    split_idx_l = split_idx_1_4;
                    part_count_l = part_count_2_4;
                    bundle_idx_l = next_bundle_idx_5;
                    bundle_item_idx_l = next_bundle_item_idx_5;
                    if (valid_l == 0) {
                        break;
                    }
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
            int batch_idx_6 = 0;
            int q_row_idx_6 = 0;
            int kv_head_idx_6 = 0;
            int split_idx_8 = 0;
            int part_count_8 = NUM_SPLIT;
            int bundle_idx_6 = 0;
            int bundle_item_idx_6 = 0;
            {
                int flat_tile_idx_6 = blockIdx.x;
                int schedule_batch_idx_12 = 0;
                {
                    q_row_idx_6 = blockIdx.x;
                    bundle_idx_6 = blockIdx.z;
                    int tile_rank_11 = 0;
                    {
                        int starter_count_18 = batch_size - 152;
                        int starter_rank_11 = 74;
                        int tail_rank_11 = 158;
                        if (batch_size == 160) {
                            starter_rank_11 = 8;
                            tail_rank_11 = 68;
                        }
                        if (batch_size == 192) {
                            starter_rank_11 = 56;
                            tail_rank_11 = 188;
                        }
                        tile_rank_11 = starter_rank_11 + bundle_idx_6;
                        if (bundle_idx_6 >= starter_count_18) {
                            if (bundle_idx_6 < 152) {
                                int local_bundle_count_11 = 152 - starter_count_18;
                                int remaining_idx_11 = bundle_idx_6 - starter_count_18;
                                if (bundle_item_idx_6 != 0) {
                                    remaining_idx_11 = 2 * local_bundle_count_11 - 1 - remaining_idx_11;
                                }
                                tile_rank_11 = remaining_idx_11;
                                if (remaining_idx_11 >= starter_rank_11) {
                                    tile_rank_11 = remaining_idx_11 + starter_count_18;
                                }
                                if (remaining_idx_11 >= tail_rank_11 - starter_count_18) {
                                    tile_rank_11 = remaining_idx_11 + 2 * starter_count_18;
                                }
                            } else {
                                tile_rank_11 = tail_rank_11 + (starter_count_18 - 1 - (bundle_idx_6 - 152));
                            }
                        }
                    }
                    int tile_rank_0_11 = tile_rank_11;
                    int schedule_batch_idx_1_6 = 0;
                    int split_idx_2_11 = 0;
                    int split_requests_11 = 304 - batch_size;
                    {
                        if (batch_size == 160) {
                            schedule_batch_idx_1_6 = tile_rank_0_11 - 294;
                            split_idx_2_11 = 0;
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 294) ? tile_rank_0_11 - 284 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 294) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 284) ? 10 + tile_rank_0_11 - 274 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 284) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 274) ? 10 + tile_rank_0_11 - 264 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 274) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 264) ? 20 + tile_rank_0_11 - 254 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 264) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 254) ? 20 + tile_rank_0_11 - 244 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 254) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 244) ? 30 + tile_rank_0_11 - 234 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 244) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 234) ? 30 + tile_rank_0_11 - 224 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 234) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 224) ? 40 + tile_rank_0_11 - 214 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 224) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 214) ? 40 + tile_rank_0_11 - 204 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 214) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 204) ? 50 + tile_rank_0_11 - 194 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 204) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 194) ? 50 + tile_rank_0_11 - 184 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 194) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 184) ? 60 + tile_rank_0_11 - 174 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 184) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 174) ? 60 + tile_rank_0_11 - 164 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 174) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 164) ? 70 + tile_rank_0_11 - 154 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 164) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 154) ? 70 + tile_rank_0_11 - 144 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 154) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 144) ? 80 + tile_rank_0_11 - 134 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 144) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 134) ? 80 + tile_rank_0_11 - 124 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 134) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 124) ? 90 + tile_rank_0_11 - 114 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 124) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 114) ? 144 + tile_rank_0_11 - 108 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 114) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 108) ? 90 + tile_rank_0_11 - 98 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 108) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 98) ? 100 + tile_rank_0_11 - 88 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 98) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 88) ? 100 + tile_rank_0_11 - 78 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 88) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 78) ? 110 + tile_rank_0_11 - 68 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 78) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 68) ? 110 + tile_rank_0_11 - 58 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 68) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 58) ? 120 + tile_rank_0_11 - 48 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 58) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 48) ? 120 + tile_rank_0_11 - 38 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 48) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 38) ? 130 + tile_rank_0_11 - 28 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 38) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 28) ? 130 + tile_rank_0_11 - 18 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 28) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 18) ? 140 + tile_rank_0_11 - 14 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 18) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 14) ? 140 + tile_rank_0_11 - 10 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 14) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 10) ? 150 + tile_rank_0_11 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 10) ? 0 : split_idx_2_11);
                        } else if (batch_size == 192) {
                            schedule_batch_idx_1_6 = tile_rank_0_11 - 292;
                            split_idx_2_11 = 0;
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 292) ? tile_rank_0_11 - 280 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 292) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 280) ? 12 + tile_rank_0_11 - 268 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 280) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 268) ? 112 + tile_rank_0_11 - 260 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 268) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 260) ? 12 + tile_rank_0_11 - 248 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 260) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 248) ? 24 + tile_rank_0_11 - 236 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 248) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 236) ? 120 + tile_rank_0_11 - 224 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 236) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 224) ? 24 + tile_rank_0_11 - 212 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 224) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 212) ? 36 + tile_rank_0_11 - 200 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 212) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 200) ? 132 + tile_rank_0_11 - 188 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 200) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 188) ? 36 + tile_rank_0_11 - 176 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 188) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 176) ? 48 + tile_rank_0_11 - 164 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 176) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 164) ? 48 + tile_rank_0_11 - 152 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 164) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 152) ? 60 + tile_rank_0_11 - 140 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 152) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 140) ? 144 + tile_rank_0_11 - 128 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 140) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 128) ? 60 + tile_rank_0_11 - 116 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 128) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 116) ? 72 + tile_rank_0_11 - 104 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 116) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 104) ? 156 + tile_rank_0_11 - 92 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 104) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 92) ? 72 + tile_rank_0_11 - 80 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 92) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 80) ? 84 + tile_rank_0_11 - 68 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 80) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 68) ? 84 + tile_rank_0_11 - 56 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 68) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 56) ? 96 + tile_rank_0_11 - 44 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 56) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 44) ? 96 + tile_rank_0_11 - 32 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 44) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 32) ? 108 + tile_rank_0_11 - 28 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 32) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 28) ? 168 + tile_rank_0_11 - 16 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 28) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 16) ? 108 + tile_rank_0_11 - 12 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 16) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 12) ? 180 + tile_rank_0_11 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 12) ? 0 : split_idx_2_11);
                        } else {
                            schedule_batch_idx_1_6 = 80 + tile_rank_0_11 - 300;
                            split_idx_2_11 = 0;
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 300) ? 84 + tile_rank_0_11 - 286 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 300) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 286) ? 98 + tile_rank_0_11 - 272 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 286) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 272) ? 112 + tile_rank_0_11 - 258 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 272) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 258) ? tile_rank_0_11 - 244 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 258) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 244) ? tile_rank_0_11 - 230 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 244) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 230) ? 14 + tile_rank_0_11 - 216 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 230) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 216) ? 126 + tile_rank_0_11 - 202 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 216) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 202) ? 14 + tile_rank_0_11 - 188 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 202) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 188) ? 28 + tile_rank_0_11 - 174 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 188) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 174) ? 140 + tile_rank_0_11 - 160 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 174) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 160) ? 28 + tile_rank_0_11 - 146 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 160) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 146) ? 42 + tile_rank_0_11 - 132 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 146) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 132) ? 154 + tile_rank_0_11 - 118 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 132) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 118) ? 42 + tile_rank_0_11 - 104 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 118) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 104) ? 56 + tile_rank_0_11 - 90 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 104) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 90) ? 56 + tile_rank_0_11 - 76 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 90) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 76) ? 70 + tile_rank_0_11 - 66 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 76) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 66) ? 168 + tile_rank_0_11 - 52 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 66) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 52) ? 70 + tile_rank_0_11 - 42 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 52) ? 1 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 42) ? 182 + tile_rank_0_11 - 28 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 42) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 28) ? 196 + tile_rank_0_11 - 14 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 28) ? 0 : split_idx_2_11);
                            schedule_batch_idx_1_6 = ((tile_rank_0_11 < 14) ? 210 + tile_rank_0_11 : schedule_batch_idx_1_6);
                            split_idx_2_11 = ((tile_rank_0_11 < 14) ? 0 : split_idx_2_11);
                        }
                    }
                    int part_count_3_11 = 1;
                    if (schedule_batch_idx_1_6 < split_requests_11) {
                        part_count_3_11 = 2;
                    }
                    schedule_batch_idx_12 = schedule_batch_idx_1_6;
                    split_idx_8 = split_idx_2_11;
                    part_count_8 = part_count_3_11;
                }
                batch_idx_6 = schedule_batch_idx_12;
                {
                    {
                        batch_idx_6 = request_order[schedule_batch_idx_12];
                    }
                }
            }
            int batch_idx_t = batch_idx_6;
            int q_row_idx_t = q_row_idx_6;
            int kv_head_idx_t = kv_head_idx_6;
            int split_idx_t = split_idx_8;
            int part_count_t = part_count_8;
            int bundle_idx_t = bundle_idx_6;
            int bundle_item_idx_t = bundle_item_idx_6;
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
                    base_pairs_5 = total_pairs_5 / part_count_t;
                    extra_pairs_5 = total_pairs_5 % part_count_t;
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
                    int has_local_6 = 0;
                    {
                        int item_count_7 = 1;
                        {
                            int starter_count_19 = batch_size - 152;
                            if (bundle_idx_t >= starter_count_19) {
                                if (bundle_idx_t < 152) {
                                    item_count_7 = 2;
                                }
                            }
                        }
                        if (item_count_7 > bundle_item_idx_t + 1) {
                            has_local_6 = 1;
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_t) * 8, _phase_work_full_6);
                    unsigned int valid_6 = 1;
                    unsigned int flat_or_q_5 = 0;
                    unsigned int default_part_5 = 0;
                    unsigned int default_batch_6 = 0;
                    if (has_local_6 == 0) {
                        uint32_t _clc_valid_3 = 0;
                        uint32_t _clc_ctaid_x_3;
                        uint32_t _clc_ctaid_y_3;
                        uint32_t _clc_ctaid_z_3;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%4];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %3, 1, 0, p1;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_x_3), "=r"(_clc_ctaid_y_3), "=r"(_clc_ctaid_z_3), "=r"(_clc_valid_3)
                            : "r"(work_response_view_addr + work_stage_t * 16 + 0 * 16)
                            : "memory");
                        valid_6 = _clc_valid_3;
                        flat_or_q_5 = _clc_ctaid_x_3;
                        default_part_5 = _clc_ctaid_y_3;
                        default_batch_6 = _clc_ctaid_z_3;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    mbarrier_arrive(work_empty_addr + (work_stage_t) * 8);
                    work_stage_t += 1;
                    if (work_stage_t == 2) { work_stage_t = 0; _phase_work_full_6 ^= 1; }
                    unsigned int next_q_row_5 = 0;
                    unsigned int next_part_5 = 0;
                    unsigned int schedule_batch_5 = 0;
                    int kv_head_idx_0_5 = 0;
                    int split_idx_1_5 = 0;
                    int part_count_2_5 = NUM_SPLIT;
                    int next_bundle_idx_6 = 0;
                    int next_bundle_item_idx_6 = 0;
                    {
                        {
                            if (has_local_6 != 0) {
                                next_q_row_5 = (unsigned int)q_row_idx_t;
                                next_bundle_idx_6 = bundle_idx_t;
                                next_bundle_item_idx_6 = bundle_item_idx_t + 1;
                            } else {
                                next_q_row_5 = flat_or_q_5;
                                next_bundle_idx_6 = (int)default_batch_6;
                                next_bundle_item_idx_6 = 0;
                            }
                            int tile_rank_12 = 0;
                            {
                                int starter_count_20 = batch_size - 152;
                                int starter_rank_12 = 74;
                                int tail_rank_12 = 158;
                                if (batch_size == 160) {
                                    starter_rank_12 = 8;
                                    tail_rank_12 = 68;
                                }
                                if (batch_size == 192) {
                                    starter_rank_12 = 56;
                                    tail_rank_12 = 188;
                                }
                                tile_rank_12 = starter_rank_12 + next_bundle_idx_6;
                                if (next_bundle_idx_6 >= starter_count_20) {
                                    if (next_bundle_idx_6 < 152) {
                                        int local_bundle_count_12 = 152 - starter_count_20;
                                        int remaining_idx_12 = next_bundle_idx_6 - starter_count_20;
                                        if (next_bundle_item_idx_6 != 0) {
                                            remaining_idx_12 = 2 * local_bundle_count_12 - 1 - remaining_idx_12;
                                        }
                                        tile_rank_12 = remaining_idx_12;
                                        if (remaining_idx_12 >= starter_rank_12) {
                                            tile_rank_12 = remaining_idx_12 + starter_count_20;
                                        }
                                        if (remaining_idx_12 >= tail_rank_12 - starter_count_20) {
                                            tile_rank_12 = remaining_idx_12 + 2 * starter_count_20;
                                        }
                                    } else {
                                        tile_rank_12 = tail_rank_12 + (starter_count_20 - 1 - (next_bundle_idx_6 - 152));
                                    }
                                }
                            }
                            int tile_rank_0_12 = tile_rank_12;
                            int schedule_batch_idx_13 = 0;
                            int split_idx_2_12 = 0;
                            int split_requests_12 = 304 - batch_size;
                            {
                                if (batch_size == 160) {
                                    schedule_batch_idx_13 = tile_rank_0_12 - 294;
                                    split_idx_2_12 = 0;
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 294) ? tile_rank_0_12 - 284 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 294) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 284) ? 10 + tile_rank_0_12 - 274 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 284) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 274) ? 10 + tile_rank_0_12 - 264 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 274) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 264) ? 20 + tile_rank_0_12 - 254 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 264) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 254) ? 20 + tile_rank_0_12 - 244 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 254) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 244) ? 30 + tile_rank_0_12 - 234 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 244) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 234) ? 30 + tile_rank_0_12 - 224 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 234) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 224) ? 40 + tile_rank_0_12 - 214 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 224) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 214) ? 40 + tile_rank_0_12 - 204 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 214) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 204) ? 50 + tile_rank_0_12 - 194 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 204) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 194) ? 50 + tile_rank_0_12 - 184 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 194) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 184) ? 60 + tile_rank_0_12 - 174 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 184) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 174) ? 60 + tile_rank_0_12 - 164 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 174) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 164) ? 70 + tile_rank_0_12 - 154 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 164) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 154) ? 70 + tile_rank_0_12 - 144 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 154) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 144) ? 80 + tile_rank_0_12 - 134 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 144) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 134) ? 80 + tile_rank_0_12 - 124 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 134) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 124) ? 90 + tile_rank_0_12 - 114 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 124) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 114) ? 144 + tile_rank_0_12 - 108 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 114) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 108) ? 90 + tile_rank_0_12 - 98 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 108) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 98) ? 100 + tile_rank_0_12 - 88 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 98) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 88) ? 100 + tile_rank_0_12 - 78 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 88) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 78) ? 110 + tile_rank_0_12 - 68 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 78) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 68) ? 110 + tile_rank_0_12 - 58 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 68) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 58) ? 120 + tile_rank_0_12 - 48 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 58) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 48) ? 120 + tile_rank_0_12 - 38 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 48) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 38) ? 130 + tile_rank_0_12 - 28 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 38) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 28) ? 130 + tile_rank_0_12 - 18 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 28) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 18) ? 140 + tile_rank_0_12 - 14 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 18) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 14) ? 140 + tile_rank_0_12 - 10 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 14) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 10) ? 150 + tile_rank_0_12 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 10) ? 0 : split_idx_2_12);
                                } else if (batch_size == 192) {
                                    schedule_batch_idx_13 = tile_rank_0_12 - 292;
                                    split_idx_2_12 = 0;
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 292) ? tile_rank_0_12 - 280 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 292) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 280) ? 12 + tile_rank_0_12 - 268 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 280) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 268) ? 112 + tile_rank_0_12 - 260 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 268) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 260) ? 12 + tile_rank_0_12 - 248 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 260) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 248) ? 24 + tile_rank_0_12 - 236 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 248) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 236) ? 120 + tile_rank_0_12 - 224 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 236) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 224) ? 24 + tile_rank_0_12 - 212 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 224) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 212) ? 36 + tile_rank_0_12 - 200 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 212) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 200) ? 132 + tile_rank_0_12 - 188 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 200) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 188) ? 36 + tile_rank_0_12 - 176 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 188) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 176) ? 48 + tile_rank_0_12 - 164 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 176) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 164) ? 48 + tile_rank_0_12 - 152 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 164) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 152) ? 60 + tile_rank_0_12 - 140 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 152) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 140) ? 144 + tile_rank_0_12 - 128 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 140) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 128) ? 60 + tile_rank_0_12 - 116 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 128) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 116) ? 72 + tile_rank_0_12 - 104 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 116) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 104) ? 156 + tile_rank_0_12 - 92 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 104) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 92) ? 72 + tile_rank_0_12 - 80 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 92) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 80) ? 84 + tile_rank_0_12 - 68 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 80) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 68) ? 84 + tile_rank_0_12 - 56 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 68) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 56) ? 96 + tile_rank_0_12 - 44 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 56) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 44) ? 96 + tile_rank_0_12 - 32 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 44) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 32) ? 108 + tile_rank_0_12 - 28 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 32) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 28) ? 168 + tile_rank_0_12 - 16 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 28) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 16) ? 108 + tile_rank_0_12 - 12 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 16) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 12) ? 180 + tile_rank_0_12 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 12) ? 0 : split_idx_2_12);
                                } else {
                                    schedule_batch_idx_13 = 80 + tile_rank_0_12 - 300;
                                    split_idx_2_12 = 0;
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 300) ? 84 + tile_rank_0_12 - 286 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 300) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 286) ? 98 + tile_rank_0_12 - 272 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 286) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 272) ? 112 + tile_rank_0_12 - 258 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 272) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 258) ? tile_rank_0_12 - 244 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 258) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 244) ? tile_rank_0_12 - 230 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 244) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 230) ? 14 + tile_rank_0_12 - 216 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 230) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 216) ? 126 + tile_rank_0_12 - 202 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 216) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 202) ? 14 + tile_rank_0_12 - 188 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 202) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 188) ? 28 + tile_rank_0_12 - 174 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 188) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 174) ? 140 + tile_rank_0_12 - 160 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 174) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 160) ? 28 + tile_rank_0_12 - 146 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 160) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 146) ? 42 + tile_rank_0_12 - 132 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 146) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 132) ? 154 + tile_rank_0_12 - 118 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 132) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 118) ? 42 + tile_rank_0_12 - 104 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 118) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 104) ? 56 + tile_rank_0_12 - 90 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 104) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 90) ? 56 + tile_rank_0_12 - 76 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 90) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 76) ? 70 + tile_rank_0_12 - 66 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 76) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 66) ? 168 + tile_rank_0_12 - 52 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 66) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 52) ? 70 + tile_rank_0_12 - 42 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 52) ? 1 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 42) ? 182 + tile_rank_0_12 - 28 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 42) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 28) ? 196 + tile_rank_0_12 - 14 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 28) ? 0 : split_idx_2_12);
                                    schedule_batch_idx_13 = ((tile_rank_0_12 < 14) ? 210 + tile_rank_0_12 : schedule_batch_idx_13);
                                    split_idx_2_12 = ((tile_rank_0_12 < 14) ? 0 : split_idx_2_12);
                                }
                            }
                            int part_count_3_12 = 1;
                            if (schedule_batch_idx_13 < split_requests_12) {
                                part_count_3_12 = 2;
                            }
                            schedule_batch_5 = (unsigned int)schedule_batch_idx_13;
                            next_part_5 = (unsigned int)(part_count_3_12 << 16 | split_idx_2_12);
                        }
                    }
                    unsigned int next_batch_5 = schedule_batch_5;
                    {
                        if (valid_6 != 0) {
                            {
                                {
                                    next_batch_5 = request_order[schedule_batch_5];
                                }
                            }
                        }
                    }
                    {
                        int packed_part_5 = (int)next_part_5;
                        split_idx_1_5 = packed_part_5 % 65536;
                        part_count_2_5 = packed_part_5 / 65536;
                    }
                    unsigned int valid_t = valid_6;
                    batch_idx_t = (int)next_batch_5;
                    q_row_idx_t = (int)next_q_row_5;
                    kv_head_idx_t = kv_head_idx_0_5;
                    split_idx_t = split_idx_1_5;
                    part_count_t = part_count_2_5;
                    bundle_idx_t = next_bundle_idx_6;
                    bundle_item_idx_t = next_bundle_item_idx_6;
                    if (valid_t == 0) {
                        break;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
