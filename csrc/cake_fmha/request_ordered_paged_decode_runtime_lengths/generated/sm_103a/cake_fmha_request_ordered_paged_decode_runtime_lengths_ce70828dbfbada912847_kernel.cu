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
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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
#define SMEM_SPLIT_WEIGHTS_OFF 145296
#define SMEM_SPLIT_WEIGHTS_STAGE_BYTES 2048
#define SMEM_SPLIT_WEIGHTS_STRIDE 2048
#define SMEM_SMEM_PAGE_OFFSETS_OFF 141440
#define SMEM_SMEM_PAGE_OFFSETS_STAGE_BYTES 768
#define SMEM_SMEM_PAGE_OFFSETS_STRIDE 768
#define SMEM_RUNTIME_UNITS_OFF 142208
#define SMEM_RUNTIME_UNITS_STAGE_BYTES 1024
#define SMEM_RUNTIME_UNITS_STRIDE 1024
#define SMEM_RUNTIME_PARTS_OFF 143232
#define SMEM_RUNTIME_PARTS_STAGE_BYTES 1024
#define SMEM_RUNTIME_PARTS_STRIDE 1024
#define SMEM_RUNTIME_PREFIX_OFF 144256
#define SMEM_RUNTIME_PREFIX_STAGE_BYTES 1028
#define SMEM_RUNTIME_PREFIX_STRIDE 1028
#define SMEM_RUNTIME_PLAN_OFF 145284
#define SMEM_RUNTIME_PLAN_STAGE_BYTES 8
#define SMEM_RUNTIME_PLAN_STRIDE 8
#define SMEM_TOTAL 147456
#define THREADS 512
#define BLOCK_N 128
#define HEAD_DIM 256
#define HEAD_DIM_HALF 128
#define TILE_Q 8
#define Q_GROUPS_PER_KV 1
#define PAGE_SIZE 64
#define NUM_RAW_KV_STAGES 4
#define NUM_TRANSFORMED_KV_STAGES 2
#define Q_LEN 1
#define UNIFORM_KV_LEN 0
#define USE_REQUEST_ORDER 0
#define USE_SCALE_POINTERS 1
#define NUM_SPLIT 64
#define USE_SEGMENTED_CLC 1
#define USE_HIGH_BATCH_TWO_WAVE 0
#define USE_TWO_CTA_REDUCER 0
#define USE_MMA_LOOP_PEEL 1
#define USE_PAGE_OFFSET_CPASYNC 0
#define USE_LEGACY_PAGE_VEC4 0
#define WRITE_LSE 1

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
kernel_cake_fmha_request_ordered_paged_decode_runtime_lengths_ce70828dbfbada912847(CakeTensorMap const* Qt, CakeTensorMap const* K, CakeTensorMap const* V, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_LSE, unsigned int* __restrict__ split_completion, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, int* __restrict__ request_order, int max_pages_per_seq, int page_table_v_offset, float softmax_scale_log2, float output_scale, float* __restrict__ bmm1_scale_ptr, float* __restrict__ bmm2_scale_ptr, int bmm1_is_log2, int num_q_heads, int num_kv_heads, int batch_size, unsigned int total_tiles)
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
    float* split_weights = reinterpret_cast<float*>(smem_raw + 145296);
    const int split_weights_addr = smem + 145296;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 141440);
    const int smem_page_offsets_addr = smem + 141440;
    int* runtime_units = reinterpret_cast<int*>(smem_raw + 142208);
    const int runtime_units_addr = smem + 142208;
    int* runtime_parts = reinterpret_cast<int*>(smem_raw + 143232);
    const int runtime_parts_addr = smem + 143232;
    int* runtime_prefix = reinterpret_cast<int*>(smem_raw + 144256);
    const int runtime_prefix_addr = smem + 144256;
    int* runtime_plan = reinterpret_cast<int*>(smem_raw + 145284);
    const int runtime_plan_addr = smem + 145284;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Qt)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp == 10) {
        if (elect_sync()) {
            unsigned int runtime_sum = 0;
            int runtime_min_len = 2147483647;
            int runtime_max_len = 0;
            #pragma unroll 1
            for (int runtime_slot = 0; runtime_slot < batch_size; runtime_slot++) {
                int runtime_request = runtime_slot;
                int runtime_length = seq_lens_kv[runtime_request];
                int _min_0 = ((runtime_min_len) < (runtime_length) ? (runtime_min_len) : (runtime_length));
                runtime_min_len = _min_0;
                int _max_0 = ((runtime_max_len) > (runtime_length) ? (runtime_max_len) : (runtime_length));
                runtime_max_len = _max_0;
                int _max_1 = ((runtime_length - Q_LEN + 1) > (0) ? (runtime_length - Q_LEN + 1) : (0));
                int runtime_visible = _max_1;
                int runtime_unit_count = runtime_visible / (2 * BLOCK_N);
                if (runtime_visible % (2 * BLOCK_N) != 0) {
                    runtime_unit_count += 1;
                }
                int _max_2 = ((runtime_unit_count) > (1) ? (runtime_unit_count) : (1));
                runtime_unit_count = _max_2;
                runtime_units[runtime_slot] = runtime_unit_count;
                runtime_sum += (unsigned int)runtime_unit_count;
            }
            int _max_3 = ((batch_size) > (304) ? (batch_size) : (304));
            unsigned int runtime_target = (unsigned int)_max_3;
            unsigned int runtime_pairs = runtime_sum / runtime_target;
            if (runtime_sum % runtime_target != 0) {
                runtime_pairs += 1;
            }
            unsigned int _max_4 = ((runtime_pairs) > (4) ? (runtime_pairs) : (4));
            runtime_pairs = _max_4;
            int runtime_uniform = 0;
            if (batch_size > 0) {
                if (runtime_min_len == runtime_max_len) {
                    int runtime_uniform_units = runtime_units[0];
                    int _min_1 = ((gridDim.z / batch_size) < (runtime_uniform_units) ? (gridDim.z / batch_size) : (runtime_uniform_units));
                    runtime_uniform = _min_1;
                }
            }
            runtime_prefix[0] = 0;
            int runtime_total = 0;
            #pragma unroll 1
            for (int runtime_slot_p = 0; runtime_slot_p < batch_size; runtime_slot_p++) {
                int runtime_unit_value = runtime_units[runtime_slot_p];
                unsigned int runtime_u = (unsigned int)runtime_unit_value;
                int runtime_p = (int)(runtime_u / runtime_pairs);
                if (runtime_u % runtime_pairs != 0) {
                    runtime_p += 1;
                }
                int _max_5 = ((runtime_p) > (1) ? (runtime_p) : (1));
                int _min_2 = ((64) < (_max_5) ? (64) : (_max_5));
                runtime_p = _min_2;
                if (runtime_uniform != 0) {
                    runtime_p = runtime_uniform;
                }
                runtime_parts[runtime_slot_p] = runtime_p;
                runtime_total += runtime_p;
                runtime_prefix[runtime_slot_p + 1] = runtime_total;
            }
            runtime_plan[0] = runtime_total;
            runtime_plan[1] = runtime_uniform;
        }
    }
    __syncthreads();

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
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

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
            const int wg_tid = warp_in_wg * 32 + threadIdx.x % 32;
            int col_pair = wg_tid % 4;
            int col_pair_base = col_pair * 2;
            unsigned int work_stage_s = 0;
            int sm_stage = 0;
            int sm_phase = 0;
            int corr_prod_stage = 0;
            int corr_prod_phase = 1;
            float bmm1_scale_log2_s = softmax_scale_log2;
            float bmm1_scale_log2_p = softmax_scale_log2;
            int runtime_virtual = blockIdx.z;
            int runtime_tile_total = runtime_plan[0];
            int runtime_uniform_count = runtime_plan[1];
            int runtime_batch = 0;
            int runtime_split = 0;
            int runtime_count = 1;
            if (runtime_virtual < runtime_tile_total) {
                if (runtime_uniform_count != 0) {
                    runtime_count = runtime_uniform_count;
                    runtime_batch = runtime_virtual / runtime_count;
                    runtime_split = runtime_virtual % runtime_count;
                } else {
                    int runtime_lo = 0;
                    int runtime_hi = batch_size;
                    #pragma unroll 1
                    for (int runtime_search = 0; runtime_search < 9; runtime_search++) {
                        if (runtime_lo < runtime_hi) {
                            int runtime_mid = (runtime_lo + runtime_hi) / 2;
                            int runtime_prefix_end = runtime_prefix[runtime_mid + 1];
                            if (runtime_prefix_end <= runtime_virtual) {
                                runtime_lo = runtime_mid + 1;
                            } else {
                                runtime_hi = runtime_mid;
                            }
                        }
                    }
                    runtime_batch = runtime_lo;
                    int runtime_prefix_start = runtime_prefix[runtime_batch];
                    runtime_split = runtime_virtual - runtime_prefix_start;
                    int runtime_read_count = runtime_parts[runtime_batch];
                    runtime_count = runtime_read_count;
                }
            }
            int batch_idx_s = runtime_batch;
            int q_row_idx_s = blockIdx.x;
            int kv_head_idx_s = blockIdx.y;
            int split_idx_s = runtime_split;
            int part_count_s = runtime_count;
            int bundle_idx_s = blockIdx.z;
            int bundle_item_idx_s = 0;
            unsigned int runtime_work = 0;
            int runtime_work_total = runtime_plan[0];
            int runtime_remaining = runtime_work_total - blockIdx.z;
            if (runtime_remaining > 0) {
                runtime_work = (unsigned int)((runtime_remaining - 1) / gridDim.z + 1);
            }
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < runtime_work; _tile_iter_s++) {
                int visible_keys = UNIFORM_KV_LEN - Q_LEN + q_row_idx_s + 1;
                {
                    visible_keys = seq_lens_kv[batch_idx_s] - Q_LEN + q_row_idx_s + 1;
                }
                if (visible_keys < 0) {
                    visible_keys = 0;
                }
                int seqlen_kv_s = visible_keys;
                int num_n_blocks = 0;
                num_n_blocks = seqlen_kv_s / BLOCK_N;
                if (seqlen_kv_s % BLOCK_N != 0) {
                    num_n_blocks += 1;
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
                    int ldtm_row_base = warp_in_wg * 32 + threadIdx.x % 32 / 4;
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
                    float _max_6 = max_noftz(pair_max[0], sv[0]);
                    pair_max[0] = _max_6;
                    float _max_7 = max_noftz(pair_max[0], sv[2]);
                    pair_max[0] = _max_7;
                    float _max_8 = max_noftz(pair_max[1], sv[1]);
                    pair_max[1] = _max_8;
                    float _max_9 = max_noftz(pair_max[1], sv[3]);
                    pair_max[1] = _max_9;
                    float _max_10 = max_noftz(pair_max[0], sv[4]);
                    pair_max[0] = _max_10;
                    float _max_11 = max_noftz(pair_max[0], sv[6]);
                    pair_max[0] = _max_11;
                    float _max_12 = max_noftz(pair_max[1], sv[5]);
                    pair_max[1] = _max_12;
                    float _max_13 = max_noftz(pair_max[1], sv[7]);
                    pair_max[1] = _max_13;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 2; c_1++) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 16);
                        float _max_14 = max_noftz(pair_max[c_1], _shfl_xor_0);
                        pair_max[c_1] = _max_14;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 8);
                        float _max_15 = max_noftz(pair_max[c_1], _shfl_xor_1);
                        pair_max[c_1] = _max_15;
                    }
                    float new_max_pair[2];
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 2; c_2++) {
                        float _max_16 = max_noftz(row_max_pair[c_2], pair_max[c_2]);
                        new_max_pair[c_2] = _max_16;
                    }
                    if (threadIdx.x % 32 < 8) {
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
                        float _shfl_0;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(acc_scale_pair[0]), "r"(cp));
                        acc_scale[cp * 2] = _shfl_0;
                        float _shfl_1;
                        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_1) : "f"(acc_scale_pair[1]), "r"(cp));
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
                    {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
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
                    int runtime_next_item = bundle_item_idx_s + 1;
                    int runtime_next_id = bundle_idx_s + runtime_next_item * gridDim.z;
                    unsigned int runtime_valid = 0;
                    int runtime_next_total = runtime_plan[0];
                    if (runtime_next_id < runtime_next_total) {
                        runtime_valid = 1;
                    }
                    int runtime_virtual_0 = runtime_next_id;
                    int runtime_tile_total_1 = runtime_plan[0];
                    int runtime_uniform_count_2 = runtime_plan[1];
                    int runtime_batch_3 = 0;
                    int runtime_split_4 = 0;
                    int runtime_count_5 = 1;
                    if (runtime_virtual_0 < runtime_tile_total_1) {
                        if (runtime_uniform_count_2 != 0) {
                            runtime_count_5 = runtime_uniform_count_2;
                            runtime_batch_3 = runtime_virtual_0 / runtime_count_5;
                            runtime_split_4 = runtime_virtual_0 % runtime_count_5;
                        } else {
                            int runtime_lo_1 = 0;
                            int runtime_hi_1 = batch_size;
                            #pragma unroll 1
                            for (int runtime_search_1 = 0; runtime_search_1 < 9; runtime_search_1++) {
                                if (runtime_lo_1 < runtime_hi_1) {
                                    int runtime_mid_1 = (runtime_lo_1 + runtime_hi_1) / 2;
                                    int runtime_prefix_end_1 = runtime_prefix[runtime_mid_1 + 1];
                                    if (runtime_prefix_end_1 <= runtime_virtual_0) {
                                        runtime_lo_1 = runtime_mid_1 + 1;
                                    } else {
                                        runtime_hi_1 = runtime_mid_1;
                                    }
                                }
                            }
                            runtime_batch_3 = runtime_lo_1;
                            int runtime_prefix_start_1 = runtime_prefix[runtime_batch_3];
                            runtime_split_4 = runtime_virtual_0 - runtime_prefix_start_1;
                            int runtime_read_count_1 = runtime_parts[runtime_batch_3];
                            runtime_count_5 = runtime_read_count_1;
                        }
                    }
                    unsigned int valid_s = runtime_valid;
                    batch_idx_s = runtime_batch_3;
                    q_row_idx_s = q_row_idx_s;
                    kv_head_idx_s = kv_head_idx_s;
                    split_idx_s = runtime_split_4;
                    part_count_s = runtime_count_5;
                    bundle_idx_s = bundle_idx_s;
                    bundle_item_idx_s = runtime_next_item;
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
            const int corr_tid = warp_in_wg_c * 32 + threadIdx.x % 32;
            const int col_pair_c = corr_tid % 4;
            const int col_pair_base_c = col_pair_c * 2;
            unsigned int work_stage_c = 0;
            int corr_cons_stage = 0;
            int corr_cons_phase = 0;
            int p_stage_c = 0;
            int d_idx = warp % 4 * 32 + (unsigned int)(threadIdx.x % 32);
            int group_ratio_rt = num_q_heads / (num_kv_heads * Q_GROUPS_PER_KV);
            float bmm1_scale_log2_c = softmax_scale_log2;
            float bmm2_scale_c = output_scale;
            int runtime_virtual_1 = blockIdx.z;
            int runtime_tile_total_2 = runtime_plan[0];
            int runtime_uniform_count_1 = runtime_plan[1];
            int runtime_batch_1 = 0;
            int runtime_split_1 = 0;
            int runtime_count_1 = 1;
            if (runtime_virtual_1 < runtime_tile_total_2) {
                if (runtime_uniform_count_1 != 0) {
                    runtime_count_1 = runtime_uniform_count_1;
                    runtime_batch_1 = runtime_virtual_1 / runtime_count_1;
                    runtime_split_1 = runtime_virtual_1 % runtime_count_1;
                } else {
                    int runtime_lo_2 = 0;
                    int runtime_hi_2 = batch_size;
                    #pragma unroll 1
                    for (int runtime_search_2 = 0; runtime_search_2 < 9; runtime_search_2++) {
                        if (runtime_lo_2 < runtime_hi_2) {
                            int runtime_mid_2 = (runtime_lo_2 + runtime_hi_2) / 2;
                            int runtime_prefix_end_2 = runtime_prefix[runtime_mid_2 + 1];
                            if (runtime_prefix_end_2 <= runtime_virtual_1) {
                                runtime_lo_2 = runtime_mid_2 + 1;
                            } else {
                                runtime_hi_2 = runtime_mid_2;
                            }
                        }
                    }
                    runtime_batch_1 = runtime_lo_2;
                    int runtime_prefix_start_2 = runtime_prefix[runtime_batch_1];
                    runtime_split_1 = runtime_virtual_1 - runtime_prefix_start_2;
                    int runtime_read_count_2 = runtime_parts[runtime_batch_1];
                    runtime_count_1 = runtime_read_count_2;
                }
            }
            int batch_idx_c = runtime_batch_1;
            int q_row_idx_c = blockIdx.x;
            int kv_head_idx_c = blockIdx.y;
            int split_idx_c = runtime_split_1;
            int part_count_c = runtime_count_1;
            int bundle_idx_c = blockIdx.z;
            int bundle_item_idx_c = 0;
            unsigned int runtime_work_1 = 0;
            int runtime_work_total_1 = runtime_plan[0];
            int runtime_remaining_1 = runtime_work_total_1 - blockIdx.z;
            if (runtime_remaining_1 > 0) {
                runtime_work_1 = (unsigned int)((runtime_remaining_1 - 1) / gridDim.z + 1);
            }
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < runtime_work_1; _tile_iter_c++) {
                int visible_keys_1 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_c + 1;
                {
                    visible_keys_1 = seq_lens_kv[batch_idx_c] - Q_LEN + q_row_idx_c + 1;
                }
                if (visible_keys_1 < 0) {
                    visible_keys_1 = 0;
                }
                int seqlen_kv_c = visible_keys_1;
                int num_n_blocks_1 = 0;
                num_n_blocks_1 = seqlen_kv_c / BLOCK_N;
                if (seqlen_kv_c % BLOCK_N != 0) {
                    num_n_blocks_1 += 1;
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
                if (num_pairs_1 * 2 > 0) {
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
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
                if (threadIdx.x % 32 < 4) {
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
                    if (threadIdx.x % 32 >= 4) {
                        stats_head_c = col_pair_base_c + 1;
                        stats_sum_c = total_sum_pair[1];
                        stats_max_c = _tmem_load_3[3];
                    }
                    if (stats_head_c < group_ratio_rt) {
                        int stats_q_head_c = kv_head_idx_c * group_ratio_rt + stats_head_c;
                        long long stats_output_row_c = ((long long)batch_idx_c * (long long)Q_LEN + (long long)q_row_idx_c) * (long long)num_q_heads + (long long)stats_q_head_c;
                        {
                            if (part_count_c == 1) {
                                {
                                    float _log2_0;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(stats_sum_c));
                                    float lse_value_final_seg = _log2_0 + stats_max_c * bmm1_scale_log2_c;
                                    if (num_pairs_1 * 2 == 0) {
                                        lse_value_final_seg = -CAKE_INF;
                                    }
                                    *(reinterpret_cast<float*>(LSE + stats_output_row_c) + (0)) = lse_value_final_seg;
                                }
                            } else {
                                float _log2_1;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(stats_sum_c));
                                float lse_value_partial_seg = _log2_1 + stats_max_c * bmm1_scale_log2_c;
                                long long partial_lse_idx = (long long)stats_output_row_c * 64 + (long long)split_idx_c;
                                *(reinterpret_cast<float*>(partial_LSE + partial_lse_idx) + (0)) = lse_value_partial_seg;
                            }
                        }
                    }
                }
                float total_sum[8];
                float total_max[8];
                #pragma unroll
                for (int cp_1 = 0; cp_1 < 4; cp_1++) {
                    float _shfl_2;
                    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_2) : "f"(total_sum_pair[0]), "r"(cp_1));
                    total_sum[cp_1 * 2] = _shfl_2;
                    float _shfl_3;
                    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_3) : "f"(total_sum_pair[1]), "r"(cp_1));
                    total_sum[cp_1 * 2 + 1] = _shfl_3;
                    float _shfl_4;
                    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_4) : "f"(_tmem_load_3[2]), "r"(cp_1));
                    total_max[cp_1 * 2] = _shfl_4;
                    float _shfl_5;
                    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_5) : "f"(_tmem_load_3[3]), "r"(cp_1));
                    total_max[cp_1 * 2 + 1] = _shfl_5;
                }
                float split_o_hi_epi[8];
                float split_o_lo_epi[8];
                split_o_hi_epi[0] = 0.0f;
                split_o_hi_epi[1] = 0.0f;
                split_o_hi_epi[2] = 0.0f;
                split_o_hi_epi[3] = 0.0f;
                split_o_hi_epi[4] = 0.0f;
                split_o_hi_epi[5] = 0.0f;
                split_o_hi_epi[6] = 0.0f;
                split_o_hi_epi[7] = 0.0f;
                split_o_lo_epi[0] = 0.0f;
                split_o_lo_epi[1] = 0.0f;
                split_o_lo_epi[2] = 0.0f;
                split_o_lo_epi[3] = 0.0f;
                split_o_lo_epi[4] = 0.0f;
                split_o_lo_epi[5] = 0.0f;
                split_o_lo_epi[6] = 0.0f;
                split_o_lo_epi[7] = 0.0f;
                if (num_pairs_1 * 2 > 0) {
                    tmem_ld_x8(&split_o_hi_epi[0], taddr + 32 + (unsigned int)corr_row);
                    tmem_ld_x8(&split_o_lo_epi[0], taddr + 40 + (unsigned int)corr_row);
                }
                #pragma unroll
                for (int h_2 = 0; h_2 < 8; h_2++) {
                    float final_o_hi = 0.0f;
                    float final_o_lo = 0.0f;
                    if (num_pairs_1 * 2 > 0) {
                        float _rcp_0 = approx_rcp(total_sum[h_2]);
                        float inv_total = _rcp_0;
                        final_o_hi = split_o_hi_epi[h_2] * inv_total * bmm2_scale_c;
                        final_o_lo = split_o_lo_epi[h_2] * inv_total * bmm2_scale_c;
                    }
                    if (group_ratio_rt > h_2) {
                        int q_head = kv_head_idx_c * group_ratio_rt + h_2;
                        long long output_row = ((long long)batch_idx_c * (long long)Q_LEN + (long long)q_row_idx_c) * (long long)num_q_heads + (long long)q_head;
                        {
                            if (part_count_c == 1) {
                                long long direct_out_idx_hi_seg = (long long)output_row * (long long)HEAD_DIM + (long long)d_idx;
                                long long direct_out_idx_lo_seg = (long long)direct_out_idx_hi_seg + (long long)HEAD_DIM_HALF;
                                *(reinterpret_cast<__nv_bfloat16*>(O + direct_out_idx_hi_seg) + (0)) = __float2bfloat16_rn(final_o_hi);
                                *(reinterpret_cast<__nv_bfloat16*>(O + direct_out_idx_lo_seg) + (0)) = __float2bfloat16_rn(final_o_lo);
                            } else {
                                long long partial_row = (long long)output_row * 64 + (long long)split_idx_c;
                                long long partial_idx_hi = (long long)partial_row * (long long)HEAD_DIM + (long long)d_idx;
                                long long partial_idx_lo = (long long)partial_idx_hi + (long long)HEAD_DIM_HALF;
                                *(reinterpret_cast<__nv_bfloat16*>(partial_O + partial_idx_hi) + (0)) = __float2bfloat16_rn(final_o_hi);
                                *(reinterpret_cast<__nv_bfloat16*>(partial_O + partial_idx_lo) + (0)) = __float2bfloat16_rn(final_o_lo);
                            }
                        }
                    }
                }
                if (num_pairs_1 * 2 > 0) {
                    mbarrier_arrive(o_empty_addr);
                }
                if (USE_SEGMENTED_CLC != 0 && part_count_c > 1) {
                    int base_tile_idx_seg = (batch_idx_c * Q_LEN + q_row_idx_c) * (num_kv_heads * Q_GROUPS_PER_KV) + kv_head_idx_c;
                    __threadfence();
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    int assisted_reduce_seg = 0;
                    int assisted_total_seg = runtime_plan[0];
                    if (assisted_total_seg <= gridDim.z) {
                        if (gridDim.x * gridDim.y * gridDim.z <= 152) {
                            assisted_reduce_seg = 1;
                        }
                    }
                    if (d_idx == 0) {
                        uint32_t _atomic_inc_old_0;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_inc_old_0) : "l"(&split_completion[base_tile_idx_seg]), "r"(static_cast<uint32_t>(part_count_c - 1)) : "memory");
                        unsigned int old_count_seg = _atomic_inc_old_0;
                        split_reduce_flag[0] = 0;
                        if (assisted_reduce_seg != 0) {
                            if (part_count_c <= (int)old_count_seg + 2) {
                                split_reduce_flag[0] = part_count_c - (int)old_count_seg;
                                if ((int)old_count_seg + 2 == part_count_c) {
                                    {
                                        const uint32_t* _awe_p_0 = &split_completion[base_tile_idx_seg];
                                        while (true) {
                                            uint32_t _awe_v_0;
                                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_awe_v_0) : "l"(_awe_p_0) : "memory");
                                            if (_awe_v_0 == static_cast<uint32_t>(0)) break;
                                        }
                                    }
                                }
                            }
                        } else {
                            split_reduce_flag[0] = (((int)old_count_seg + 1 == part_count_c) ? 1 : 0);
                        }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (split_reduce_flag[0] != 0) {
                        __threadfence();
                        int reduce_head_seg = d_idx / 16;
                        int reduce_lane_seg = d_idx % 16;
                        int reduce_head_valid_seg = 0;
                        if (reduce_head_seg < group_ratio_rt) {
                            if (reduce_head_seg < TILE_Q) {
                                reduce_head_valid_seg = 1;
                            }
                        }
                        if (assisted_reduce_seg != 0) {
                            int assisted_head_begin_seg = (2 - split_reduce_flag[0]) * 4;
                            if (reduce_head_seg < assisted_head_begin_seg) {
                                reduce_head_valid_seg = 0;
                            }
                            if (reduce_head_seg >= assisted_head_begin_seg + 4) {
                                reduce_head_valid_seg = 0;
                            }
                        }
                        int reduce_q_head_seg = kv_head_idx_c * group_ratio_rt + reduce_head_seg;
                        long long reduce_stat_base_seg = (((long long)batch_idx_c * (long long)Q_LEN + (long long)q_row_idx_c) * (long long)num_q_heads + (long long)reduce_q_head_seg) * 64;
                        int split0_seg = reduce_lane_seg;
                        int split1_seg = reduce_lane_seg + 8;
                        float lse0_seg = -CAKE_INF;
                        float lse1_seg = -CAKE_INF;
                        float cached_lse_seg[4];
                        cached_lse_seg[0] = -CAKE_INF;
                        cached_lse_seg[1] = -CAKE_INF;
                        cached_lse_seg[2] = -CAKE_INF;
                        cached_lse_seg[3] = -CAKE_INF;
                        #pragma unroll
                        for (int cache_chunk_seg = 0; cache_chunk_seg < 4; cache_chunk_seg++) {
                            int cache_part_seg = cache_chunk_seg * 16 + reduce_lane_seg;
                            if (reduce_head_valid_seg != 0) {
                                if (cache_part_seg < part_count_c) {
                                    cached_lse_seg[cache_chunk_seg] = partial_LSE[reduce_stat_base_seg + (long long)cache_part_seg];
                                }
                            }
                        }
                        float lane_max_seg = -CAKE_INF;
                        #pragma unroll
                        for (int stats_chunk_seg = 0; stats_chunk_seg < 4; stats_chunk_seg++) {
                            float _max_17 = max_noftz(lane_max_seg, cached_lse_seg[stats_chunk_seg]);
                            lane_max_seg = _max_17;
                        }
                        int subgroup_lane_base_seg = threadIdx.x % 32 / 16 * 16;
                        float merged_max_seg = lane_max_seg;
                        #pragma unroll
                        for (int max_stage_seg = 0; max_stage_seg < 4; max_stage_seg++) {
                            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, merged_max_seg, 1 << max_stage_seg);
                            float peer_max_seg = _shfl_xor_5;
                            float _max_18 = max_noftz(merged_max_seg, peer_max_seg);
                            merged_max_seg = _max_18;
                        }
                        float weight0_seg = 0.0f;
                        float weight1_seg = 0.0f;
                        float lane_weight_sum_seg = 0.0f;
                        #pragma unroll
                        for (int weight_chunk_seg = 0; weight_chunk_seg < 4; weight_chunk_seg++) {
                            int weight_part_seg = weight_chunk_seg * 16 + reduce_lane_seg;
                            if (reduce_head_valid_seg != 0) {
                                if (weight_part_seg < part_count_c) {
                                    float weight_lse_seg = cached_lse_seg[weight_chunk_seg];
                                    float weight_value_seg = 0.0f;
                                    if (weight_lse_seg != -CAKE_INF) {
                                        float _exp2_2 = approx_exp2(weight_lse_seg - merged_max_seg);
                                        weight_value_seg = _exp2_2;
                                    }
                                    split_weights[weight_part_seg * TILE_Q + reduce_head_seg] = weight_value_seg;
                                    lane_weight_sum_seg = lane_weight_sum_seg + weight_value_seg;
                                }
                            }
                        }
                        float weight_sum_seg = lane_weight_sum_seg;
                        #pragma unroll
                        for (int sum_stage_seg = 0; sum_stage_seg < 4; sum_stage_seg++) {
                            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, weight_sum_seg, 1 << sum_stage_seg);
                            float peer_sum_seg = _shfl_xor_6;
                            weight_sum_seg = weight_sum_seg + peer_sum_seg;
                        }
                        float _rcp_1 = approx_rcp(weight_sum_seg);
                        float inv_weight_sum_seg = ((weight_sum_seg > 0.0f) ? _rcp_1 : 0.0f);
                        if (reduce_head_valid_seg != 0) {
                            #pragma unroll 1
                            for (int norm_chunk_seg = 0; norm_chunk_seg < (part_count_c + 16 - 1) / 16; norm_chunk_seg++) {
                                int norm_part_seg = norm_chunk_seg * 16 + reduce_lane_seg;
                                if (norm_part_seg < part_count_c) {
                                    float norm_weight_seg = split_weights[norm_part_seg * TILE_Q + reduce_head_seg];
                                    split_weights[norm_part_seg * TILE_Q + reduce_head_seg] = norm_weight_seg * inv_weight_sum_seg;
                                }
                            }
                            if (reduce_lane_seg == 0) {
                                {
                                    float merged_lse_seg = -CAKE_INF;
                                    if (weight_sum_seg > 0.0f) {
                                        float _log2_4;
                                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_4) : "f"(weight_sum_seg));
                                        merged_lse_seg = merged_max_seg + _log2_4;
                                    }
                                    long long final_lse_idx_seg = ((long long)batch_idx_c * (long long)Q_LEN + (long long)q_row_idx_c) * (long long)num_q_heads + (long long)reduce_q_head_seg;
                                    if (split_reduce_flag[0] == 2) {
                                        *(reinterpret_cast<float*>(partial_LSE + reduce_stat_base_seg) + (0)) = merged_lse_seg;
                                    } else {
                                        *(reinterpret_cast<float*>(LSE + final_lse_idx_seg) + (0)) = merged_lse_seg;
                                    }
                                }
                            }
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        int wide_head_begin_seg = 0;
                        int wide_head_batches_seg = 2;
                        if (assisted_reduce_seg != 0) {
                            wide_head_begin_seg = (2 - split_reduce_flag[0]) * 4;
                            wide_head_batches_seg = 1;
                        }
                        #pragma unroll 1
                        for (int wide_head_batch_seg = 0; wide_head_batch_seg < wide_head_batches_seg; wide_head_batch_seg++) {
                            int wide_head_seg = wide_head_begin_seg + wide_head_batch_seg * 4 + d_idx / 32;
                            int wide_elem_seg = d_idx % 32 * 8;
                            int wide_global_head_seg = kv_head_idx_c * group_ratio_rt + wide_head_seg;
                            long long wide_final_row_seg = ((long long)batch_idx_c * (long long)Q_LEN + (long long)q_row_idx_c) * (long long)num_q_heads + (long long)wide_global_head_seg;
                            long long wide_partial_base_seg = wide_final_row_seg * 64 * (long long)HEAD_DIM + (long long)wide_elem_seg;
                            float _vec_load_0[8];
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(partial_O + wide_partial_base_seg + 0);
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
                                            : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                            : "r"(_vpairs_1[_pair]));
                                    }
                                }
                            }
                            float wide_weight0_seg = split_weights[wide_head_seg];
                            #pragma unroll
                            for (int wide_element_seg = 0; wide_element_seg < 8; wide_element_seg++) {
                                _vec_load_0[wide_element_seg] = _vec_load_0[wide_element_seg] * wide_weight0_seg;
                            }
                            #pragma unroll 1
                            for (int wide_part_base_seg = 1; wide_part_base_seg < part_count_c; wide_part_base_seg += 4) {
                                float wide_values_seg[32];
                                float wide_weights_seg[4];
                                #pragma unroll
                                for (int wide_set_seg = 0; wide_set_seg < 4; wide_set_seg++) {
                                    int _min_3 = ((wide_part_base_seg + wide_set_seg) < (part_count_c - 1) ? (wide_part_base_seg + wide_set_seg) : (part_count_c - 1));
                                    int wide_part_seg = _min_3;
                                    float _vec_load_1[8];
                                    {
                                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(partial_O + (wide_partial_base_seg + (long long)(wide_part_seg * HEAD_DIM)) + 0);
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
                                                    : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                                    : "r"(_vpairs_2[_pair]));
                                            }
                                        }
                                    }
                                    #pragma unroll
                                    for (int wide_load_element_seg = 0; wide_load_element_seg < 8; wide_load_element_seg++) {
                                        wide_values_seg[wide_set_seg * 8 + wide_load_element_seg] = _vec_load_1[wide_load_element_seg];
                                    }
                                    wide_weights_seg[wide_set_seg] = split_weights[wide_part_seg * TILE_Q + wide_head_seg];
                                }
                                #pragma unroll
                                for (int wide_consume_seg = 0; wide_consume_seg < 4; wide_consume_seg++) {
                                    if (part_count_c > wide_part_base_seg + wide_consume_seg) {
                                        #pragma unroll
                                        for (int wide_fma_element_seg = 0; wide_fma_element_seg < 8; wide_fma_element_seg++) {
                                            float _fma_1 = __fmaf_rn(wide_values_seg[wide_consume_seg * 8 + wide_fma_element_seg], wide_weights_seg[wide_consume_seg], _vec_load_0[wide_fma_element_seg]);
                                            _vec_load_0[wide_fma_element_seg] = _fma_1;
                                        }
                                    }
                                }
                            }
                            if (split_reduce_flag[0] == 2) {
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_vec_load_0[0 + 0], _vec_load_0[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_vec_load_0[0 + 2], _vec_load_0[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_vec_load_0[0 + 4], _vec_load_0[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_vec_load_0[0 + 6], _vec_load_0[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + wide_partial_base_seg))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            } else {
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_vec_load_0[0 + 0], _vec_load_0[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_vec_load_0[0 + 2], _vec_load_0[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_vec_load_0[0 + 4], _vec_load_0[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_vec_load_0[0 + 6], _vec_load_0[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (wide_final_row_seg * (long long)HEAD_DIM + (long long)wide_elem_seg)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                        if (assisted_reduce_seg != 0) {
                            if (split_reduce_flag[0] == 2) {
                                __threadfence();
                                asm volatile("barrier.sync 9, 128;" ::: "memory");
                                if (d_idx == 0) {
                                    uint32_t _atomic_inc_old_1;
                                    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                                        : "=r"(_atomic_inc_old_1) : "l"(&split_completion[base_tile_idx_seg]), "r"(static_cast<uint32_t>(1)) : "memory");
                                    unsigned int helper_signal_seg = _atomic_inc_old_1;
                                }
                            } else {
                                if (d_idx == 0) {
                                    {
                                        const uint32_t* _awe_p_3 = &split_completion[base_tile_idx_seg];
                                        while (true) {
                                            uint32_t _awe_v_3;
                                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_awe_v_3) : "l"(_awe_p_3) : "memory");
                                            if (_awe_v_3 == static_cast<uint32_t>(1)) break;
                                        }
                                    }
                                }
                                asm volatile("barrier.sync 9, 128;" ::: "memory");
                                __threadfence();
                                int copied_head_seg = d_idx / 32;
                                int copied_elem_seg = d_idx % 32 * 8;
                                int copied_global_head_seg = kv_head_idx_c * group_ratio_rt + copied_head_seg;
                                long long copied_row_seg = ((long long)batch_idx_c * (long long)Q_LEN + (long long)q_row_idx_c) * (long long)num_q_heads + (long long)copied_global_head_seg;
                                float _vec_load_2[8];
                                {
                                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(partial_O + (copied_row_seg * 64 * (long long)HEAD_DIM + (long long)copied_elem_seg) + 0);
                                    uint4 _vld_4[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_4[_blk] = _vptr_4[_blk];
                                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                                                : "r"(_vpairs_4[_pair]));
                                        }
                                    }
                                }
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_vec_load_2[0 + 0], _vec_load_2[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_vec_load_2[0 + 2], _vec_load_2[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_vec_load_2[0 + 4], _vec_load_2[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_vec_load_2[0 + 6], _vec_load_2[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (copied_row_seg * (long long)HEAD_DIM + (long long)copied_elem_seg)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                                {
                                    if (threadIdx.x % 32 == 0) {
                                        float copied_lse_seg = partial_LSE[copied_row_seg * 64];
                                        *(reinterpret_cast<float*>(LSE + copied_row_seg) + (0)) = copied_lse_seg;
                                    }
                                }
                                __threadfence();
                                asm volatile("barrier.sync 9, 128;" ::: "memory");
                                if (d_idx == 0) {
                                    uint32_t _atomic_inc_old_2;
                                    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                                        : "=r"(_atomic_inc_old_2) : "l"(&split_completion[base_tile_idx_seg]), "r"(static_cast<uint32_t>(1)) : "memory");
                                    unsigned int final_reset_seg = _atomic_inc_old_2;
                                }
                            }
                        }
                    }
                } else if (USE_SEGMENTED_CLC == 0 && NUM_SPLIT > 1) {
                    int base_tile_idx = (batch_idx_c * Q_LEN + q_row_idx_c) * (num_kv_heads * Q_GROUPS_PER_KV) + kv_head_idx_c;
                    __threadfence();
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (d_idx == 0) {
                        uint32_t _atomic_inc_old_3;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_inc_old_3) : "l"(&split_completion[base_tile_idx]), "r"(static_cast<uint32_t>(NUM_SPLIT - 1)) : "memory");
                        unsigned int old_count = _atomic_inc_old_3;
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
                        float _max_19 = max_noftz(lse0, lse1);
                        float lane_max = _max_19;
                        int subgroup_lane_base = threadIdx.x % 32 / 8 * 8;
                        float merged_max = -CAKE_INF;
                        #pragma unroll
                        for (int source_lane = 0; source_lane < 8; source_lane++) {
                            float _shfl_6;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_6) : "f"(lane_max), "r"(subgroup_lane_base + source_lane));
                            float source_max = _shfl_6;
                            float _max_20 = max_noftz(merged_max, source_max);
                            merged_max = _max_20;
                        }
                        float weight0 = 0.0f;
                        float weight1 = 0.0f;
                        if (lse0 != -CAKE_INF) {
                            float _exp2_3 = approx_exp2(lse0 - merged_max);
                            weight0 = _exp2_3;
                        }
                        if (lse1 != -CAKE_INF) {
                            float _exp2_4 = approx_exp2(lse1 - merged_max);
                            weight1 = _exp2_4;
                        }
                        float lane_weight_sum = weight0 + weight1;
                        float weight_sum = 0.0f;
                        #pragma unroll
                        for (int source_lane_1 = 0; source_lane_1 < 8; source_lane_1++) {
                            float _shfl_7;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_7) : "f"(lane_weight_sum), "r"(subgroup_lane_base + source_lane_1));
                            weight_sum = weight_sum + _shfl_7;
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
                                {
                                    float merged_lse = -CAKE_INF;
                                    if (weight_sum > 0.0f) {
                                        float _log2_5;
                                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_5) : "f"(weight_sum));
                                        merged_lse = merged_max + _log2_5;
                                    }
                                    int final_lse_idx = (batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + reduce_q_head;
                                    *(reinterpret_cast<float*>(LSE + final_lse_idx) + (0)) = merged_lse;
                                }
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
                                float _vec_load_3[8];
                                {
                                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(partial_O + partial_o_base + 0);
                                    uint4 _vld_5[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_5[_blk] = _vptr_5[_blk];
                                        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                                : "r"(_vpairs_5[_pair]));
                                        }
                                    }
                                }
                                float merge_weight0 = split_weights[merge_head];
                                #pragma unroll
                                for (int elem = 0; elem < 8; elem++) {
                                    _vec_load_3[elem] = _vec_load_3[elem] * merge_weight0;
                                }
                                #pragma unroll 2
                                for (int reduce_split = 1; reduce_split < NUM_SPLIT; reduce_split++) {
                                    float _vec_load_4[8];
                                    {
                                        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(partial_O + (partial_o_base + reduce_split * HEAD_DIM) + 0);
                                        uint4 _vld_6[1];
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vld_6[_blk] = _vptr_6[_blk];
                                            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                                                    : "r"(_vpairs_6[_pair]));
                                            }
                                        }
                                    }
                                    float reduce_weight = split_weights[reduce_split * TILE_Q + merge_head];
                                    #pragma unroll
                                    for (int elem_1 = 0; elem_1 < 8; elem_1++) {
                                        float _fma_2 = __fmaf_rn(_vec_load_4[elem_1], reduce_weight, _vec_load_3[elem_1]);
                                        _vec_load_3[elem_1] = _fma_2;
                                    }
                                }
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_vec_load_3[0 + 0], _vec_load_3[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_vec_load_3[0 + 2], _vec_load_3[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_vec_load_3[0 + 4], _vec_load_3[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_vec_load_3[0 + 6], _vec_load_3[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + final_o_idx))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    }
                }
                {
                    int runtime_next_item_1 = bundle_item_idx_c + 1;
                    int runtime_next_id_1 = bundle_idx_c + runtime_next_item_1 * gridDim.z;
                    unsigned int runtime_valid_1 = 0;
                    int runtime_next_total_1 = runtime_plan[0];
                    if (runtime_next_id_1 < runtime_next_total_1) {
                        runtime_valid_1 = 1;
                    }
                    int runtime_virtual_0_1 = runtime_next_id_1;
                    int runtime_tile_total_1_1 = runtime_plan[0];
                    int runtime_uniform_count_2_1 = runtime_plan[1];
                    int runtime_batch_3_1 = 0;
                    int runtime_split_4_1 = 0;
                    int runtime_count_5_1 = 1;
                    if (runtime_virtual_0_1 < runtime_tile_total_1_1) {
                        if (runtime_uniform_count_2_1 != 0) {
                            runtime_count_5_1 = runtime_uniform_count_2_1;
                            runtime_batch_3_1 = runtime_virtual_0_1 / runtime_count_5_1;
                            runtime_split_4_1 = runtime_virtual_0_1 % runtime_count_5_1;
                        } else {
                            int runtime_lo_3 = 0;
                            int runtime_hi_3 = batch_size;
                            #pragma unroll 1
                            for (int runtime_search_3 = 0; runtime_search_3 < 9; runtime_search_3++) {
                                if (runtime_lo_3 < runtime_hi_3) {
                                    int runtime_mid_3 = (runtime_lo_3 + runtime_hi_3) / 2;
                                    int runtime_prefix_end_3 = runtime_prefix[runtime_mid_3 + 1];
                                    if (runtime_prefix_end_3 <= runtime_virtual_0_1) {
                                        runtime_lo_3 = runtime_mid_3 + 1;
                                    } else {
                                        runtime_hi_3 = runtime_mid_3;
                                    }
                                }
                            }
                            runtime_batch_3_1 = runtime_lo_3;
                            int runtime_prefix_start_3 = runtime_prefix[runtime_batch_3_1];
                            runtime_split_4_1 = runtime_virtual_0_1 - runtime_prefix_start_3;
                            int runtime_read_count_3 = runtime_parts[runtime_batch_3_1];
                            runtime_count_5_1 = runtime_read_count_3;
                        }
                    }
                    unsigned int valid_c = runtime_valid_1;
                    batch_idx_c = runtime_batch_3_1;
                    q_row_idx_c = q_row_idx_c;
                    kv_head_idx_c = kv_head_idx_c;
                    split_idx_c = runtime_split_4_1;
                    part_count_c = runtime_count_5_1;
                    bundle_idx_c = bundle_idx_c;
                    bundle_item_idx_c = runtime_next_item_1;
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
            int runtime_virtual_2 = blockIdx.z;
            int runtime_tile_total_3 = runtime_plan[0];
            int runtime_uniform_count_3 = runtime_plan[1];
            int runtime_batch_2 = 0;
            int runtime_split_2 = 0;
            int runtime_count_2 = 1;
            if (runtime_virtual_2 < runtime_tile_total_3) {
                if (runtime_uniform_count_3 != 0) {
                    runtime_count_2 = runtime_uniform_count_3;
                    runtime_batch_2 = runtime_virtual_2 / runtime_count_2;
                    runtime_split_2 = runtime_virtual_2 % runtime_count_2;
                } else {
                    int runtime_lo_4 = 0;
                    int runtime_hi_4 = batch_size;
                    #pragma unroll 1
                    for (int runtime_search_4 = 0; runtime_search_4 < 9; runtime_search_4++) {
                        if (runtime_lo_4 < runtime_hi_4) {
                            int runtime_mid_4 = (runtime_lo_4 + runtime_hi_4) / 2;
                            int runtime_prefix_end_4 = runtime_prefix[runtime_mid_4 + 1];
                            if (runtime_prefix_end_4 <= runtime_virtual_2) {
                                runtime_lo_4 = runtime_mid_4 + 1;
                            } else {
                                runtime_hi_4 = runtime_mid_4;
                            }
                        }
                    }
                    runtime_batch_2 = runtime_lo_4;
                    int runtime_prefix_start_4 = runtime_prefix[runtime_batch_2];
                    runtime_split_2 = runtime_virtual_2 - runtime_prefix_start_4;
                    int runtime_read_count_4 = runtime_parts[runtime_batch_2];
                    runtime_count_2 = runtime_read_count_4;
                }
            }
            int batch_idx_m = runtime_batch_2;
            int q_row_idx_m = blockIdx.x;
            int kv_head_idx_m = blockIdx.y;
            int split_idx_m = runtime_split_2;
            int part_count_m = runtime_count_2;
            int bundle_idx_m = blockIdx.z;
            int bundle_item_idx_m = 0;
            unsigned int runtime_work_2 = 0;
            int runtime_work_total_2 = runtime_plan[0];
            int runtime_remaining_2 = runtime_work_total_2 - blockIdx.z;
            if (runtime_remaining_2 > 0) {
                runtime_work_2 = (unsigned int)((runtime_remaining_2 - 1) / gridDim.z + 1);
            }
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_p_full_0_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < runtime_work_2; _tile_iter_m++) {
                int visible_keys_2 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_m + 1;
                {
                    visible_keys_2 = seq_lens_kv[batch_idx_m] - Q_LEN + q_row_idx_m + 1;
                }
                if (visible_keys_2 < 0) {
                    visible_keys_2 = 0;
                }
                int seqlen_kv_m = visible_keys_2;
                int num_n_blocks_2 = 0;
                num_n_blocks_2 = seqlen_kv_m / BLOCK_N;
                if (seqlen_kv_m % BLOCK_N != 0) {
                    num_n_blocks_2 += 1;
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
                    if (num_pairs_2 * 2 > 0) {
                        mbarrier_wait(s_empty_0_addr + (sm_stage_1) * 8, sm_empty_phase_m);
                        uint32_t _mbar_token_1 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                        mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_1);
                        int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (transformed_stage) * 2048);
                        int _mma_b_lo_0 = make_warp_uniform(((smem_qt_hi_addr) >> 4) & 0x3FFF);
                        {
                            uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                            uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                        }
                        elect_commit(kv_empty_addr + (transformed_stage) * 8);
                        transformed_stage += 1;
                        if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                        uint32_t _mbar_token_2 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                        mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_2);
                        int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (transformed_stage) * 2048);
                        int _mma_b_lo_1 = make_warp_uniform(((smem_qt_lo_addr) >> 4) & 0x3FFF);
                        {
                            uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                            uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                            }
                        }
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
                            {
                                uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                                uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 0);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134349968, 1);
                                }
                            }
                            elect_commit(kv_empty_addr + (transformed_stage) * 8);
                            transformed_stage += 1;
                            if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                            uint32_t _mbar_token_4 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                            mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_4);
                            int _mma_a_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (transformed_stage) * 2048);
                            int _mma_b_lo_3 = make_warp_uniform(((smem_qt_lo_addr) >> 4) & 0x3FFF);
                            {
                                uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                                uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 1018U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s0 + (sm_stage_1 * 8)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134349968, 1);
                                }
                            }
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
                            {
                                uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                                uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, ((first_pv_flag) ? 0 : 1));
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 58U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                                }
                            }
                            elect_commit(kv_empty_addr + (transformed_stage) * 8);
                            transformed_stage += 1;
                            if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                            uint32_t _mbar_token_6 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                            mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_6);
                            int _mma_a_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (transformed_stage) * 2048);
                            int _mma_b_lo_5 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x400000) + (p_stage_m) * 128);
                            {
                                uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                                uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_5);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, ((first_pv_flag) ? 0 : 1));
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 58U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 128U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134382736, 1);
                                }
                            }
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
                        {
                            uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_6);
                            uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_6);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, ((last_pv_init) ? 0 : 1));
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_6, 58U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_hi, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134382736, 1);
                            }
                        }
                        elect_commit(kv_empty_addr + (transformed_stage) * 8);
                        transformed_stage += 1;
                        if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                        uint32_t _mbar_token_8 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                        mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_8);
                        int _mma_a_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (transformed_stage) * 2048);
                        int _mma_b_lo_7 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x400000) + (p_stage_m) * 128);
                        {
                            uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_7);
                            uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, ((last_pv_init) ? 0 : 1));
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_7, 58U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o_lo, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134382736, 1);
                            }
                        }
                        elect_commit(kv_empty_addr + (transformed_stage) * 8);
                        transformed_stage += 1;
                        if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                        elect_commit(o_full_addr);
                        p_stage_m += 1;
                        if (p_stage_m == 2) { p_stage_m = 0; p_phase_m ^= 1; }
                    } else {
                        elect_commit(q_empty_addr);
                    }
                }
                int runtime_next_item_2 = bundle_item_idx_m + 1;
                int runtime_next_id_2 = bundle_idx_m + runtime_next_item_2 * gridDim.z;
                unsigned int runtime_valid_2 = 0;
                int runtime_next_total_2 = runtime_plan[0];
                if (runtime_next_id_2 < runtime_next_total_2) {
                    runtime_valid_2 = 1;
                }
                int runtime_virtual_0_2 = runtime_next_id_2;
                int runtime_tile_total_1_2 = runtime_plan[0];
                int runtime_uniform_count_2_2 = runtime_plan[1];
                int runtime_batch_3_2 = 0;
                int runtime_split_4_2 = 0;
                int runtime_count_5_2 = 1;
                if (runtime_virtual_0_2 < runtime_tile_total_1_2) {
                    if (runtime_uniform_count_2_2 != 0) {
                        runtime_count_5_2 = runtime_uniform_count_2_2;
                        runtime_batch_3_2 = runtime_virtual_0_2 / runtime_count_5_2;
                        runtime_split_4_2 = runtime_virtual_0_2 % runtime_count_5_2;
                    } else {
                        int runtime_lo_5 = 0;
                        int runtime_hi_5 = batch_size;
                        #pragma unroll 1
                        for (int runtime_search_5 = 0; runtime_search_5 < 9; runtime_search_5++) {
                            if (runtime_lo_5 < runtime_hi_5) {
                                int runtime_mid_5 = (runtime_lo_5 + runtime_hi_5) / 2;
                                int runtime_prefix_end_5 = runtime_prefix[runtime_mid_5 + 1];
                                if (runtime_prefix_end_5 <= runtime_virtual_0_2) {
                                    runtime_lo_5 = runtime_mid_5 + 1;
                                } else {
                                    runtime_hi_5 = runtime_mid_5;
                                }
                            }
                        }
                        runtime_batch_3_2 = runtime_lo_5;
                        int runtime_prefix_start_5 = runtime_prefix[runtime_batch_3_2];
                        runtime_split_4_2 = runtime_virtual_0_2 - runtime_prefix_start_5;
                        int runtime_read_count_5 = runtime_parts[runtime_batch_3_2];
                        runtime_count_5_2 = runtime_read_count_5;
                    }
                }
                unsigned int valid_m = runtime_valid_2;
                batch_idx_m = runtime_batch_3_2;
                q_row_idx_m = q_row_idx_m;
                kv_head_idx_m = kv_head_idx_m;
                split_idx_m = runtime_split_4_2;
                part_count_m = runtime_count_5_2;
                bundle_idx_m = bundle_idx_m;
                bundle_item_idx_m = runtime_next_item_2;
                if (valid_m == 0) {
                    break;
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
            int runtime_virtual_3 = blockIdx.z;
            int runtime_tile_total_4 = runtime_plan[0];
            int runtime_uniform_count_4 = runtime_plan[1];
            int runtime_batch_4 = 0;
            int runtime_split_3 = 0;
            int runtime_count_3 = 1;
            if (runtime_virtual_3 < runtime_tile_total_4) {
                if (runtime_uniform_count_4 != 0) {
                    runtime_count_3 = runtime_uniform_count_4;
                    runtime_batch_4 = runtime_virtual_3 / runtime_count_3;
                    runtime_split_3 = runtime_virtual_3 % runtime_count_3;
                } else {
                    int runtime_lo_6 = 0;
                    int runtime_hi_6 = batch_size;
                    #pragma unroll 1
                    for (int runtime_search_6 = 0; runtime_search_6 < 9; runtime_search_6++) {
                        if (runtime_lo_6 < runtime_hi_6) {
                            int runtime_mid_6 = (runtime_lo_6 + runtime_hi_6) / 2;
                            int runtime_prefix_end_6 = runtime_prefix[runtime_mid_6 + 1];
                            if (runtime_prefix_end_6 <= runtime_virtual_3) {
                                runtime_lo_6 = runtime_mid_6 + 1;
                            } else {
                                runtime_hi_6 = runtime_mid_6;
                            }
                        }
                    }
                    runtime_batch_4 = runtime_lo_6;
                    int runtime_prefix_start_6 = runtime_prefix[runtime_batch_4];
                    runtime_split_3 = runtime_virtual_3 - runtime_prefix_start_6;
                    int runtime_read_count_6 = runtime_parts[runtime_batch_4];
                    runtime_count_3 = runtime_read_count_6;
                }
            }
            int batch_idx_p = runtime_batch_4;
            int q_row_idx_p = blockIdx.x;
            int kv_head_idx_p = blockIdx.y;
            int split_idx_p = runtime_split_3;
            int part_count_p = runtime_count_3;
            int bundle_idx_p = blockIdx.z;
            int bundle_item_idx_p = 0;
            unsigned int runtime_work_3 = 0;
            int runtime_work_total_3 = runtime_plan[0];
            int runtime_remaining_3 = runtime_work_total_3 - blockIdx.z;
            if (runtime_remaining_3 > 0) {
                runtime_work_3 = (unsigned int)((runtime_remaining_3 - 1) / gridDim.z + 1);
            }
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < runtime_work_3; _tile_iter_p++) {
                int visible_keys_3 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_p + 1;
                {
                    visible_keys_3 = seq_lens_kv[batch_idx_p] - Q_LEN + q_row_idx_p + 1;
                }
                if (visible_keys_3 < 0) {
                    visible_keys_3 = 0;
                }
                int seqlen_kv_p = visible_keys_3;
                int num_n_blocks_3 = 0;
                num_n_blocks_3 = seqlen_kv_p / BLOCK_N;
                if (seqlen_kv_p % BLOCK_N != 0) {
                    num_n_blocks_3 += 1;
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
                int pages_per_seq_p = 0;
                pages_per_seq_p = seqlen_kv_p / PAGE_SIZE;
                if (seqlen_kv_p % PAGE_SIZE != 0) {
                    pages_per_seq_p += 1;
                }
                int max_page_p = pages_per_seq_p - 1;
                long long pt_base_p = (long long)batch_idx_p * (long long)max_pages_per_seq;
                long long pt_base_v_p = (long long)pt_base_p + (long long)page_table_v_offset;
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
                                    smem_page_offsets[page_smem_base_p + page_in_block_p] = page_table[pt_base_p + (long long)clamped_page_p];
                                    smem_page_offsets[page_smem_base_p + 2 + page_in_block_p] = page_table[pt_base_v_p + (long long)clamped_page_p];
                                }
                            }
                            mbarrier_arrive(page_offsets_full_addr + (page_prod_stage) * 8);
                            page_prod_stage += 1;
                            if (page_prod_stage == 6) { page_prod_stage = 0; page_prod_phase ^= 1; }
                        }
                    }
                }
                int runtime_next_item_3 = bundle_item_idx_p + 1;
                int runtime_next_id_3 = bundle_idx_p + runtime_next_item_3 * gridDim.z;
                unsigned int runtime_valid_3 = 0;
                int runtime_next_total_3 = runtime_plan[0];
                if (runtime_next_id_3 < runtime_next_total_3) {
                    runtime_valid_3 = 1;
                }
                int runtime_virtual_0_3 = runtime_next_id_3;
                int runtime_tile_total_1_3 = runtime_plan[0];
                int runtime_uniform_count_2_3 = runtime_plan[1];
                int runtime_batch_3_3 = 0;
                int runtime_split_4_3 = 0;
                int runtime_count_5_3 = 1;
                if (runtime_virtual_0_3 < runtime_tile_total_1_3) {
                    if (runtime_uniform_count_2_3 != 0) {
                        runtime_count_5_3 = runtime_uniform_count_2_3;
                        runtime_batch_3_3 = runtime_virtual_0_3 / runtime_count_5_3;
                        runtime_split_4_3 = runtime_virtual_0_3 % runtime_count_5_3;
                    } else {
                        int runtime_lo_7 = 0;
                        int runtime_hi_7 = batch_size;
                        #pragma unroll 1
                        for (int runtime_search_7 = 0; runtime_search_7 < 9; runtime_search_7++) {
                            if (runtime_lo_7 < runtime_hi_7) {
                                int runtime_mid_7 = (runtime_lo_7 + runtime_hi_7) / 2;
                                int runtime_prefix_end_7 = runtime_prefix[runtime_mid_7 + 1];
                                if (runtime_prefix_end_7 <= runtime_virtual_0_3) {
                                    runtime_lo_7 = runtime_mid_7 + 1;
                                } else {
                                    runtime_hi_7 = runtime_mid_7;
                                }
                            }
                        }
                        runtime_batch_3_3 = runtime_lo_7;
                        int runtime_prefix_start_7 = runtime_prefix[runtime_batch_3_3];
                        runtime_split_4_3 = runtime_virtual_0_3 - runtime_prefix_start_7;
                        int runtime_read_count_7 = runtime_parts[runtime_batch_3_3];
                        runtime_count_5_3 = runtime_read_count_7;
                    }
                }
                unsigned int valid_p = runtime_valid_3;
                batch_idx_p = runtime_batch_3_3;
                q_row_idx_p = q_row_idx_p;
                kv_head_idx_p = kv_head_idx_p;
                split_idx_p = runtime_split_4_3;
                part_count_p = runtime_count_5_3;
                bundle_idx_p = bundle_idx_p;
                bundle_item_idx_p = runtime_next_item_3;
                if (valid_p == 0) {
                    break;
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
            unsigned int _phase_work_full = 0;
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
            int runtime_virtual_4 = blockIdx.z;
            int runtime_tile_total_5 = runtime_plan[0];
            int runtime_uniform_count_5 = runtime_plan[1];
            int runtime_batch_5 = 0;
            int runtime_split_5 = 0;
            int runtime_count_4 = 1;
            if (runtime_virtual_4 < runtime_tile_total_5) {
                if (runtime_uniform_count_5 != 0) {
                    runtime_count_4 = runtime_uniform_count_5;
                    runtime_batch_5 = runtime_virtual_4 / runtime_count_4;
                    runtime_split_5 = runtime_virtual_4 % runtime_count_4;
                } else {
                    int runtime_lo_8 = 0;
                    int runtime_hi_8 = batch_size;
                    #pragma unroll 1
                    for (int runtime_search_8 = 0; runtime_search_8 < 9; runtime_search_8++) {
                        if (runtime_lo_8 < runtime_hi_8) {
                            int runtime_mid_8 = (runtime_lo_8 + runtime_hi_8) / 2;
                            int runtime_prefix_end_8 = runtime_prefix[runtime_mid_8 + 1];
                            if (runtime_prefix_end_8 <= runtime_virtual_4) {
                                runtime_lo_8 = runtime_mid_8 + 1;
                            } else {
                                runtime_hi_8 = runtime_mid_8;
                            }
                        }
                    }
                    runtime_batch_5 = runtime_lo_8;
                    int runtime_prefix_start_8 = runtime_prefix[runtime_batch_5];
                    runtime_split_5 = runtime_virtual_4 - runtime_prefix_start_8;
                    int runtime_read_count_8 = runtime_parts[runtime_batch_5];
                    runtime_count_4 = runtime_read_count_8;
                }
            }
            int batch_idx = runtime_batch_5;
            int q_row_idx = blockIdx.x;
            int kv_head_idx = blockIdx.y;
            int split_idx_l = runtime_split_5;
            int part_count_l = runtime_count_4;
            int bundle_idx_l = blockIdx.z;
            int bundle_item_idx_l = 0;
            unsigned int runtime_work_4 = 0;
            int runtime_work_total_4 = runtime_plan[0];
            int runtime_remaining_4 = runtime_work_total_4 - blockIdx.z;
            if (runtime_remaining_4 > 0) {
                runtime_work_4 = (unsigned int)((runtime_remaining_4 - 1) / gridDim.z + 1);
            }
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < runtime_work_4; _tile_iter_l++) {
                {
                }
                int visible_keys_4 = UNIFORM_KV_LEN - Q_LEN + q_row_idx + 1;
                {
                    visible_keys_4 = seq_lens_kv[batch_idx] - Q_LEN + q_row_idx + 1;
                }
                if (visible_keys_4 < 0) {
                    visible_keys_4 = 0;
                }
                int seqlen_kv = visible_keys_4;
                int num_n_blocks_4 = 0;
                num_n_blocks_4 = seqlen_kv / BLOCK_N;
                if (seqlen_kv % BLOCK_N != 0) {
                    num_n_blocks_4 += 1;
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
                    int group_ratio_l = num_q_heads / (num_kv_heads * Q_GROUPS_PER_KV);
                    int off_qt = (batch_idx * Q_LEN + q_row_idx) * num_q_heads + kv_head_idx * group_ratio_l;
                    mbarrier_arrive_expect_tx(q_full_addr, TILE_Q * HEAD_DIM * 2);
                    tma_3d_gmem2smem(smem_qt_hi_addr, Qt, 0, off_qt, 0, q_full_addr);
                    tma_3d_gmem2smem(smem_qt_lo_addr, Qt, 0, off_qt, 2, q_full_addr);
                    {
                        {
                            if (num_pairs_4 * 2 > 0) {
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
                                            tma_5d_gmem2smem(page_dst, K, 0, 0, dim_half, kv_head_idx / Q_GROUPS_PER_KV, physical_page, raw_kv_full_addr + (raw_stage) * 8);
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
                                                tma_5d_gmem2smem(page_dst_1, K, 0, 0, dim_half_1, kv_head_idx / Q_GROUPS_PER_KV, physical_page_1, raw_kv_full_addr + (raw_stage) * 8);
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
                                                tma_5d_gmem2smem(page_dst_2, V, 0, 0, dim_half_2, kv_head_idx / Q_GROUPS_PER_KV, physical_page_2, raw_kv_full_addr + (raw_stage) * 8);
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
                                            tma_5d_gmem2smem(page_dst_3, V, 0, 0, dim_half_3, kv_head_idx / Q_GROUPS_PER_KV, physical_page_3, raw_kv_full_addr + (raw_stage) * 8);
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
                }
                {
                    {
                        __syncwarp();
                    }
                }
                int runtime_next_item_4 = bundle_item_idx_l + 1;
                int runtime_next_id_4 = bundle_idx_l + runtime_next_item_4 * gridDim.z;
                unsigned int runtime_valid_4 = 0;
                int runtime_next_total_4 = runtime_plan[0];
                if (runtime_next_id_4 < runtime_next_total_4) {
                    runtime_valid_4 = 1;
                }
                int runtime_virtual_0_4 = runtime_next_id_4;
                int runtime_tile_total_1_4 = runtime_plan[0];
                int runtime_uniform_count_2_4 = runtime_plan[1];
                int runtime_batch_3_4 = 0;
                int runtime_split_4_4 = 0;
                int runtime_count_5_4 = 1;
                if (runtime_virtual_0_4 < runtime_tile_total_1_4) {
                    if (runtime_uniform_count_2_4 != 0) {
                        runtime_count_5_4 = runtime_uniform_count_2_4;
                        runtime_batch_3_4 = runtime_virtual_0_4 / runtime_count_5_4;
                        runtime_split_4_4 = runtime_virtual_0_4 % runtime_count_5_4;
                    } else {
                        int runtime_lo_9 = 0;
                        int runtime_hi_9 = batch_size;
                        #pragma unroll 1
                        for (int runtime_search_9 = 0; runtime_search_9 < 9; runtime_search_9++) {
                            if (runtime_lo_9 < runtime_hi_9) {
                                int runtime_mid_9 = (runtime_lo_9 + runtime_hi_9) / 2;
                                int runtime_prefix_end_9 = runtime_prefix[runtime_mid_9 + 1];
                                if (runtime_prefix_end_9 <= runtime_virtual_0_4) {
                                    runtime_lo_9 = runtime_mid_9 + 1;
                                } else {
                                    runtime_hi_9 = runtime_mid_9;
                                }
                            }
                        }
                        runtime_batch_3_4 = runtime_lo_9;
                        int runtime_prefix_start_9 = runtime_prefix[runtime_batch_3_4];
                        runtime_split_4_4 = runtime_virtual_0_4 - runtime_prefix_start_9;
                        int runtime_read_count_9 = runtime_parts[runtime_batch_3_4];
                        runtime_count_5_4 = runtime_read_count_9;
                    }
                }
                unsigned int valid_l = runtime_valid_4;
                batch_idx = runtime_batch_3_4;
                q_row_idx = q_row_idx;
                kv_head_idx = kv_head_idx;
                split_idx_l = runtime_split_4_4;
                part_count_l = runtime_count_5_4;
                bundle_idx_l = bundle_idx_l;
                bundle_item_idx_l = runtime_next_item_4;
                if (valid_l == 0) {
                    break;
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
            int runtime_virtual_5 = blockIdx.z;
            int runtime_tile_total_6 = runtime_plan[0];
            int runtime_uniform_count_6 = runtime_plan[1];
            int runtime_batch_6 = 0;
            int runtime_split_6 = 0;
            int runtime_count_6 = 1;
            if (runtime_virtual_5 < runtime_tile_total_6) {
                if (runtime_uniform_count_6 != 0) {
                    runtime_count_6 = runtime_uniform_count_6;
                    runtime_batch_6 = runtime_virtual_5 / runtime_count_6;
                    runtime_split_6 = runtime_virtual_5 % runtime_count_6;
                } else {
                    int runtime_lo_10 = 0;
                    int runtime_hi_10 = batch_size;
                    #pragma unroll 1
                    for (int runtime_search_10 = 0; runtime_search_10 < 9; runtime_search_10++) {
                        if (runtime_lo_10 < runtime_hi_10) {
                            int runtime_mid_10 = (runtime_lo_10 + runtime_hi_10) / 2;
                            int runtime_prefix_end_10 = runtime_prefix[runtime_mid_10 + 1];
                            if (runtime_prefix_end_10 <= runtime_virtual_5) {
                                runtime_lo_10 = runtime_mid_10 + 1;
                            } else {
                                runtime_hi_10 = runtime_mid_10;
                            }
                        }
                    }
                    runtime_batch_6 = runtime_lo_10;
                    int runtime_prefix_start_10 = runtime_prefix[runtime_batch_6];
                    runtime_split_6 = runtime_virtual_5 - runtime_prefix_start_10;
                    int runtime_read_count_10 = runtime_parts[runtime_batch_6];
                    runtime_count_6 = runtime_read_count_10;
                }
            }
            int batch_idx_t = runtime_batch_6;
            int q_row_idx_t = blockIdx.x;
            int kv_head_idx_t = blockIdx.y;
            int split_idx_t = runtime_split_6;
            int part_count_t = runtime_count_6;
            int bundle_idx_t = blockIdx.z;
            int bundle_item_idx_t = 0;
            unsigned int runtime_work_5 = 0;
            int runtime_work_total_5 = runtime_plan[0];
            int runtime_remaining_5 = runtime_work_total_5 - blockIdx.z;
            if (runtime_remaining_5 > 0) {
                runtime_work_5 = (unsigned int)((runtime_remaining_5 - 1) / gridDim.z + 1);
            }
            #pragma unroll 1
            for (unsigned int _tile_iter_t = 0; _tile_iter_t < runtime_work_5; _tile_iter_t++) {
                int visible_keys_5 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_t + 1;
                {
                    visible_keys_5 = seq_lens_kv[batch_idx_t] - Q_LEN + q_row_idx_t + 1;
                }
                if (visible_keys_5 < 0) {
                    visible_keys_5 = 0;
                }
                int seqlen_kv_t = visible_keys_5;
                int num_n_blocks_5 = 0;
                num_n_blocks_5 = seqlen_kv_t / BLOCK_N;
                if (seqlen_kv_t % BLOCK_N != 0) {
                    num_n_blocks_5 += 1;
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
                int prefetch_tid_t = threadIdx.x - 384;
                #pragma unroll 1
                for (int _item = 0; _item < total_half_items; _item += 2) {
                    int next_raw_stage_t = raw_stage_t;
                    int next_raw_phase_t = raw_phase_t;
                    next_raw_stage_t += 1;
                    if (next_raw_stage_t == 4) { next_raw_stage_t = 0; next_raw_phase_t ^= 1; }
                    unsigned int current_raw_t[16];
                    unsigned int prefetched_raw_t[32];
                    mbarrier_wait(raw_kv_full_addr + (raw_stage_t) * 8, raw_phase_t);
                    mbarrier_wait(kv_empty_addr + (transformed_stage_t) * 8, transformed_phase_t);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    #pragma unroll
                    for (int first_head_chunk_t = 0; first_head_chunk_t < 8; first_head_chunk_t++) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&current_raw_t[first_head_chunk_t * 2])), "=r"(*reinterpret_cast<uint32_t*>(&current_raw_t[(first_head_chunk_t * 2) + 1]))
                            : "r"(smem_kv_fp8_addr + (unsigned int)(raw_stage_t * 16384) + (unsigned int)((first_head_chunk_t * 128 + prefetch_tid_t) * 8)) : "memory");
                    }
                    #pragma unroll
                    for (int first_head_chunk_t_1 = 0; first_head_chunk_t_1 < 8; first_head_chunk_t_1++) {
                        uint32_t current_raw_t_bf16[4];
                        #pragma unroll
                        for (int _fp8_word_0 = 0; _fp8_word_0 < 2; ++_fp8_word_0) {
                            #pragma unroll
                            for (int _fp8_pair_0 = 0; _fp8_pair_0 < 2; ++_fp8_pair_0) {
                                uint16_t _e4m3x2 = (uint16_t)(current_raw_t[(first_head_chunk_t_1 * 2) + _fp8_word_0] >> (_fp8_pair_0 * 16));
                                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(current_raw_t_bf16[_fp8_word_0 * 2 + _fp8_pair_0]) : "h"(_e4m3x2));
                                #else
                                uint32_t _f16x2;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                float _f0;
                                float _f1;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(current_raw_t_bf16[_fp8_word_0 * 2 + _fp8_pair_0]) : "f"(_f1), "f"(_f0));
                                #endif
                            }
                        }
                        int first_head_element_t = (first_head_chunk_t_1 * 128 + prefetch_tid_t) * 8;
                        int first_head_row_t = first_head_element_t % 128 / 64 * 128 + first_head_element_t / 128;
                        int first_head_byte_t = first_head_row_t * 128 + first_head_element_t % 64 * 16 / 8;
                        int first_head_swizzle_t = first_head_byte_t ^ first_head_row_t % 8 * 16;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_kv_addr + (unsigned int)(transformed_stage_t * 32768) + (unsigned int)first_head_swizzle_t), "r"(*reinterpret_cast<uint32_t*>(&current_raw_t_bf16[0])), "r"(*reinterpret_cast<uint32_t*>(&current_raw_t_bf16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&current_raw_t_bf16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&current_raw_t_bf16[(0) + 3])));
                    }
                    mbarrier_wait(raw_kv_full_addr + (next_raw_stage_t) * 8, next_raw_phase_t);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    #pragma unroll
                    for (int next_head_chunk_t = 0; next_head_chunk_t < 16; next_head_chunk_t++) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&prefetched_raw_t[next_head_chunk_t * 2])), "=r"(*reinterpret_cast<uint32_t*>(&prefetched_raw_t[(next_head_chunk_t * 2) + 1]))
                            : "r"(smem_kv_fp8_addr + (unsigned int)(next_raw_stage_t * 16384) + (unsigned int)((next_head_chunk_t * 128 + prefetch_tid_t) * 8)) : "memory");
                    }
                    #pragma unroll
                    for (int first_tail_chunk_t = 0; first_tail_chunk_t < 8; first_tail_chunk_t++) {
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&current_raw_t[first_tail_chunk_t * 2])), "=r"(*reinterpret_cast<uint32_t*>(&current_raw_t[(first_tail_chunk_t * 2) + 1]))
                            : "r"(smem_kv_fp8_addr + (unsigned int)(raw_stage_t * 16384) + (unsigned int)(((first_tail_chunk_t + 8) * 128 + prefetch_tid_t) * 8)) : "memory");
                    }
                    #pragma unroll
                    for (int first_tail_chunk_t_1 = 0; first_tail_chunk_t_1 < 8; first_tail_chunk_t_1++) {
                        uint32_t current_raw_t_bf16_1[4];
                        #pragma unroll
                        for (int _fp8_word_1 = 0; _fp8_word_1 < 2; ++_fp8_word_1) {
                            #pragma unroll
                            for (int _fp8_pair_1 = 0; _fp8_pair_1 < 2; ++_fp8_pair_1) {
                                uint16_t _e4m3x2 = (uint16_t)(current_raw_t[(first_tail_chunk_t_1 * 2) + _fp8_word_1] >> (_fp8_pair_1 * 16));
                                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(current_raw_t_bf16_1[_fp8_word_1 * 2 + _fp8_pair_1]) : "h"(_e4m3x2));
                                #else
                                uint32_t _f16x2;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                float _f0;
                                float _f1;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(current_raw_t_bf16_1[_fp8_word_1 * 2 + _fp8_pair_1]) : "f"(_f1), "f"(_f0));
                                #endif
                            }
                        }
                        int first_tail_element_t = ((first_tail_chunk_t_1 + 8) * 128 + prefetch_tid_t) * 8;
                        int first_tail_row_t = first_tail_element_t % 128 / 64 * 128 + first_tail_element_t / 128;
                        int first_tail_byte_t = first_tail_row_t * 128 + first_tail_element_t % 64 * 16 / 8;
                        int first_tail_swizzle_t = first_tail_byte_t ^ first_tail_row_t % 8 * 16;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_kv_addr + (unsigned int)(transformed_stage_t * 32768) + (unsigned int)first_tail_swizzle_t), "r"(*reinterpret_cast<uint32_t*>(&current_raw_t_bf16_1[0])), "r"(*reinterpret_cast<uint32_t*>(&current_raw_t_bf16_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&current_raw_t_bf16_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&current_raw_t_bf16_1[(0) + 3])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(raw_kv_empty_addr + (raw_stage_t) * 8);
                        mbarrier_arrive(kv_full_addr + (transformed_stage_t) * 8);
                    }
                    raw_stage_t += 1;
                    if (raw_stage_t == 4) { raw_stage_t = 0; raw_phase_t ^= 1; }
                    transformed_stage_t += 1;
                    if (transformed_stage_t == 2) { transformed_stage_t = 0; transformed_phase_t ^= 1; }
                    mbarrier_wait(kv_empty_addr + (transformed_stage_t) * 8, transformed_phase_t);
                    #pragma unroll
                    for (int next_head_chunk_t_1 = 0; next_head_chunk_t_1 < 8; next_head_chunk_t_1++) {
                        uint32_t prefetched_raw_t_bf16[4];
                        #pragma unroll
                        for (int _fp8_word_2 = 0; _fp8_word_2 < 2; ++_fp8_word_2) {
                            #pragma unroll
                            for (int _fp8_pair_2 = 0; _fp8_pair_2 < 2; ++_fp8_pair_2) {
                                uint16_t _e4m3x2 = (uint16_t)(prefetched_raw_t[(next_head_chunk_t_1 * 2) + _fp8_word_2] >> (_fp8_pair_2 * 16));
                                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(prefetched_raw_t_bf16[_fp8_word_2 * 2 + _fp8_pair_2]) : "h"(_e4m3x2));
                                #else
                                uint32_t _f16x2;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                float _f0;
                                float _f1;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(prefetched_raw_t_bf16[_fp8_word_2 * 2 + _fp8_pair_2]) : "f"(_f1), "f"(_f0));
                                #endif
                            }
                        }
                        int next_head_element_t = (next_head_chunk_t_1 * 128 + prefetch_tid_t) * 8;
                        int next_head_row_t = next_head_element_t % 128 / 64 * 128 + next_head_element_t / 128;
                        int next_head_byte_t = next_head_row_t * 128 + next_head_element_t % 64 * 16 / 8;
                        int next_head_swizzle_t = next_head_byte_t ^ next_head_row_t % 8 * 16;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_kv_addr + (unsigned int)(transformed_stage_t * 32768) + (unsigned int)next_head_swizzle_t), "r"(*reinterpret_cast<uint32_t*>(&prefetched_raw_t_bf16[0])), "r"(*reinterpret_cast<uint32_t*>(&prefetched_raw_t_bf16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&prefetched_raw_t_bf16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&prefetched_raw_t_bf16[(0) + 3])));
                    }
                    #pragma unroll
                    for (int next_tail_chunk_t = 0; next_tail_chunk_t < 8; next_tail_chunk_t++) {
                        uint32_t prefetched_raw_t_bf16_1[4];
                        #pragma unroll
                        for (int _fp8_word_3 = 0; _fp8_word_3 < 2; ++_fp8_word_3) {
                            #pragma unroll
                            for (int _fp8_pair_3 = 0; _fp8_pair_3 < 2; ++_fp8_pair_3) {
                                uint16_t _e4m3x2 = (uint16_t)(prefetched_raw_t[((next_tail_chunk_t + 8) * 2) + _fp8_word_3] >> (_fp8_pair_3 * 16));
                                #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(prefetched_raw_t_bf16_1[_fp8_word_3 * 2 + _fp8_pair_3]) : "h"(_e4m3x2));
                                #else
                                uint32_t _f16x2;
                                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                float _f0;
                                float _f1;
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(prefetched_raw_t_bf16_1[_fp8_word_3 * 2 + _fp8_pair_3]) : "f"(_f1), "f"(_f0));
                                #endif
                            }
                        }
                        int next_tail_element_t = ((next_tail_chunk_t + 8) * 128 + prefetch_tid_t) * 8;
                        int next_tail_row_t = next_tail_element_t % 128 / 64 * 128 + next_tail_element_t / 128;
                        int next_tail_byte_t = next_tail_row_t * 128 + next_tail_element_t % 64 * 16 / 8;
                        int next_tail_swizzle_t = next_tail_byte_t ^ next_tail_row_t % 8 * 16;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_kv_addr + (unsigned int)(transformed_stage_t * 32768) + (unsigned int)next_tail_swizzle_t), "r"(*reinterpret_cast<uint32_t*>(&prefetched_raw_t_bf16_1[0])), "r"(*reinterpret_cast<uint32_t*>(&prefetched_raw_t_bf16_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&prefetched_raw_t_bf16_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&prefetched_raw_t_bf16_1[(0) + 3])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
                int runtime_next_item_5 = bundle_item_idx_t + 1;
                int runtime_next_id_5 = bundle_idx_t + runtime_next_item_5 * gridDim.z;
                unsigned int runtime_valid_5 = 0;
                int runtime_next_total_5 = runtime_plan[0];
                if (runtime_next_id_5 < runtime_next_total_5) {
                    runtime_valid_5 = 1;
                }
                int runtime_virtual_0_5 = runtime_next_id_5;
                int runtime_tile_total_1_5 = runtime_plan[0];
                int runtime_uniform_count_2_5 = runtime_plan[1];
                int runtime_batch_3_5 = 0;
                int runtime_split_4_5 = 0;
                int runtime_count_5_5 = 1;
                if (runtime_virtual_0_5 < runtime_tile_total_1_5) {
                    if (runtime_uniform_count_2_5 != 0) {
                        runtime_count_5_5 = runtime_uniform_count_2_5;
                        runtime_batch_3_5 = runtime_virtual_0_5 / runtime_count_5_5;
                        runtime_split_4_5 = runtime_virtual_0_5 % runtime_count_5_5;
                    } else {
                        int runtime_lo_11 = 0;
                        int runtime_hi_11 = batch_size;
                        #pragma unroll 1
                        for (int runtime_search_11 = 0; runtime_search_11 < 9; runtime_search_11++) {
                            if (runtime_lo_11 < runtime_hi_11) {
                                int runtime_mid_11 = (runtime_lo_11 + runtime_hi_11) / 2;
                                int runtime_prefix_end_11 = runtime_prefix[runtime_mid_11 + 1];
                                if (runtime_prefix_end_11 <= runtime_virtual_0_5) {
                                    runtime_lo_11 = runtime_mid_11 + 1;
                                } else {
                                    runtime_hi_11 = runtime_mid_11;
                                }
                            }
                        }
                        runtime_batch_3_5 = runtime_lo_11;
                        int runtime_prefix_start_11 = runtime_prefix[runtime_batch_3_5];
                        runtime_split_4_5 = runtime_virtual_0_5 - runtime_prefix_start_11;
                        int runtime_read_count_11 = runtime_parts[runtime_batch_3_5];
                        runtime_count_5_5 = runtime_read_count_11;
                    }
                }
                unsigned int valid_t = runtime_valid_5;
                batch_idx_t = runtime_batch_3_5;
                q_row_idx_t = q_row_idx_t;
                kv_head_idx_t = kv_head_idx_t;
                split_idx_t = runtime_split_4_5;
                part_count_t = runtime_count_5_5;
                bundle_idx_t = bundle_idx_t;
                bundle_item_idx_t = runtime_next_item_5;
                if (valid_t == 0) {
                    break;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
