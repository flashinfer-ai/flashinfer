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
#define SMEM_SPLIT_WEIGHTS_OFF 1408
#define SMEM_SPLIT_WEIGHTS_STAGE_BYTES 512
#define SMEM_SPLIT_WEIGHTS_STRIDE 512
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
#define SMEM_TOTAL 145408
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
#define NUM_SPLIT 16
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
kernel_cake_fmha_request_ordered_paged_decode_runtime_lengths_one_page_78016ad9cfcc62004dc2(CakeTensorMap const* Qt, CakeTensorMap const* K, CakeTensorMap const* V, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_LSE, unsigned int* __restrict__ split_completion, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, int* __restrict__ request_order, int max_pages_per_seq, int page_table_v_offset, float softmax_scale_log2, float output_scale, float* __restrict__ bmm1_scale_ptr, float* __restrict__ bmm2_scale_ptr, int bmm1_is_log2, int num_q_heads, int num_kv_heads, int batch_size, unsigned int total_tiles)
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

    // Mbarrier init (20 pipeline groups, 0 ordered-sequence groups, 47 barriers)
    // Mbarriers at smem_raw[0..376)

    if (warp == 9) {
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
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    // Publish explicit kernel-setup mbarrier initialization.
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");

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
            int one_page_batch = blockIdx.z;
            int batch_idx_s = one_page_batch;
            int q_row_idx_s = blockIdx.x;
            int kv_head_idx_s = blockIdx.y;
            int split_idx_s = 0;
            int part_count_s = 1;
            int bundle_idx_s = blockIdx.z;
            int bundle_item_idx_s = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < 1; _tile_iter_s++) {
                int visible_keys = UNIFORM_KV_LEN - Q_LEN + q_row_idx_s + 1;
                {
                    visible_keys = seq_lens_kv[batch_idx_s] - Q_LEN + q_row_idx_s + 1;
                }
                if (visible_keys < 0) {
                    visible_keys = 0;
                }
                int seqlen_kv_s = visible_keys;
                int one_page_blocks = 0;
                if (seqlen_kv_s > 0) {
                    one_page_blocks = 1;
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
                for (int n = 0; n < one_page_blocks; n++) {
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
                    int my_block = n;
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
                    float _max_7 = max_noftz(pair_max[0], sv[0]);
                    pair_max[0] = _max_7;
                    float _max_8 = max_noftz(pair_max[0], sv[2]);
                    pair_max[0] = _max_8;
                    float _max_9 = max_noftz(pair_max[1], sv[1]);
                    pair_max[1] = _max_9;
                    float _max_10 = max_noftz(pair_max[1], sv[3]);
                    pair_max[1] = _max_10;
                    float _max_11 = max_noftz(pair_max[0], sv[4]);
                    pair_max[0] = _max_11;
                    float _max_12 = max_noftz(pair_max[0], sv[6]);
                    pair_max[0] = _max_12;
                    float _max_13 = max_noftz(pair_max[1], sv[5]);
                    pair_max[1] = _max_13;
                    float _max_14 = max_noftz(pair_max[1], sv[7]);
                    pair_max[1] = _max_14;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 2; c_1++) {
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 16);
                        float _max_15 = max_noftz(pair_max[c_1], _shfl_xor_3);
                        pair_max[c_1] = _max_15;
                        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 8);
                        float _max_16 = max_noftz(pair_max[c_1], _shfl_xor_4);
                        pair_max[c_1] = _max_16;
                    }
                    float new_max_pair[2];
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 2; c_2++) {
                        float _max_17 = max_noftz(row_max_pair[c_2], pair_max[c_2]);
                        new_max_pair[c_2] = _max_17;
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
            int one_page_batch_1 = blockIdx.z;
            int batch_idx_c = one_page_batch_1;
            int q_row_idx_c = blockIdx.x;
            int kv_head_idx_c = blockIdx.y;
            int split_idx_c = 0;
            int part_count_c = 1;
            int bundle_idx_c = blockIdx.z;
            int bundle_item_idx_c = 0;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < 1; _tile_iter_c++) {
                int visible_keys_1 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_c + 1;
                {
                    visible_keys_1 = seq_lens_kv[batch_idx_c] - Q_LEN + q_row_idx_c + 1;
                }
                if (visible_keys_1 < 0) {
                    visible_keys_1 = 0;
                }
                int seqlen_kv_c = visible_keys_1;
                int one_page_blocks_1 = 0;
                if (seqlen_kv_c > 0) {
                    one_page_blocks_1 = 1;
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
                for (int _n = 0; _n < one_page_blocks_1; _n++) {
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
                if (one_page_blocks_1 > 0) {
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                float reduced_sum_pair[2];
                #pragma unroll
                for (int c_8 = 0; c_8 < 2; c_8++) {
                    reduced_sum_pair[c_8] = _tmem_load_3[c_8];
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, reduced_sum_pair[c_8], 16);
                    reduced_sum_pair[c_8] = reduced_sum_pair[c_8] + _shfl_xor_5;
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, reduced_sum_pair[c_8], 8);
                    reduced_sum_pair[c_8] = reduced_sum_pair[c_8] + _shfl_xor_6;
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, reduced_sum_pair[c_8], 4);
                    reduced_sum_pair[c_8] = reduced_sum_pair[c_8] + _shfl_xor_7;
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
                            {
                                {
                                    float _log2_0;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(stats_sum_c));
                                    float lse_value_final_seg = _log2_0 + stats_max_c * bmm1_scale_log2_c;
                                    if (one_page_blocks_1 == 0) {
                                        lse_value_final_seg = -CAKE_INF;
                                    }
                                    *(reinterpret_cast<float*>(LSE + stats_output_row_c) + (0)) = lse_value_final_seg;
                                }
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
                if (one_page_blocks_1 > 0) {
                    tmem_ld_x8(&split_o_hi_epi[0], taddr + 32 + (unsigned int)corr_row);
                    tmem_ld_x8(&split_o_lo_epi[0], taddr + 40 + (unsigned int)corr_row);
                }
                #pragma unroll
                for (int h_2 = 0; h_2 < 8; h_2++) {
                    float final_o_hi = 0.0f;
                    float final_o_lo = 0.0f;
                    if (one_page_blocks_1 > 0) {
                        float _rcp_0 = approx_rcp(total_sum[h_2]);
                        float inv_total = _rcp_0;
                        final_o_hi = split_o_hi_epi[h_2] * inv_total * bmm2_scale_c;
                        final_o_lo = split_o_lo_epi[h_2] * inv_total * bmm2_scale_c;
                    }
                    if (group_ratio_rt > h_2) {
                        int q_head = kv_head_idx_c * group_ratio_rt + h_2;
                        long long output_row = ((long long)batch_idx_c * (long long)Q_LEN + (long long)q_row_idx_c) * (long long)num_q_heads + (long long)q_head;
                        {
                            {
                                long long direct_out_idx_hi_seg = (long long)output_row * (long long)HEAD_DIM + (long long)d_idx;
                                long long direct_out_idx_lo_seg = (long long)direct_out_idx_hi_seg + (long long)HEAD_DIM_HALF;
                                *(reinterpret_cast<__nv_bfloat16*>(O + direct_out_idx_hi_seg) + (0)) = __float2bfloat16_rn(final_o_hi);
                                *(reinterpret_cast<__nv_bfloat16*>(O + direct_out_idx_lo_seg) + (0)) = __float2bfloat16_rn(final_o_lo);
                            }
                        }
                    }
                }
                if (one_page_blocks_1 > 0) {
                    mbarrier_arrive(o_empty_addr);
                }
                {
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
            int one_page_batch_2 = blockIdx.z;
            int batch_idx_m = one_page_batch_2;
            int q_row_idx_m = blockIdx.x;
            int kv_head_idx_m = blockIdx.y;
            int split_idx_m = 0;
            int part_count_m = 1;
            int bundle_idx_m = blockIdx.z;
            int bundle_item_idx_m = 0;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_p_full_0_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < 1; _tile_iter_m++) {
                int visible_keys_2 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_m + 1;
                {
                    visible_keys_2 = seq_lens_kv[batch_idx_m] - Q_LEN + q_row_idx_m + 1;
                }
                if (visible_keys_2 < 0) {
                    visible_keys_2 = 0;
                }
                int seqlen_kv_m = visible_keys_2;
                int one_page_blocks_2 = 0;
                if (seqlen_kv_m > 0) {
                    one_page_blocks_2 = 1;
                }
                int first_pv = 1;
                {
                    uint32_t _mbar_token_0 = mbarrier_try_wait(q_full_addr, q_phase_m);
                    mbarrier_wait_token(q_full_addr, q_phase_m, _mbar_token_0);
                    q_phase_m ^= 1;
                    if (one_page_blocks_2 > 0) {
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
                        for (int _body_n_m = 0; _body_n_m < one_page_blocks_2 - 1; _body_n_m++) {
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
                        mbarrier_wait(s_empty_0_addr + (sm_stage_1 ^ 1) * 8, sm_empty_phase_m ^ sm_stage_1);
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
            int one_page_batch_3 = blockIdx.z;
            int batch_idx = one_page_batch_3;
            int q_row_idx = blockIdx.x;
            int kv_head_idx = blockIdx.y;
            int split_idx_l = 0;
            int part_count_l = 1;
            int bundle_idx_l = blockIdx.z;
            int bundle_item_idx_l = 0;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < 1; _tile_iter_l++) {
                {
                }
                int visible_keys_3 = UNIFORM_KV_LEN - Q_LEN + q_row_idx + 1;
                {
                    visible_keys_3 = seq_lens_kv[batch_idx] - Q_LEN + q_row_idx + 1;
                }
                if (visible_keys_3 < 0) {
                    visible_keys_3 = 0;
                }
                int seqlen_kv = visible_keys_3;
                int one_page_blocks_3 = 0;
                if (seqlen_kv > 0) {
                    one_page_blocks_3 = 1;
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
                    if (one_page_blocks_3 > 0) {
                        long long one_page_pt_l = (long long)batch_idx * (long long)max_pages_per_seq;
                        int one_page_k_l = page_table[one_page_pt_l];
                        int one_page_v_l = page_table[one_page_pt_l + (long long)page_table_v_offset];
                        #pragma unroll
                        for (int dim_half = 0; dim_half < 2; dim_half++) {
                            mbarrier_wait(raw_kv_empty_addr + (raw_stage) * 8, raw_phase);
                            mbarrier_arrive_expect_tx(raw_kv_full_addr + (raw_stage) * 8, BLOCK_N * HEAD_DIM_HALF);
                            int raw_dst = smem_kv_fp8_addr + (unsigned int)(raw_stage * 16384);
                            #pragma unroll
                            for (int page_in_block = 0; page_in_block < 2; page_in_block++) {
                                int physical_page = one_page_k_l;
                                int page_dst = raw_dst + page_in_block * PAGE_SIZE * HEAD_DIM_HALF;
                                {
                                    tma_5d_gmem2smem(page_dst, K, 0, 0, dim_half, kv_head_idx / Q_GROUPS_PER_KV, physical_page, raw_kv_full_addr + (raw_stage) * 8);
                                }
                            }
                            raw_stage += 1;
                            if (raw_stage == 4) { raw_stage = 0; raw_phase ^= 1; }
                        }
                        #pragma unroll
                        for (int dim_half_1 = 0; dim_half_1 < 2; dim_half_1++) {
                            mbarrier_wait(raw_kv_empty_addr + (raw_stage) * 8, raw_phase);
                            mbarrier_arrive_expect_tx(raw_kv_full_addr + (raw_stage) * 8, BLOCK_N * HEAD_DIM_HALF);
                            int raw_dst_1 = smem_kv_fp8_addr + (unsigned int)(raw_stage * 16384);
                            #pragma unroll
                            for (int page_in_block_1 = 0; page_in_block_1 < 2; page_in_block_1++) {
                                int physical_page_1 = one_page_v_l;
                                int page_dst_1 = raw_dst_1 + page_in_block_1 * PAGE_SIZE * HEAD_DIM_HALF;
                                {
                                    tma_5d_gmem2smem(page_dst_1, V, 0, 0, dim_half_1, kv_head_idx / Q_GROUPS_PER_KV, physical_page_1, raw_kv_full_addr + (raw_stage) * 8);
                                }
                            }
                            raw_stage += 1;
                            if (raw_stage == 4) { raw_stage = 0; raw_phase ^= 1; }
                        }
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
            int one_page_batch_4 = blockIdx.z;
            int batch_idx_t = one_page_batch_4;
            int q_row_idx_t = blockIdx.x;
            int kv_head_idx_t = blockIdx.y;
            int split_idx_t = 0;
            int part_count_t = 1;
            int bundle_idx_t = blockIdx.z;
            int bundle_item_idx_t = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_t = 0; _tile_iter_t < 1; _tile_iter_t++) {
                int visible_keys_4 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_t + 1;
                {
                    visible_keys_4 = seq_lens_kv[batch_idx_t] - Q_LEN + q_row_idx_t + 1;
                }
                if (visible_keys_4 < 0) {
                    visible_keys_4 = 0;
                }
                int seqlen_kv_t = visible_keys_4;
                int one_page_blocks_4 = 0;
                if (seqlen_kv_t > 0) {
                    one_page_blocks_4 = 1;
                }
                int total_half_items = one_page_blocks_4 * 4;
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
            }
        }
    }

    // Cleanup
}

} // extern "C"
