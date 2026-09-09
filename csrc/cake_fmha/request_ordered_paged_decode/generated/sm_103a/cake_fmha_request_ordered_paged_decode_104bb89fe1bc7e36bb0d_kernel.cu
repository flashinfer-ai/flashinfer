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
#define SMEM_SMEM_CORR_OFF 173184
#define SMEM_SMEM_CORR_STAGE_BYTES 128
#define SMEM_SMEM_CORR_STRIDE 128
#define SMEM_SMEM_EXCH_OFF 172928
#define SMEM_SMEM_EXCH_STAGE_BYTES 256
#define SMEM_SMEM_EXCH_STRIDE 256
#define SMEM_SMEM_EXCH_U32_OFF 172928
#define SMEM_SMEM_EXCH_U32_STAGE_BYTES 256
#define SMEM_SMEM_EXCH_U32_STRIDE 256
#define SMEM_SMEM_QT_HI_OFF 0
#define SMEM_SMEM_QT_HI_STAGE_BYTES 2048
#define SMEM_SMEM_QT_HI_STRIDE 2048
#define SMEM_SMEM_QT_LO_OFF 2048
#define SMEM_SMEM_QT_LO_STAGE_BYTES 2048
#define SMEM_SMEM_QT_LO_STRIDE 2048
#define SMEM_SMEM_KV_FP8_OFF 4096
#define SMEM_SMEM_KV_FP8_STAGE_BYTES 16384
#define SMEM_SMEM_KV_FP8_STRIDE 16384
#define SMEM_SMEM_KV_OFF 69632
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 69632
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P_OFF 135168
#define SMEM_SMEM_P_STAGE_BYTES 2048
#define SMEM_SMEM_P_STRIDE 2048
#define SMEM_SMEM_PAGE_OFFSETS_OFF 139264
#define SMEM_SMEM_PAGE_OFFSETS_STAGE_BYTES 768
#define SMEM_SMEM_PAGE_OFFSETS_STRIDE 768
#define SMEM_WORK_RESPONSE_VIEW_OFF 172928
#define SMEM_WORK_RESPONSE_VIEW_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_VIEW_STRIDE 16
#define SMEM_SMEM_CGA_O_OFF 140032
#define SMEM_SMEM_CGA_O_STAGE_BYTES 33280
#define SMEM_SMEM_CGA_O_STRIDE 33280
#define SMEM_SMEM_CGA_STATS_OFF 140032
#define SMEM_SMEM_CGA_STATS_STAGE_BYTES 33280
#define SMEM_SMEM_CGA_STATS_STRIDE 33280
#define SMEM_TOTAL 173824
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
#define NUM_SPLIT 1
#define USE_SEGMENTED_CLC 0
#define USE_HIGH_BATCH_TWO_WAVE 0
#define USE_TWO_CTA_REDUCER 0
#define USE_MMA_LOOP_PEEL 1
#define USE_PAGE_OFFSET_CPASYNC 0
#define USE_LEGACY_PAGE_VEC4 1
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


__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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


__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}


__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
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

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,1,1) void
kernel_cake_fmha_request_ordered_paged_decode_104bb89fe1bc7e36bb0d(CakeTensorMap const* Qt, CakeTensorMap const* K, CakeTensorMap const* V, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_LSE, unsigned int* __restrict__ split_completion, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, int* __restrict__ request_order, int max_pages_per_seq, int page_table_v_offset, float softmax_scale_log2, float output_scale, float* __restrict__ bmm1_scale_ptr, float* __restrict__ bmm2_scale_ptr, int bmm1_is_log2, int num_q_heads, int num_kv_heads, int batch_size, unsigned int total_tiles)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 173312;
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
    #define low_q1_cga_reduction_addr (mbar_base + 376)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Qt)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(V)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* smem_corr = reinterpret_cast<float*>(smem_raw + 173184);
    const int smem_corr_addr = smem + 173184;
    float* smem_exch = reinterpret_cast<float*>(smem_raw + 172928);
    const int smem_exch_addr = smem + 172928;
    unsigned int* smem_exch_u32 = reinterpret_cast<unsigned int*>(smem_raw + 172928);
    const int smem_exch_u32_addr = smem + 172928;
    __nv_bfloat16* smem_qt_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_qt_hi_addr = smem + 0;
    __nv_bfloat16* smem_qt_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_qt_lo_addr = smem + 2048;
    uint8_t* smem_kv_fp8 = reinterpret_cast<uint8_t*>(smem_raw + 4096);
    const int smem_kv_fp8_addr = smem + 4096;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 69632);
    const int smem_kv_addr = smem + 69632;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 69632);
    const int smem_v_addr = smem + 69632;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 135168);
    const int smem_p_addr = smem + 135168;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 139264);
    const int smem_page_offsets_addr = smem + 139264;
    unsigned int* work_response_view = reinterpret_cast<unsigned int*>(smem_raw + 172928);
    const int work_response_view_addr = smem + 172928;
    __nv_bfloat16* smem_cga_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140032);
    const int smem_cga_o_addr = smem + 140032;
    float* smem_cga_stats = reinterpret_cast<float*>(smem_raw + 140032);
    const int smem_cga_stats_addr = smem + 140032;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Qt)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory");

    // Mbarrier init (21 pipeline groups, 0 ordered-sequence groups, 48 barriers)
    // Mbarriers at smem_raw[173312..173696)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 173312, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 173320, 1);
            // raw_kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 173328, 1);
            mbarrier_init(smem + 173336, 1);
            mbarrier_init(smem + 173344, 1);
            mbarrier_init(smem + 173352, 1);
            // raw_kv_empty: 4 barriers, init_count=4
            mbarrier_init(smem + 173360, 4);
            mbarrier_init(smem + 173368, 4);
            mbarrier_init(smem + 173376, 4);
            mbarrier_init(smem + 173384, 4);
            // kv_full: 2 barriers, init_count=4
            mbarrier_init(smem + 173392, 4);
            mbarrier_init(smem + 173400, 4);
            // kv_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 173408, 1);
            mbarrier_init(smem + 173416, 1);
            // --- pipeline 'sm_pipe' ---
            // s_full_0: 2 barriers, init_count=1
            mbarrier_init(smem + 173424, 1);
            mbarrier_init(smem + 173432, 1);
            // s_empty_0: 2 barriers, init_count=128
            mbarrier_init(smem + 173440, 128);
            mbarrier_init(smem + 173448, 128);
            // --- pipeline 'p_pipe' ---
            // p_full_0: 2 barriers, init_count=256
            mbarrier_init(smem + 173456, 256);
            mbarrier_init(smem + 173464, 256);
            // --- pipeline 'corr_pipe' ---
            // corr_scale_0: 2 barriers, init_count=128
            mbarrier_init(smem + 173472, 128);
            mbarrier_init(smem + 173480, 128);
            // corr_empty_0: 2 barriers, init_count=128
            mbarrier_init(smem + 173488, 128);
            mbarrier_init(smem + 173496, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 173504, 1);
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 173512, 128);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 173520, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 173528, 1);
            mbarrier_init(smem + 173536, 1);
            // work_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 173544, 512);
            mbarrier_init(smem + 173552, 512);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 2 barriers, init_count=32
            mbarrier_init(smem + 173560, 32);
            mbarrier_init(smem + 173568, 32);
            // throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 173576, 32);
            mbarrier_init(smem + 173584, 32);
            // --- pipeline 'page_pipe' ---
            // page_offsets_full: 6 barriers, init_count=32
            mbarrier_init(smem + 173592, 32);
            mbarrier_init(smem + 173600, 32);
            mbarrier_init(smem + 173608, 32);
            mbarrier_init(smem + 173616, 32);
            mbarrier_init(smem + 173624, 32);
            mbarrier_init(smem + 173632, 32);
            // page_offsets_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 173640, 1);
            mbarrier_init(smem + 173648, 1);
            mbarrier_init(smem + 173656, 1);
            mbarrier_init(smem + 173664, 1);
            mbarrier_init(smem + 173672, 1);
            mbarrier_init(smem + 173680, 1);
            // low_q1_cga_reduction: 1 barriers, init_count=1
            mbarrier_init(smem + 173688, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 173688, 33280);
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 48 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 173696);
    if (warp == 0) {
        int _tmem_hold = smem + 173696;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
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
            q_row_idx = 0;
            int schedule_batch_idx = blockIdx.z;
            split_idx = cta_rank;
            part_count = 2;
            batch_idx = schedule_batch_idx;
            {
                batch_idx = request_order[schedule_batch_idx];
            }
            int batch_idx_s = batch_idx;
            int q_row_idx_s = q_row_idx;
            int kv_head_idx_s = kv_head_idx;
            int split_idx_s = split_idx;
            int part_count_s = part_count;
            int bundle_idx_s = bundle_idx;
            int bundle_item_idx_s = bundle_item_idx;
            if (cta_rank < 2) {
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(low_q1_cga_reduction_addr);
                    }
                }
            }
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
                base_pairs = total_pairs / 2;
                extra_pairs = total_pairs % 2;
                int num_pairs = base_pairs;
                int split_start_pair = extra_pairs * (base_pairs + 1) + (split_idx_s - extra_pairs) * base_pairs;
                if (split_idx_s < extra_pairs) {
                    num_pairs = base_pairs + 1;
                    split_start_pair = split_idx_s * (base_pairs + 1);
                }
                if (num_pairs < 1) {
                    num_pairs = 1;
                    split_start_pair = split_idx_s;
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
            q_row_idx_1 = 0;
            int schedule_batch_idx_1 = blockIdx.z;
            split_idx_1 = cta_rank;
            part_count_1 = 2;
            batch_idx_1 = schedule_batch_idx_1;
            {
                batch_idx_1 = request_order[schedule_batch_idx_1];
            }
            int batch_idx_c = batch_idx_1;
            int q_row_idx_c = q_row_idx_1;
            int kv_head_idx_c = kv_head_idx_1;
            int split_idx_c = split_idx_1;
            int part_count_c = part_count_1;
            int bundle_idx_c = bundle_idx_1;
            int bundle_item_idx_c = bundle_item_idx_1;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_low_q1_cga_reduction_0 = 0;
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
                base_pairs_1 = total_pairs_1 / 2;
                extra_pairs_1 = total_pairs_1 % 2;
                int num_pairs_1 = base_pairs_1;
                int split_start_pair_1 = extra_pairs_1 * (base_pairs_1 + 1) + (split_idx_c - extra_pairs_1) * base_pairs_1;
                if (split_idx_c < extra_pairs_1) {
                    num_pairs_1 = base_pairs_1 + 1;
                    split_start_pair_1 = split_idx_c * (base_pairs_1 + 1);
                }
                if (num_pairs_1 < 1) {
                    num_pairs_1 = 1;
                    split_start_pair_1 = split_idx_c;
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
                float cga_partial_hi[8];
                float cga_partial_lo[8];
                #pragma unroll
                for (int h_2 = 0; h_2 < 8; h_2++) {
                    cga_partial_hi[h_2] = _tmem_load_4[h_2] * bmm2_scale_c;
                    cga_partial_lo[h_2] = _tmem_load_5[h_2] * bmm2_scale_c;
                }
                int cga_store_atom_col = d_idx % 64 * 2;
                #pragma unroll
                for (int cga_store_head = 0; cga_store_head < TILE_Q; cga_store_head++) {
                    int cga_store_atom_row = cga_store_head * 2 + d_idx / 64;
                    {
                        __nv_bfloat16 _bval_0 = __float2bfloat16_rn(cga_partial_hi[cga_store_head]);
                        uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                        uint32_t _addr_0 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(cga_store_atom_row * 128 + cga_store_atom_col ^ (cga_store_atom_row * 128 + cga_store_atom_col >> 7 & 7) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                    }
                }
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                int cga_copy_row = corr_tid / 16;
                int cga_copy_chunk = corr_tid % 16;
                int cga_copy_atom_row = cga_copy_row * 2 + cga_copy_chunk / 8;
                int cga_copy_atom_col = cga_copy_chunk % 8 * 16;
                int cga_copy_smem_offset = cga_copy_atom_row * 128 + (cga_copy_atom_col ^ (cga_copy_atom_row & 7) << 4);
                int cga_owner = cga_copy_row / 4;
                int cga_owner_row = cga_copy_row % 4;
                uint32_t _mapa_0;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_0) : "r"(smem_cga_o_addr), "r"(cga_owner));
                uint32_t _mapa_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_1) : "r"(low_q1_cga_reduction_addr), "r"(cga_owner));
                unsigned int cga_copy_vec_hi[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&cga_copy_vec_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&cga_copy_vec_hi[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&cga_copy_vec_hi[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&cga_copy_vec_hi[(0) + 3]))
                    : "r"(smem_p_addr + (unsigned int)cga_copy_smem_offset));
                int cga_o_offset_hi = (cta_rank * 4 + cga_owner_row) * HEAD_DIM * 2 + cga_copy_chunk * 16;
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + (unsigned int)cga_o_offset_hi), "r"(cga_copy_vec_hi[0]), "r"(cga_copy_vec_hi[1]), "r"(cga_copy_vec_hi[2]), "r"(cga_copy_vec_hi[3]), "r"(_mapa_1) : "memory");
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                #pragma unroll
                for (int cga_store_head_lo = 0; cga_store_head_lo < TILE_Q; cga_store_head_lo++) {
                    int cga_store_atom_row_lo = cga_store_head_lo * 2 + d_idx / 64;
                    {
                        __nv_bfloat16 _bval_1 = __float2bfloat16_rn(cga_partial_lo[cga_store_head_lo]);
                        uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                        uint32_t _addr_1 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(cga_store_atom_row_lo * 128 + cga_store_atom_col ^ (cga_store_atom_row_lo * 128 + cga_store_atom_col >> 7 & 7) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
                    }
                }
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                unsigned int cga_copy_vec_lo[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&cga_copy_vec_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&cga_copy_vec_lo[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&cga_copy_vec_lo[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&cga_copy_vec_lo[(0) + 3]))
                    : "r"(smem_p_addr + (unsigned int)cga_copy_smem_offset));
                int cga_o_offset_lo = cga_o_offset_hi + HEAD_DIM_HALF * 2;
                asm volatile(
                    "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                    :: "r"(_mapa_0 + (unsigned int)cga_o_offset_lo), "r"(cga_copy_vec_lo[0]), "r"(cga_copy_vec_lo[1]), "r"(cga_copy_vec_lo[2]), "r"(cga_copy_vec_lo[3]), "r"(_mapa_1) : "memory");
                #pragma unroll
                for (int cga_pair = 0; cga_pair < TILE_Q / 2; cga_pair++) {
                    if (corr_tid == cga_pair) {
                        int cga_stats_owner = cga_pair / 2;
                        int cga_stats_local_pair = cga_pair % 2;
                        uint32_t _mapa_2;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_2) : "r"(smem_cga_stats_addr), "r"(cga_stats_owner));
                        uint32_t _mapa_3;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_3) : "r"(low_q1_cga_reduction_addr), "r"(cga_stats_owner));
                        int cga_active_o_bytes = 8 * HEAD_DIM * 2;
                        int cga_stats_offset = cga_active_o_bytes + (cta_rank * 4 + cga_stats_local_pair * 2) * 2 * 4;
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_mapa_2 + (unsigned int)cga_stats_offset), "r"(__float_as_uint(total_max[cga_pair * 2])), "r"(__float_as_uint(total_sum[cga_pair * 2])), "r"(__float_as_uint(total_max[cga_pair * 2 + 1])), "r"(__float_as_uint(total_sum[cga_pair * 2 + 1])), "r"(_mapa_3) : "memory");
                    }
                }
                if (cta_rank < 2) {
                    if (warp == 4) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(low_q1_cga_reduction_addr), "r"((uint32_t)(29120)) : "memory");
                        }
                    }
                    mbarrier_wait_cluster_hint(low_q1_cga_reduction_addr, _phase_low_q1_cga_reduction_0, 10000000);
                    _phase_low_q1_cga_reduction_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int cga_owner_head_begin = cta_rank * 4;
                    int cga_stats_base = 8 * HEAD_DIM / 2;
                    int cga_num_n_blocks = (seqlen_kv_c + BLOCK_N - 1) / BLOCK_N;
                    if (cga_num_n_blocks < 1) {
                        cga_num_n_blocks = 1;
                    }
                    int cga_valid_sources = (cga_num_n_blocks + 1) / 2;
                    if (cga_valid_sources > 2) {
                        cga_valid_sources = 2;
                    }
                    #pragma unroll
                    for (int cga_local_head = 0; cga_local_head < 4; cga_local_head++) {
                        float cga_global_max = -CAKE_INF;
                        #pragma unroll
                        for (int cga_source = 0; cga_source < 2; cga_source++) {
                            if (cga_valid_sources > cga_source) {
                                float cga_source_max = smem_cga_stats[cga_stats_base + (cga_source * 4 + cga_local_head) * 2];
                                float _max_11 = max_noftz(cga_global_max, cga_source_max);
                                cga_global_max = _max_11;
                            }
                        }
                        float cga_global_sum = 0.0f;
                        float cga_global_o_hi = 0.0f;
                        float cga_global_o_lo = 0.0f;
                        #pragma unroll
                        for (int cga_source_1 = 0; cga_source_1 < 2; cga_source_1++) {
                            if (cga_valid_sources > cga_source_1) {
                                int cga_source_row = cga_source_1 * 4 + cga_local_head;
                                float cga_source_max_2 = smem_cga_stats[cga_stats_base + cga_source_row * 2];
                                float cga_source_sum = smem_cga_stats[cga_stats_base + cga_source_row * 2 + 1];
                                if (cga_source_max_2 != -CAKE_INF) {
                                    float _exp2_2 = approx_exp2(bmm1_scale_log2_c * (cga_source_max_2 - cga_global_max));
                                    float cga_weight = _exp2_2;
                                    __nv_bfloat16 cga_source_o_hi_b = smem_cga_o[cga_source_row * HEAD_DIM + d_idx];
                                    __nv_bfloat16 cga_source_o_lo_b = smem_cga_o[cga_source_row * HEAD_DIM + HEAD_DIM_HALF + d_idx];
                                    float _fma_1 = __fmaf_rn(cga_source_sum, cga_weight, cga_global_sum);
                                    cga_global_sum = _fma_1;
                                    float _cvt_f32_0 = __bfloat162float(cga_source_o_hi_b);
                                    float _fma_2 = __fmaf_rn(_cvt_f32_0, cga_weight, cga_global_o_hi);
                                    cga_global_o_hi = _fma_2;
                                    float _cvt_f32_1 = __bfloat162float(cga_source_o_lo_b);
                                    float _fma_3 = __fmaf_rn(_cvt_f32_1, cga_weight, cga_global_o_lo);
                                    cga_global_o_lo = _fma_3;
                                }
                            }
                        }
                        float cga_final_hi = 0.0f;
                        float cga_final_lo = 0.0f;
                        if (cga_global_sum > 0.0f) {
                            float _rcp_0 = approx_rcp(cga_global_sum);
                            float cga_inv_sum = _rcp_0;
                            cga_final_hi = cga_global_o_hi * cga_inv_sum;
                            cga_final_lo = cga_global_o_lo * cga_inv_sum;
                        }
                        int cga_q_head = kv_head_idx_c * group_ratio_rt + cga_owner_head_begin + cga_local_head;
                        if (cga_q_head < num_q_heads) {
                            int cga_output_row = (batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + cga_q_head;
                            int cga_out_hi = cga_output_row * HEAD_DIM + d_idx;
                            *(reinterpret_cast<__nv_bfloat16*>(O + cga_out_hi) + (0)) = __float2bfloat16_rn(cga_final_hi);
                            *(reinterpret_cast<__nv_bfloat16*>(O + (cga_out_hi + HEAD_DIM_HALF)) + (0)) = __float2bfloat16_rn(cga_final_lo);
                        }
                    }
                }
                mbarrier_arrive(o_empty_addr);
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
            q_row_idx_2 = 0;
            int schedule_batch_idx_2 = blockIdx.z;
            split_idx_2 = cta_rank;
            part_count_2 = 2;
            batch_idx_2 = schedule_batch_idx_2;
            {
                batch_idx_2 = request_order[schedule_batch_idx_2];
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
                base_pairs_2 = total_pairs_2 / 2;
                extra_pairs_2 = total_pairs_2 % 2;
                int num_pairs_2 = base_pairs_2;
                int split_start_pair_2 = extra_pairs_2 * (base_pairs_2 + 1) + (split_idx_m - extra_pairs_2) * base_pairs_2;
                if (split_idx_m < extra_pairs_2) {
                    num_pairs_2 = base_pairs_2 + 1;
                    split_start_pair_2 = split_idx_m * (base_pairs_2 + 1);
                }
                if (num_pairs_2 < 1) {
                    num_pairs_2 = 1;
                    split_start_pair_2 = split_idx_m;
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
            q_row_idx_3 = 0;
            int schedule_batch_idx_3 = blockIdx.z;
            split_idx_3 = cta_rank;
            part_count_3 = 2;
            batch_idx_3 = schedule_batch_idx_3;
            {
                batch_idx_3 = request_order[schedule_batch_idx_3];
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
                base_pairs_3 = total_pairs_3 / 2;
                extra_pairs_3 = total_pairs_3 % 2;
                int num_pairs_3 = base_pairs_3;
                int split_start_pair_3 = extra_pairs_3 * (base_pairs_3 + 1) + (split_idx_p - extra_pairs_3) * base_pairs_3;
                if (split_idx_p < extra_pairs_3) {
                    num_pairs_3 = base_pairs_3 + 1;
                    split_start_pair_3 = split_idx_p * (base_pairs_3 + 1);
                }
                if (num_pairs_3 < 1) {
                    num_pairs_3 = 1;
                    split_start_pair_3 = split_idx_p;
                }
                int pages_per_seq_p = (seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE;
                int max_page_p = pages_per_seq_p - 1;
                int pt_base_p = batch_idx_p * max_pages_per_seq;
                int pt_base_v_p = pt_base_p + page_table_v_offset;
                {
                    {
                        int paired_n_blocks_p = num_pairs_3 * 2 - 2;
                        #pragma unroll 1
                        for (int n_pair_p = 0; n_pair_p < paired_n_blocks_p; n_pair_p += 2) {
                            int n_block_p = split_start_pair_3 * 2 + n_pair_p;
                            int logical_page_base_p = n_block_p * 2;
                            int page_ids_k_p[4];
                            int page_ids_v_p[4];
                            if (elect_sync()) {
                                {
                                    int4 _iv4 = *reinterpret_cast<const int4*>(page_table + pt_base_p + logical_page_base_p);
                                    page_ids_k_p[0 + 0] = _iv4.x;
                                    page_ids_k_p[0 + 1] = _iv4.y;
                                    page_ids_k_p[0 + 2] = _iv4.z;
                                    page_ids_k_p[0 + 3] = _iv4.w;
                                }
                                {
                                    int4 _iv4 = *reinterpret_cast<const int4*>(page_table + pt_base_v_p + logical_page_base_p);
                                    page_ids_v_p[0 + 0] = _iv4.x;
                                    page_ids_v_p[0 + 1] = _iv4.y;
                                    page_ids_v_p[0 + 2] = _iv4.z;
                                    page_ids_v_p[0 + 3] = _iv4.w;
                                }
                            }
                            #pragma unroll
                            for (int pair_block_p = 0; pair_block_p < 2; pair_block_p++) {
                                mbarrier_wait(page_offsets_empty_addr + (page_prod_stage) * 8, page_prod_phase);
                                if (elect_sync()) {
                                    int page_smem_base_p = page_prod_stage * 4;
                                    int page_vec_base_p = pair_block_p * 2;
                                    smem_page_offsets[page_smem_base_p] = page_ids_k_p[page_vec_base_p];
                                    smem_page_offsets[page_smem_base_p + 1] = page_ids_k_p[page_vec_base_p + 1];
                                    smem_page_offsets[page_smem_base_p + 2] = page_ids_v_p[page_vec_base_p];
                                    smem_page_offsets[page_smem_base_p + 3] = page_ids_v_p[page_vec_base_p + 1];
                                }
                                mbarrier_arrive(page_offsets_full_addr + (page_prod_stage) * 8);
                                page_prod_stage += 1;
                                if (page_prod_stage == 6) { page_prod_stage = 0; page_prod_phase ^= 1; }
                            }
                        }
                        #pragma unroll 1
                        for (int n_p = paired_n_blocks_p; n_p < num_pairs_3 * 2; n_p++) {
                            mbarrier_wait(page_offsets_empty_addr + (page_prod_stage) * 8, page_prod_phase);
                            if (elect_sync()) {
                                #pragma unroll
                                for (int page_in_block_p = 0; page_in_block_p < 2; page_in_block_p++) {
                                    int logical_page_p = (split_start_pair_3 * 2 + n_p) * 2 + page_in_block_p;
                                    int clamped_page_p = ((logical_page_p > max_page_p) ? max_page_p : logical_page_p);
                                    smem_page_offsets[page_prod_stage * 4 + page_in_block_p] = page_table[pt_base_p + clamped_page_p];
                                    smem_page_offsets[page_prod_stage * 4 + 2 + page_in_block_p] = page_table[pt_base_v_p + clamped_page_p];
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
            q_row_idx_4 = 0;
            int schedule_batch_idx_4 = blockIdx.z;
            split_idx_4 = cta_rank;
            part_count_4 = 2;
            batch_idx_4 = schedule_batch_idx_4;
            {
                batch_idx_4 = request_order[schedule_batch_idx_4];
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
                base_pairs_4 = total_pairs_4 / 2;
                extra_pairs_4 = total_pairs_4 % 2;
                int num_pairs_4 = base_pairs_4;
                int split_start_pair_4 = extra_pairs_4 * (base_pairs_4 + 1) + (split_idx_l - extra_pairs_4) * base_pairs_4;
                if (split_idx_l < extra_pairs_4) {
                    num_pairs_4 = base_pairs_4 + 1;
                    split_start_pair_4 = split_idx_l * (base_pairs_4 + 1);
                }
                if (num_pairs_4 < 1) {
                    num_pairs_4 = 1;
                    split_start_pair_4 = split_idx_l;
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
            q_row_idx_5 = 0;
            int schedule_batch_idx_5 = blockIdx.z;
            split_idx_5 = cta_rank;
            part_count_5 = 2;
            batch_idx_5 = schedule_batch_idx_5;
            {
                batch_idx_5 = request_order[schedule_batch_idx_5];
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
                base_pairs_5 = total_pairs_5 / 2;
                extra_pairs_5 = total_pairs_5 % 2;
                int num_pairs_5 = base_pairs_5;
                int split_start_pair_5 = extra_pairs_5 * (base_pairs_5 + 1) + (split_idx_t - extra_pairs_5) * base_pairs_5;
                if (split_idx_t < extra_pairs_5) {
                    num_pairs_5 = base_pairs_5 + 1;
                    split_start_pair_5 = split_idx_t * (base_pairs_5 + 1);
                }
                if (num_pairs_5 < 1) {
                    num_pairs_5 = 1;
                    split_start_pair_5 = split_idx_t;
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
