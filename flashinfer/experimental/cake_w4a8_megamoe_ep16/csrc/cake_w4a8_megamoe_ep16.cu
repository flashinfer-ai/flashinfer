/*
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 72
#define TMEM_ACC_OFFSET 0
#define TMEM_SFW_OFFSET 68
#define TMEM_SFX_OFFSET 64
#define NUM_TASK_PIPE_STAGES 2
#define NUM_PIPE_STAGES 10
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_INPUT_ROW_OFF 3072
#define SMEM_INPUT_ROW_STAGE_BYTES 6144
#define SMEM_INPUT_ROW_STRIDE 6144
#define SMEM_COUNTS_OFF 1024
#define SMEM_COUNTS_STAGE_BYTES 2048
#define SMEM_COUNTS_STRIDE 2048
#define SMEM_PULL_STORAGE_OFF 3072
#define SMEM_PULL_STORAGE_STAGE_BYTES 12288
#define SMEM_PULL_STORAGE_STRIDE 12288
#define SMEM_TASKS_OFF 218624
#define SMEM_TASKS_STAGE_BYTES 64
#define SMEM_TASKS_STRIDE 64
#define SMEM_W_OFF 44032
#define SMEM_W_STAGE_BYTES 16384
#define SMEM_W_STRIDE 16384
#define SMEM_X_OFF 23552
#define SMEM_X_STAGE_BYTES 2048
#define SMEM_X_STRIDE 2048
#define SMEM_SW_OFF 212992
#define SMEM_SW_STAGE_BYTES 512
#define SMEM_SW_STRIDE 512
#define SMEM_SX_OFF 207872
#define SMEM_SX_STAGE_BYTES 512
#define SMEM_SX_STRIDE 512
#define SMEM_SCRATCH_OFF 218112
#define SMEM_SCRATCH_STAGE_BYTES 512
#define SMEM_SCRATCH_STRIDE 512
#define SMEM_STAGING_OFF 15360
#define SMEM_STAGING_STAGE_BYTES 8192
#define SMEM_STAGING_STRIDE 8192
#define SMEM_COMBINE_STORAGE_OFF 1024
#define SMEM_COMBINE_STORAGE_STAGE_BYTES 147456
#define SMEM_COMBINE_STORAGE_STRIDE 147456
#define SMEM_TOTAL 218752
#define THREADS 512

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


__device__ __forceinline__ void tcgen05_mma_mxf8_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ss_step_cg2(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::mxf8f6f4 [%2], da, db, %3, "
        "{m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}


__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
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


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo128(int addr) {
    const int SBO = 128;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}


__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x4_wait(float* dst, int addr) {
    tmem_ld_x4(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(512) __cluster_dims__(2,1,1) void
kernel_cake_w4a8_megamoe_ep16(__nv_bfloat16* __restrict__ caller_x, float* __restrict__ caller_rw, unsigned int* __restrict__ staged_x, uint8_t* __restrict__ staged_sf, long long* __restrict__ staged_ids, float* __restrict__ staged_rw, const __grid_constant__ CUtensorMap W1, const __grid_constant__ CUtensorMap W2, const __grid_constant__ CUtensorMap X1, const __grid_constant__ CUtensorMap X2, const __grid_constant__ CUtensorMap SW1, const __grid_constant__ CUtensorMap SW2, const __grid_constant__ CUtensorMap SX1, const __grid_constant__ CUtensorMap SX2, float* __restrict__ RW, const __grid_constant__ CUtensorMap Q, uint8_t* __restrict__ QData, uint8_t* __restrict__ SF, unsigned long long* __restrict__ output_peers, uint8_t* __restrict__ slots, uint8_t* __restrict__ output, unsigned int* __restrict__ epilogue_grid, unsigned int* __restrict__ l1_full, unsigned int* __restrict__ l1_empty, unsigned int* __restrict__ l2_full, unsigned int* __restrict__ l2_empty, long long* __restrict__ ids, unsigned long long* __restrict__ send, unsigned long long* __restrict__ rank_counts, unsigned int* __restrict__ indices, unsigned long long* __restrict__ src_peers, unsigned long long* __restrict__ recv_peers, unsigned long long* __restrict__ sum_peers, unsigned int* __restrict__ grid_counter, unsigned int* __restrict__ signals, unsigned int* __restrict__ status, unsigned long long* __restrict__ signal_peers, unsigned int rank, unsigned int live_tokens, unsigned long long* __restrict__ token_peers, unsigned long long* __restrict__ sf_peers, unsigned long long* __restrict__ weight_peers, uint8_t* __restrict__ XData, unsigned int* __restrict__ XSFData, unsigned int* __restrict__ metadata, unsigned long long* __restrict__ recv, unsigned int* __restrict__ claims, unsigned int pool_blocks, int active_n, int valid_m, int epoch, int fc1_tiles, int fc2_tiles, int fc1_k, int fc2_k)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define pull_addr (mbar_base + 0)
    #define task_full_addr (mbar_base + 32)
    #define task_empty_addr (mbar_base + 48)
    #define full_addr (mbar_base + 64)
    #define empty_addr (mbar_base + 144)
    #define done_addr (mbar_base + 224)
    #define released_addr (mbar_base + 240)
    #define combine_barriers_addr (mbar_base + 256)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 384);

    // Kernel setup ops
    __nv_bfloat16* input_row = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int input_row_addr = smem + 3072;
    unsigned int* counts = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int counts_addr = smem + 1024;
    uint8_t* pull_storage = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int pull_storage_addr = smem + 3072;
    unsigned int* tasks = reinterpret_cast<unsigned int*>(smem_raw + 218624);
    const int tasks_addr = smem + 218624;
    uint8_t* w = reinterpret_cast<uint8_t*>(smem_raw + 44032);
    const int w_addr = smem + 44032;
    uint8_t* x = reinterpret_cast<uint8_t*>(smem_raw + 23552);
    const int x_addr = smem + 23552;
    unsigned int* sw = reinterpret_cast<unsigned int*>(smem_raw + 212992);
    const int sw_addr = smem + 212992;
    unsigned int* sx = reinterpret_cast<unsigned int*>(smem_raw + 207872);
    const int sx_addr = smem + 207872;
    float* scratch = reinterpret_cast<float*>(smem_raw + 218112);
    const int scratch_addr = smem + 218112;
    unsigned int* staging = reinterpret_cast<unsigned int*>(smem_raw + 15360);
    const int staging_addr = smem + 15360;
    unsigned int* combine_storage = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int combine_storage_addr = smem + 1024;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 48 barriers)
    // Mbarriers at smem_raw[0..384)

    if (warp == 1) {
        // pull: 4 barriers, init_count=1
        if (lane == 0) {
            mbarrier_init(smem + 0, 1);
        }
        if (lane == 1) {
            mbarrier_init(smem + 8, 1);
        }
        if (lane == 2) {
            mbarrier_init(smem + 16, 1);
        }
        if (lane == 3) {
            mbarrier_init(smem + 24, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'task_pipe' ---
            // task_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // task_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 48, 512);
            mbarrier_init(smem + 56, 512);
            // --- pipeline 'pipe' ---
            // full: 10 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            mbarrier_init(smem + 96, 4);
            mbarrier_init(smem + 104, 4);
            mbarrier_init(smem + 112, 4);
            mbarrier_init(smem + 120, 4);
            mbarrier_init(smem + 128, 4);
            mbarrier_init(smem + 136, 4);
            // empty: 10 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // --- pipeline 'acc_pipe' ---
            // done: 2 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // released: 2 barriers, init_count=512
            mbarrier_init(smem + 240, 512);
            mbarrier_init(smem + 248, 512);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        // combine_barriers: 16 barriers, init_count=1
        if (lane == 0) {
            mbarrier_init(smem + 256, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 264, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 272, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 280, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 288, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 296, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 304, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 312, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 320, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 328, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 336, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 344, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 352, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 360, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 368, 1);
        }
        if (lane == 0) {
            mbarrier_init(smem + 376, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    // Kernel pre-TMEM ops
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (128 columns, 72 used)
    if (warp == 3) {
        int _tmem_hold = smem + 384;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_sfw = taddr + 68;
    const int tmem_sfx = taddr + 64;

    // ---- Role: dispatch ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // dispatch_main
            #pragma unroll
            for (int token = bid; token < 384; token += num_bids) {
                if ((unsigned int)token < live_tokens) {
                    #pragma unroll
                    for (int i = 0; i < 3; i++) {
                        unsigned int chunk = tid + i * 128;
                        unsigned int byte = chunk * 16;
                        unsigned int swizzled = byte ^ byte >> 3 & 32;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16;"
                            :: "r"(input_row_addr + swizzled), "l"(caller_x + ((unsigned int)(token * 3072) + chunk * 8)));
                        if (i == 1 || i == 2) {
                            asm volatile("cp.async.commit_group;");
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 2;");
                        asm volatile("barrier.sync 0, 128;" ::: "memory");
                        unsigned int chunk_1 = tid + j * 128;
                        float values[16];
                        float amax = 0.0f;
                        if (chunk_1 < 192) {
                            unsigned int byte_1 = chunk_1 * 32;
                            unsigned int swizzled_1 = byte_1 ^ byte_1 >> 3 & 32;
                            __nv_bfloat16 _input_row_reg_0[16];
                            {
                                const __nv_bfloat16* _smem_ptr = reinterpret_cast<const __nv_bfloat16*>(input_row);
                                #pragma unroll
                                for (int _lr = 0; _lr < 16; _lr++)
                                    _input_row_reg_0[_lr] = _smem_ptr[(swizzled_1 / 2) + _lr];
                            }
                            #pragma unroll
                            for (int i_1 = 0; i_1 < 16; i_1++) {
                                float _cvt_f32_0 = __bfloat162float(_input_row_reg_0[i_1]);
                                values[i_1] = _cvt_f32_0;
                                float _fabs_0 = fabsf(values[i_1]);
                                float _max_0 = max_noftz(amax, _fabs_0);
                                amax = _max_0;
                            }
                        }
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
                        float _max_1 = max_noftz(amax, _shfl_xor_0);
                        amax = _max_1;
                        if (chunk_1 < 192) {
                            float scaled = amax * 0.002232142857142857f;
                            unsigned int bits = __as_u32(scaled);
                            unsigned int exponent = (bits >> 23) + (unsigned int)((((bits & 8388607) != 0) ? 1 : 0));
                            if (bits <= 4194304) {
                                exponent = 0;
                            }
                            unsigned int inverse_bits = ((exponent == 0) ? (unsigned int)2139095039 : 254 - exponent << 23);
                            float inverse = __uint_as_float(inverse_bits);
                            unsigned int packed[4];
                            #pragma unroll
                            for (int i_2 = 0; i_2 < 4; i_2++) {
                                uint16_t _e4m3x2_f32_0;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(values[i_2 * 4 + 1] * inverse), "f"(values[i_2 * 4] * inverse));
                                uint16_t _e4m3x2_f32_1;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(values[i_2 * 4 + 3] * inverse), "f"(values[i_2 * 4 + 2] * inverse));
                                packed[i_2] = (unsigned int)_e4m3x2_f32_0 | (unsigned int)_e4m3x2_f32_1 << 16;
                            }
                            reinterpret_cast<int4*>(staged_x + ((unsigned int)(token * 768) + chunk_1 * 4))[0] = reinterpret_cast<int4*>(packed)[0];
                            if (chunk_1 % 2 == 0) {
                                uint8_t byte_scale = exponent;
                                staged_sf[(unsigned int)(token * 96) + chunk_1 / 2] = byte_scale;
                            }
                        }
                    }
                    if (tid < 8) {
                        staged_ids[token * 8 + tid] = ids[token * 8 + tid];
                        staged_rw[token * 8 + tid] = caller_rw[token * 8 + tid];
                    }
                } else if (tid < 8) {
                    staged_ids[token * 8 + tid] = -1;
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
            }
            #pragma unroll
            for (int j_1 = 0; j_1 < 4; j_1++) {
                counts[tid + j_1 * 128] = 0;
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            #pragma unroll
            for (int token_1 = (bid * 4 + tid / 32) * 4; token_1 < live_tokens; token_1 += num_bids * 16) {
                if ((unsigned int)token_1 + lane / 8 < live_tokens) {
                    int expert = (int)ids[(unsigned int)(token_1 * 8) + lane];
                    if (expert >= 0) {
                        unsigned int _atomic_old_0 = atomicAdd_block(&counts[expert], static_cast<unsigned int>(1));
                    }
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            #pragma unroll
            for (int j_2 = 0; j_2 < 4; j_2++) {
                unsigned int expert_1 = tid + j_2 * 128;
                unsigned int _counts_reg_0[1];
                {
                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(counts);
                    #pragma unroll
                    for (int _lr = 0; _lr < 1; _lr++)
                        _counts_reg_0[_lr] = _smem_ptr[(expert_1) + _lr];
                }
                unsigned long long contribution = 4294967296 | (unsigned long long)_counts_reg_0[0];
                unsigned long long _atomic_old_1 = atomicAdd(&send[expert_1], contribution);
                counts[expert_1] = (unsigned int)_atomic_old_1;
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            #pragma unroll
            for (int token_2 = (bid * 4 + tid / 32) * 4; token_2 < live_tokens; token_2 += num_bids * 16) {
                if ((unsigned int)token_2 + lane / 8 < live_tokens) {
                    int expert_2 = (int)ids[(unsigned int)(token_2 * 8) + lane];
                    if (expert_2 >= 0) {
                        unsigned int _atomic_old_2 = atomicAdd_block(&counts[expert_2], static_cast<unsigned int>(1));
                        unsigned int offset = ((unsigned int)(expert_2 % 32 * 16) + rank) * 6144 + _atomic_old_2;
                        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(src_peers[expert_2 / 32])) + (offset)) = (unsigned int)(token_2 * 8) + lane;
                    }
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (tid == 0) {
                unsigned int delta = ((bid == 0) ? 2147483649 - num_bids : 1);
                unsigned int _atomic_old_3;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_3) : "l"(reinterpret_cast<unsigned int*>(grid_counter)), "r"(static_cast<uint32_t>(delta)) : "memory");
                {
                    unsigned int* _gca_p = reinterpret_cast<unsigned int*>(grid_counter) + (0);
                    while (true) {
                        unsigned int _gca_v;
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p) : "memory");
                        if (((_gca_v ^ (unsigned int)(_atomic_old_3)) & 0x80000000u) != 0) break;
                    }
                }
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (bid == 0) {
                #pragma unroll
                for (int j_3 = 0; j_3 < 4; j_3++) {
                    unsigned int expert_3 = tid + j_3 * 128;
                    unsigned long long value = send[expert_3];
                    *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(recv_peers[expert_3 / 32])) + (rank * 32 + expert_3 % 32)) = value & 4294967295;
                    unsigned long long _atomic_old_4;
                    asm volatile("atom.relaxed.sys.global.add.u64 %0, [%1], %2;"
                        : "=l"(_atomic_old_4) : "l"(&reinterpret_cast<unsigned long long*>(sum_peers[expert_3 / 32])[expert_3 % 32]), "l"(static_cast<uint64_t>(value)) : "memory");
                }
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (bid == 0) {
                unsigned int state = status[0] & 3;
                unsigned int phase = state & 1;
                unsigned int sign = state >> 1;
                if (tid < 16) {
                    unsigned int delta_1 = ((sign != 0) ? 4294967295 : 1);
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(signal_peers[tid])) + (phase);
                        unsigned int _gc_one = delta_1;
                        asm volatile("red.release.sys.global.add.u32 [%0], %1;" :: "l"(_gc_p), "r"(_gc_one) : "memory");
                    }
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (tid == 0) {
                    unsigned int _atomic_old_5 = atomicAdd(status, 1);
                    unsigned int target = ((sign != 0) ? 0 : 16);
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(signals) + (phase);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v == (unsigned int)(target)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (tid == 0) {
                unsigned int delta_2 = ((bid == 0) ? 2147483649 - num_bids : 1);
                unsigned int _atomic_old_6;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_6) : "l"(reinterpret_cast<unsigned int*>(grid_counter)), "r"(static_cast<uint32_t>(delta_2)) : "memory");
                {
                    unsigned int* _gca_p = reinterpret_cast<unsigned int*>(grid_counter) + (0);
                    while (true) {
                        unsigned int _gca_v;
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p) : "memory");
                        if (((_gca_v ^ (unsigned int)(_atomic_old_6)) & 0x80000000u) != 0) break;
                    }
                }
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            unsigned long long count = 0;
            while (count >> 32 != 2432) {
                count = reinterpret_cast<volatile unsigned long long*>(recv)[lane];
            }
            unsigned int _warp_redux_u32_0;
            asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"((unsigned int)count));
            __syncwarp();
            int expert_4 = -1;
            unsigned int start = 0;
            unsigned int end = 0;
            unsigned int pool_blocks_0 = 0;
            unsigned int stored = 0;
            unsigned int phase_1 = 0;
            unsigned int warp_1 = tid / 32;
            unsigned int ring_blocks = 1284;
            #pragma unroll
            for (int token_3 = (unsigned int)(bid * 4) + warp_1; token_3 < _warp_redux_u32_0; token_3 += num_bids * 4) {
                int previous = expert_4;
                while (end <= (unsigned int)token_3) {
                    expert_4 = expert_4 + 1;
                    pool_blocks_0 = pool_blocks_0 + (end - start + 32 - 1) / 32;
                    start = end;
                    end = end + (unsigned int)recv[expert_4];
                }
                if (previous != expert_4) {
                    stored = 0;
                    if (lane < 16) {
                        stored = (unsigned int)rank_counts[lane * 32 + (unsigned int)expert_4];
                    }
                }
                unsigned int remaining = stored;
                unsigned int slot = (unsigned int)token_3 - start;
                unsigned int offset_1 = 0;
                unsigned int selected = 0;
                unsigned int source_rank = 0;
                unsigned int source_slot = 0;
                while (selected == 0) {
                    unsigned int active = ((remaining > 0) ? 1 : 0);
                    unsigned int minimum = 4294967295;
                    if (remaining > 0) {
                        minimum = remaining;
                    }
                    unsigned int _warp_redux_u32_1;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(active));
                    unsigned int _warp_redux_u32_2;
                    asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"(minimum));
                    unsigned int round_tokens = _warp_redux_u32_2 * _warp_redux_u32_1;
                    if (slot < round_tokens) {
                        unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, remaining > 0);
                        unsigned int _find_nth_set_0 = __fns(static_cast<unsigned int>(_vote_0), static_cast<unsigned int>(0), static_cast<int>((int)(slot % _warp_redux_u32_1 + 1)));
                        source_rank = _find_nth_set_0;
                        source_slot = offset_1 + slot / _warp_redux_u32_1;
                        selected = 1;
                    } else {
                        slot = slot - round_tokens;
                        offset_1 = offset_1 + _warp_redux_u32_2;
                        unsigned int _min_0 = ((remaining) < (_warp_redux_u32_2) ? (remaining) : (_warp_redux_u32_2));
                        remaining = remaining - _min_0;
                    }
                }
                unsigned int src_slot = indices[((unsigned int)(expert_4 * 16) + source_rank) * 6144 + source_slot];
                unsigned int src_token = src_slot / 8;
                unsigned int pool_token = pool_blocks_0 * 32 + (unsigned int)token_3 - start;
                unsigned int pool_block = pool_token / 32;
                unsigned int target_1 = pool_block / ring_blocks * 80;
                if (target_1 > 0) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(l1_empty) + (pool_block % ring_blocks);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(target_1)) break;
                        }
                    }
                }
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                        :: "r"(pull_storage_addr + warp_1 * 3072), "l"(reinterpret_cast<uint8_t*>(token_peers[source_rank] + (unsigned long long)src_token * 3072)), "r"((uint32_t)(3072)), "r"(pull_addr + (warp_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(pull_addr + (warp_1) * 8, 3072);
                }
                __syncwarp();
                unsigned int* remote_sf = reinterpret_cast<unsigned int*>(sf_peers[source_rank] + (unsigned long long)src_token * 96);
                unsigned int sf_row = pool_block % ring_blocks * 128 + ((unsigned int)token_3 - start) % 32 * 4;
                if (lane < 24) {
                    XSFData[lane * 657408 + sf_row] = remote_sf[lane];
                }
                __syncwarp();
                if (elect_sync()) {
                    float* remote_weight = reinterpret_cast<float*>(weight_peers[source_rank]);
                    RW[pool_token % 41088] = remote_weight[src_slot];
                    metadata[pool_token * 3] = source_rank;
                    metadata[pool_token * 3 + 1] = src_token;
                    metadata[pool_token * 3 + 2] = src_slot % 8;
                    mbarrier_wait_relaxed(pull_addr + (warp_1) * 8, phase_1);
                    phase_1 = phase_1 ^ 1;
                    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;"
                        :: "l"(XData + (pool_token % 41088 * 3072)), "r"(pull_storage_addr + warp_1 * 3072), "r"((uint32_t)(3072)), "l"(0x1000000000000000ULL) : "memory");
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                    unsigned int increment = 1;
                    if ((unsigned int)token_3 == end - 1) {
                        increment = 32 - ((unsigned int)token_3 - start) % 32;
                    }
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(l1_full) + (pool_block % ring_blocks);
                        unsigned int _gc_one = increment;
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(_gc_p), "r"(_gc_one) : "memory");
                    }
                }
                __syncwarp();
            }
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            unsigned int* l1f = reinterpret_cast<unsigned int*>(l1_full);
            unsigned int* l1e = reinterpret_cast<unsigned int*>(l1_empty);
            unsigned int* l2f = reinterpret_cast<unsigned int*>(l2_full);
            unsigned int* l2e = reinterpret_cast<unsigned int*>(l2_empty);
            if (bid == 0) {
                #pragma unroll
                for (int expert_5 = tid; expert_5 < 512; expert_5 += 128) {
                    send[expert_5] = 0;
                }
                if (warp == 0) {
                    if (elect_sync()) {
                        claims[0] = 0;
                        claims[1] = 0;
                    }
                }
                __syncwarp();
            } else {
                #pragma unroll
                for (int expert_6 = bid - 1; expert_6 < 32; expert_6 += num_bids - 1) {
                    unsigned int tokens = (unsigned int)recv[expert_6];
                    unsigned int blocks = (tokens + 32 - 1) / 32;
                    unsigned int prior_blocks = 0;
                    if (lane < (unsigned int)expert_6) {
                        prior_blocks = ((unsigned int)count + 32 - 1) / 32;
                    }
                    unsigned int _warp_redux_u32_3;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_3) : "r"(prior_blocks));
                    unsigned int offset_2 = _warp_redux_u32_3;
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    if (warp == 0) {
                        recv[expert_6] = 0;
                    }
                    #pragma unroll
                    for (int peer = tid; peer < 16; peer += 128) {
                        rank_counts[peer * 32 + expert_6] = 0;
                    }
                    __syncwarp();
                    #pragma unroll
                    for (int block = tid; block < blocks; block += 128) {
                        unsigned int physical = (offset_2 + (unsigned int)block) % 1284;
                        l1f[physical] = 0;
                        l1e[physical] = 0;
                        l2f[physical] = 0;
                        l2e[physical] = 0;
                    }
                    __syncwarp();
                }
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (tid == 0) {
                unsigned int delta_3 = ((bid == 0) ? 2147483649 - num_bids : 1);
                unsigned int _atomic_old_7;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_7) : "l"(reinterpret_cast<unsigned int*>(grid_counter)), "r"(static_cast<uint32_t>(delta_3)) : "memory");
                {
                    unsigned int* _gca_p = reinterpret_cast<unsigned int*>(grid_counter) + (0);
                    while (true) {
                        unsigned int _gca_v;
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p) : "memory");
                        if (((_gca_v ^ (unsigned int)(_atomic_old_7)) & 0x80000000u) != 0) break;
                    }
                }
            }
            asm volatile("barrier.sync 0, 128;" ::: "memory");
            if (bid == 0) {
                unsigned int state_1 = status[0] & 3;
                unsigned int phase_0 = state_1 & 1;
                unsigned int sign_1 = state_1 >> 1;
                if (tid < 16) {
                    unsigned int delta_4 = ((sign_1 != 0) ? 4294967295 : 1);
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(signal_peers[tid])) + (phase_0);
                        unsigned int _gc_one = delta_4;
                        asm volatile("red.release.sys.global.add.u32 [%0], %1;" :: "l"(_gc_p), "r"(_gc_one) : "memory");
                    }
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (tid == 0) {
                    unsigned int _atomic_old_8 = atomicAdd(status, 1);
                    unsigned int target_2 = ((sign_1 != 0) ? 0 : 16);
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(signals) + (phase_0);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v == (unsigned int)(target_2)) break;
                        }
                    }
                }
            }
        }
    }
    // ---- Role: epi ----
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // epi_main
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            unsigned int e_stage = 0;
            unsigned int task_stage = 0;
            int running = 1;
            unsigned int _phase_task_full = 0;
            unsigned int _phase_done = 0;
            while (running != 0) {
                mbarrier_wait(task_full_addr + (task_stage) * 8, _phase_task_full);
                unsigned int _tasks_reg_3[8];
                {
                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(tasks);
                    #pragma unroll
                    for (int _lr = 0; _lr < 8; _lr++)
                        _tasks_reg_3[_lr] = _smem_ptr[(task_stage * 8) + _lr];
                }
                if (_tasks_reg_3[0] == 0) {
                    running = 0;
                } else {
                    unsigned int phase_2 = _tasks_reg_3[0] - 1;
                    unsigned int tile = _tasks_reg_3[3];
                    unsigned int expert_7 = _tasks_reg_3[1];
                    unsigned int logical_block = _tasks_reg_3[4];
                    unsigned int pool_block_1 = logical_block % 1284;
                    unsigned int ring_epoch = logical_block / 1284;
                    unsigned int task_valid = _tasks_reg_3[5];
                    unsigned int task_n = (task_valid + 15) / 16 * 16;
                    unsigned int k_count = _tasks_reg_3[7] / 128;
                    mbarrier_wait(done_addr + (e_stage) * 8, _phase_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((task_empty_addr + (task_stage) * 8) & 0xFEFFFFFF) : "memory");
                    if (phase_2 == 0) {
                        {
                            unsigned int* _gca_p = reinterpret_cast<unsigned int*>(l2_empty) + (pool_block_1);
                            while (true) {
                                unsigned int _gca_v;
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                if (_gca_v == (unsigned int)(ring_epoch * 24)) break;
                            }
                        }
                        int warp_0 = warp % 4;
                        int wg = (warp - 8) / 4;
                        if (task_valid <= (unsigned int)(wg * 16)) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((released_addr + (e_stage) * 8) & 0xFEFFFFFF) : "memory");
                        } else {
                            float values_1[8];
                            float maxima[4];
                            #pragma unroll
                            for (int atom = 0; atom < 2; atom++) {
                                int token_4 = (unsigned int)(wg * 16 + atom * 8) + lane % 4 * 2;
                                int address = taddr + e_stage * 32 + (unsigned int)(wg * 16) + (unsigned int)(atom * 8);
                                float _tmem_load_0[4];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                    " {%0, %1, %2, %3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                                    : "r"(address));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                float _tmem_load_1[4];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                    " {%0, %1, %2, %3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                                    : "r"(address + 1048576));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                if (atom == 1) {
                                    asm volatile("tcgen05.fence::before_thread_sync;");
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((released_addr + (e_stage) * 8) & 0xFEFFFFFF) : "memory");
                                }
                                float weight0 = RW[pool_block_1 * 32 + (unsigned int)token_4];
                                float weight1 = RW[pool_block_1 * 32 + (unsigned int)token_4 + 1];
                                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_tmem_load_0[0]);
                                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_0);
                                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_tmem_load_0[1]);
                                float _cvt_f32_2 = __bfloat162float(_cvt_bf16_1);
                                float2 _f2_0 = make_float2(_cvt_f32_1, _cvt_f32_2);
                                __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_tmem_load_0[2]);
                                float _cvt_f32_3 = __bfloat162float(_cvt_bf16_2);
                                __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_tmem_load_0[3]);
                                float _cvt_f32_4 = __bfloat162float(_cvt_bf16_3);
                                float2 _f2_1 = make_float2(_cvt_f32_3, _cvt_f32_4);
                                float _expf_0 = __expf(-_f2_0.x);
                                float _expf_1 = __expf(-_f2_0.y);
                                float2 _f2_2 = make_float2(_expf_0, _expf_1);
                                float2 _f2_3 = make_float2(1.0f, 1.0f);
                                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                                #error "Packed FP32x2 arithmetic requires SM100 or newer"
                                #endif
                                float2 _packed_add_f32x2_0;
                                {
                                    float2 _packed_f32x2_0_0 = _f2_3;
                                    float2 _packed_f32x2_0_1 = _f2_2;
                                    asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_0)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_0_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_0_1)));
                                }
                                float _rcp_0 = approx_rcp(_packed_add_f32x2_0.x);
                                float _rcp_1 = approx_rcp(_packed_add_f32x2_0.y);
                                float2 _f2_4 = make_float2(_rcp_0, _rcp_1);
                                float2 _mul_f32x2_0;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_4));
                                float2 _mul_f32x2_1;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_mul_f32x2_0), "l"(*(const unsigned long long*)&_f2_1));
                                float2 _f2_5 = make_float2(weight0, weight1);
                                float2 _mul_f32x2_2;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_mul_f32x2_1), "l"(*(const unsigned long long*)&_f2_5));
                                __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(_tmem_load_1[0]);
                                float _cvt_f32_5 = __bfloat162float(_cvt_bf16_4);
                                __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(_tmem_load_1[1]);
                                float _cvt_f32_6 = __bfloat162float(_cvt_bf16_5);
                                float2 _f2_6 = make_float2(_cvt_f32_5, _cvt_f32_6);
                                __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(_tmem_load_1[2]);
                                float _cvt_f32_7 = __bfloat162float(_cvt_bf16_6);
                                __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(_tmem_load_1[3]);
                                float _cvt_f32_8 = __bfloat162float(_cvt_bf16_7);
                                float2 _f2_7 = make_float2(_cvt_f32_7, _cvt_f32_8);
                                float _expf_2 = __expf(-_f2_6.x);
                                float _expf_3 = __expf(-_f2_6.y);
                                float2 _f2_8 = make_float2(_expf_2, _expf_3);
                                float2 _f2_9 = make_float2(1.0f, 1.0f);
                                #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                                #error "Packed FP32x2 arithmetic requires SM100 or newer"
                                #endif
                                float2 _packed_add_f32x2_1;
                                {
                                    float2 _packed_f32x2_1_0 = _f2_9;
                                    float2 _packed_f32x2_1_1 = _f2_8;
                                    asm("add.f32x2 %0, %1, %2;" : "=l"(reinterpret_cast<uint64_t&>(_packed_add_f32x2_1)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_1_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_1_1)));
                                }
                                float _rcp_2 = approx_rcp(_packed_add_f32x2_1.x);
                                float _rcp_3 = approx_rcp(_packed_add_f32x2_1.y);
                                float2 _f2_10 = make_float2(_rcp_2, _rcp_3);
                                float2 _mul_f32x2_3;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_f2_6), "l"(*(const unsigned long long*)&_f2_10));
                                float2 _mul_f32x2_4;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_mul_f32x2_3), "l"(*(const unsigned long long*)&_f2_7));
                                float2 _f2_11 = make_float2(weight0, weight1);
                                float2 _mul_f32x2_5;
                                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_mul_f32x2_4), "l"(*(const unsigned long long*)&_f2_11));
                                values_1[atom * 4] = _mul_f32x2_2.x;
                                values_1[atom * 4 + 1] = _mul_f32x2_2.y;
                                values_1[atom * 4 + 2] = _mul_f32x2_5.x;
                                values_1[atom * 4 + 3] = _mul_f32x2_5.y;
                                float _fabs_1 = fabsf(values_1[atom * 4]);
                                float _fabs_2 = fabsf(values_1[atom * 4 + 2]);
                                float _max_3 = max_noftz(_fabs_1, _fabs_2);
                                float a0 = _max_3;
                                float _fabs_3 = fabsf(values_1[atom * 4 + 1]);
                                float _fabs_4 = fabsf(values_1[atom * 4 + 3]);
                                float _max_4 = max_noftz(_fabs_3, _fabs_4);
                                float a1 = _max_4;
                                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, a0, 4);
                                float _max_5 = max_noftz(a0, _shfl_xor_1);
                                a0 = _max_5;
                                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, a1, 4);
                                float _max_6 = max_noftz(a1, _shfl_xor_2);
                                a1 = _max_6;
                                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, a0, 8);
                                float _max_7 = max_noftz(a0, _shfl_xor_3);
                                a0 = _max_7;
                                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, a1, 8);
                                float _max_8 = max_noftz(a1, _shfl_xor_4);
                                a1 = _max_8;
                                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, a0, 16);
                                float _max_9 = max_noftz(a0, _shfl_xor_5);
                                a0 = _max_9;
                                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, a1, 16);
                                float _max_10 = max_noftz(a1, _shfl_xor_6);
                                a1 = _max_10;
                                maxima[atom * 2] = a0;
                                maxima[atom * 2 + 1] = a1;
                                if (lane < 4) {
                                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(scratch_addr + (unsigned int)((wg * 4 + warp_0) * 64) + (unsigned int)(atom * 32) + lane * 8), "r"((__as_u32(a0))));
                                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(scratch_addr + (unsigned int)((wg * 4 + warp_0) * 64) + (unsigned int)(atom * 32) + lane * 8 + 4), "r"((__as_u32(a1))));
                                }
                                __syncwarp();
                            }
                            if (warp_0 == 0) {
                                if (elect_sync()) {
                                    asm volatile("cp.async.bulk.wait_group 1;");
                                }
                            }
                            asm volatile("barrier.sync %0, 128;" :: "r"(10 + wg) : "memory");
                            #pragma unroll
                            for (int atom_1 = 0; atom_1 < 2; atom_1++) {
                                int token_5 = (unsigned int)(wg * 16 + atom_1 * 8) + lane % 4 * 2;
                                float _scratch_reg_0[2];
                                {
                                    const float* _smem_ptr = reinterpret_cast<const float*>(scratch);
                                    #pragma unroll
                                    for (int _lr = 0; _lr < 2; _lr++)
                                        _scratch_reg_0[_lr] = _smem_ptr[((unsigned int)((wg * 4 + (warp_0 ^ 1)) * 16 + atom_1 * 8) + lane % 4 * 2) + _lr];
                                }
                                float quantized[4];
                                #pragma unroll
                                for (int t = 0; t < 2; t++) {
                                    float _max_11 = max_noftz(maxima[atom_1 * 2 + t], _scratch_reg_0[t]);
                                    float amax_1 = _max_11;
                                    float scaled_1 = amax_1 * 0.002232142857142857f;
                                    unsigned int bits_1 = __as_u32(scaled_1);
                                    unsigned int exponent_1 = (bits_1 >> 23) + (unsigned int)((((bits_1 & 8388607) != 0) ? 1 : 0));
                                    unsigned int inv_bits = 254 - exponent_1 << 23;
                                    float inverse_1 = __uint_as_float(inv_bits);
                                    if (warp_0 % 2 == 0 && lane < 4) {
                                        uint8_t sf_byte = exponent_1;
                                        SF[tile * pool_blocks * 512 + pool_block_1 * 512 + (unsigned int)((token_5 + t) * 16) + (unsigned int)(cta_rank * 2) + (unsigned int)(warp_0 / 2)] = sf_byte;
                                    }
                                    #pragma unroll
                                    for (int k = 0; k < 2; k++) {
                                        int channel = (unsigned int)(cta_rank * 64 + warp_0 * 16) + lane / 4 + (unsigned int)(k * 8);
                                        float value_1 = values_1[atom_1 * 4 + k * 2 + t];
                                        quantized[k * 2 + t] = value_1 * inverse_1;
                                    }
                                }
                                uint16_t _e4m3x2_f32_2;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_2) : "f"(quantized[1]), "f"(quantized[0]));
                                uint16_t _e4m3x2_f32_3;
                                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_3) : "f"(quantized[3]), "f"(quantized[2]));
                                unsigned int packed_1 = (unsigned int)_e4m3x2_f32_2 | (unsigned int)_e4m3x2_f32_3 << 16;
                                unsigned int addr = staging_addr + (unsigned int)(wg * 2048) + (unsigned int)(atom_1 * 512) + lane * 64 + ((unsigned int)warp_0 ^ lane / 2) * 16;
                                uint32_t _stmatrix_b8_addr_2 = static_cast<uint32_t>(addr);
                                asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n"
                                    :: "r"(_stmatrix_b8_addr_2), "r"(packed_1)
                                    : "memory");
                                __syncwarp();
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("barrier.sync %0, 128;" :: "r"(10 + wg) : "memory");
                            if (warp_0 == 0) {
                                if (elect_sync()) {
                                    tma_store_2d((&Q), tile * 128 + (unsigned int)(cta_rank * 64), pool_block_1 * 32 + (unsigned int)(wg * 16), staging_addr + (unsigned int)(wg * 2048));
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (warp_0 == 0) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                            }
                        }
                        asm volatile("barrier.sync 15, 256;" ::: "memory");
                        if (warp == 8) {
                            if (elect_sync()) {
                                {
                                    unsigned int* _gc_p = reinterpret_cast<unsigned int*>(l2_full) + (pool_block_1);
                                    unsigned int _gc_one = 1;
                                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(_gc_p), "r"(_gc_one) : "memory");
                                }
                                unsigned int _atomic_old_11 = atomicAdd(&reinterpret_cast<unsigned int*>(l1_empty)[pool_block_1], 1);
                            }
                        }
                    } else {
                        if (warp == 8) {
                            if (elect_sync()) {
                                unsigned int _atomic_old_12 = atomicAdd(&reinterpret_cast<unsigned int*>(l2_empty)[pool_block_1], 1);
                            }
                        }
                        int warp_0_1 = warp % 4;
                        int wg_1 = (warp - 8) / 4;
                        if (task_valid <= (unsigned int)(wg_1 * 16)) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((released_addr + (e_stage) * 8) & 0xFEFFFFFF) : "memory");
                        } else {
                            #pragma unroll
                            for (int atom_2 = 0; atom_2 < 2; atom_2++) {
                                int address_1 = taddr + e_stage * 32 + (unsigned int)(wg_1 * 16) + (unsigned int)(atom_2 * 8);
                                float _tmem_load_2[4];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                    " {%0, %1, %2, %3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3]))
                                    : "r"(address_1));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                float _tmem_load_3[4];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                    " {%0, %1, %2, %3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3]))
                                    : "r"(address_1 + 1048576));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                if (atom_2 == 1) {
                                    asm volatile("tcgen05.fence::before_thread_sync;");
                                    asm volatile(
                                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                        :: "r"((released_addr + (e_stage) * 8) & 0xFEFFFFFF) : "memory");
                                }
                                int token_6 = (unsigned int)(wg_1 * 16 + atom_2 * 8) + lane % 4 * 2;
                                int channel_1 = (unsigned int)(cta_rank * 128 + warp_0_1 * 32) + lane / 4;
                                __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_tmem_load_2[0], _tmem_load_2[1]));
                                __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_tmem_load_2[2], _tmem_load_2[3]));
                                __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_tmem_load_3[0], _tmem_load_3[1]));
                                __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_tmem_load_3[2], _tmem_load_3[3]));
                                unsigned int r0 = __as_u32(_bf16x2_0);
                                unsigned int r1 = __as_u32(_bf16x2_1);
                                unsigned int r2 = __as_u32(_bf16x2_2);
                                unsigned int r3 = __as_u32(_bf16x2_3);
                                int row = lane % 8;
                                int col = (unsigned int)(warp_0_1 % 2 * 4) + lane / 8;
                                unsigned int addr_1 = staging_addr + (unsigned int)(wg_1 * 4096) + (unsigned int)(warp_0_1 / 2 * 2048) + (unsigned int)(atom_2 * 1024) + (unsigned int)(row * 128) + (unsigned int)((col ^ row) * 16);
                                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(addr_1);
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&r0)), "r"(*reinterpret_cast<const uint32_t*>(&r1)), "r"(*reinterpret_cast<const uint32_t*>(&r2)), "r"(*reinterpret_cast<const uint32_t*>(&r3))
                                    : "memory");
                            }
                            asm volatile("barrier.sync %0, 128;" :: "r"(10 + wg_1) : "memory");
                            #pragma unroll
                            for (int atom_3 = 0; atom_3 < 2; atom_3++) {
                                int out_row = (unsigned int)(atom_3 * 8 + warp_0_1 * 2) + lane / 16;
                                int row_in_atom = out_row % 8;
                                int offset_3 = ((unsigned int)(wg_1 * 4096) + lane % 16 / 8 * 2048 + (unsigned int)(out_row * 128) + (lane % 8 ^ (unsigned int)row_in_atom) * 16) / 4;
                                if (task_valid > (unsigned int)(wg_1 * 16 + out_row)) {
                                    unsigned int _staging_reg_0[4];
                                    {
                                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(staging);
                                        #pragma unroll
                                        for (int _lr = 0; _lr < 4; _lr++)
                                            _staging_reg_0[_lr] = _smem_ptr[(offset_3) + _lr];
                                    }
                                    unsigned int owner = metadata[(logical_block * 32 + (unsigned int)(wg_1 * 16) + (unsigned int)out_row) * 3];
                                    unsigned int token_7 = metadata[(logical_block * 32 + (unsigned int)(wg_1 * 16) + (unsigned int)out_row) * 3 + 1];
                                    unsigned int slot_1 = metadata[(logical_block * 32 + (unsigned int)(wg_1 * 16) + (unsigned int)out_row) * 3 + 2];
                                    unsigned long long element = ((unsigned long long)slot_1 * 384 + (unsigned long long)token_7) * (unsigned long long)(fc2_tiles * 256) + (unsigned long long)(tile * 256 + (unsigned int)(cta_rank * 128) + lane % 16 * 8);
                                    reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(output_peers[owner] + element * 2))[0] = reinterpret_cast<int4*>(_staging_reg_0)[0];
                                }
                            }
                        }
                        asm volatile("barrier.sync 15, 256;" ::: "memory");
                    }
                    e_stage += 1;
                    if (e_stage == 2) { e_stage = 0; _phase_done ^= 1; }
                }
                task_stage += 1;
                if (task_stage == 2) { task_stage = 0; _phase_task_full ^= 1; }
            }
            if (warp == 8) {
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(128));
            }
            asm volatile("barrier.sync 2, 256;" ::: "memory");
            if (tid == 256) {
                unsigned int delta_5 = ((bid == 0) ? 2147483649 - num_bids : 1);
                unsigned int _atomic_old_13;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_13) : "l"(reinterpret_cast<unsigned int*>(epilogue_grid)), "r"(static_cast<uint32_t>(delta_5)) : "memory");
                {
                    unsigned int* _gca_p = reinterpret_cast<unsigned int*>(epilogue_grid) + (0);
                    while (true) {
                        unsigned int _gca_v;
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p) : "memory");
                        if (((_gca_v ^ (unsigned int)(_atomic_old_13)) & 0x80000000u) != 0) break;
                    }
                }
            }
            asm volatile("barrier.sync 2, 256;" ::: "memory");
            if (bid == 0) {
                unsigned int state_2 = status[0] & 3;
                unsigned int phase_3 = state_2 & 1;
                unsigned int sign_2 = state_2 >> 1;
                if (tid - 256 < 16) {
                    unsigned int delta_6 = ((sign_2 != 0) ? 4294967295 : 1);
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(signal_peers[tid - 256])) + (phase_3);
                        unsigned int _gc_one = delta_6;
                        asm volatile("red.release.sys.global.add.u32 [%0], %1;" :: "l"(_gc_p), "r"(_gc_one) : "memory");
                    }
                }
                asm volatile("barrier.sync 2, 256;" ::: "memory");
                if (tid == 256) {
                    unsigned int _atomic_old_14 = atomicAdd(status, 1);
                    unsigned int target_3 = ((sign_2 != 0) ? 0 : 16);
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(signals) + (phase_3);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v == (unsigned int)(target_3)) break;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 2, 256;" ::: "memory");
            if (tid == 256) {
                unsigned int delta_7 = ((bid == 0) ? 2147483649 - num_bids : 1);
                unsigned int _atomic_old_15;
                asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(_atomic_old_15) : "l"(reinterpret_cast<unsigned int*>(epilogue_grid)), "r"(static_cast<uint32_t>(delta_7)) : "memory");
                {
                    unsigned int* _gca_p = reinterpret_cast<unsigned int*>(epilogue_grid) + (0);
                    while (true) {
                        unsigned int _gca_v;
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p) : "memory");
                        if (((_gca_v ^ (unsigned int)(_atomic_old_15)) & 0x80000000u) != 0) break;
                    }
                }
            }
            asm volatile("barrier.sync 2, 256;" ::: "memory");
            asm volatile("barrier.sync 1, 384;" ::: "memory");
            unsigned int warp_0_2 = warp - 8;
            unsigned int phase_4 = 0;
            unsigned int stage = 0;
            #pragma unroll
            for (int token_8 = (unsigned int)(bid * 8) + warp_0_2; token_8 < live_tokens; token_8 += num_bids * 8) {
                int valid = 0;
                if (lane < 8) {
                    valid = ((ids[(unsigned int)(token_8 * 8) + lane] >= 0) ? 1 : 0);
                }
                unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, valid != 0);
                unsigned int mask = _vote_2;
                float reduced[96];
                #pragma unroll
                for (int i_3 = 0; i_3 < 96; i_3++) {
                    reduced[i_3] = 0.0f;
                }
                int reducing = 0;
                if (mask != 0) {
                    int _ffs_1 = __ffs(mask);
                    unsigned int slot_2 = _ffs_1 - 1;
                    mask = mask ^ 1 << slot_2;
                    if (elect_sync()) {
                        cp_async_bulk_gmem2smem(combine_storage_addr + (warp_0_2 + stage * 8) * 6144, slots + (((unsigned long long)slot_2 * 384 + (unsigned long long)token_8) * 6144), 6144, combine_barriers_addr + (warp_0_2 * 2 + stage) * 8);
                        mbarrier_arrive_expect_tx(combine_barriers_addr + (warp_0_2 * 2 + stage) * 8, 6144);
                    }
                    __syncwarp();
                    reducing = 1;
                }
                while (reducing != 0) {
                    reducing = 0;
                    if (mask != 0) {
                        int _ffs_2 = __ffs(mask);
                        unsigned int slot_3 = _ffs_2 - 1;
                        mask = mask ^ 1 << slot_3;
                        if (elect_sync()) {
                            cp_async_bulk_gmem2smem(combine_storage_addr + (warp_0_2 + (stage ^ 1) * 8) * 6144, slots + (((unsigned long long)slot_3 * 384 + (unsigned long long)token_8) * 6144), 6144, combine_barriers_addr + (warp_0_2 * 2 + (stage ^ 1)) * 8);
                            mbarrier_arrive_expect_tx(combine_barriers_addr + (warp_0_2 * 2 + (stage ^ 1)) * 8, 6144);
                        }
                        __syncwarp();
                        reducing = 1;
                    }
                    mbarrier_wait_relaxed(combine_barriers_addr + (warp_0_2 * 2 + stage) * 8, phase_4);
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 12; j_4++) {
                        unsigned int _combine_storage_reg_0[4];
                        {
                            const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(combine_storage);
                            #pragma unroll
                            for (int _lr = 0; _lr < 4; _lr++)
                                _combine_storage_reg_0[_lr] = _smem_ptr[((warp_0_2 + stage * 8) * 1536 + ((unsigned int)(j_4 * 32) + lane) * 4) + _lr];
                        }
                        #pragma unroll
                        for (int c = 0; c < 4; c++) {
                            float _bf16x2_add_f32_0[2];
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %3;\n\t"
                                "add.rn.f32.bf16 %1, hi, %4;\n\t"
                                "}\n"
                                : "=&f"(_bf16x2_add_f32_0[0]), "=&f"(_bf16x2_add_f32_0[1]) : "r"(_combine_storage_reg_0[c]), "f"(reduced[j_4 * 8 + c * 2]), "f"(reduced[j_4 * 8 + c * 2 + 1]));
                            reduced[j_4 * 8 + c * 2] = _bf16x2_add_f32_0[0];
                            reduced[j_4 * 8 + c * 2 + 1] = _bf16x2_add_f32_0[1];
                        }
                    }
                    __syncwarp();
                    phase_4 = phase_4 ^ stage;
                    stage = stage ^ 1;
                }
                #pragma unroll
                for (int j_5 = 0; j_5 < 12; j_5++) {
                    unsigned int packed_2[4];
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 4; c_1++) {
                        __nv_bfloat162 _bf16x2_4 = __float22bfloat162_rn(make_float2(reduced[j_5 * 8 + c_1 * 2], reduced[j_5 * 8 + c_1 * 2 + 1]));
                        packed_2[c_1] = __as_u32(_bf16x2_4);
                    }
                    if (j_5 == 0) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                        __syncwarp();
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(combine_storage_addr + (warp_0_2 + 16) * 6144 + ((unsigned int)(j_5 * 32) + lane) * 16), "r"(*reinterpret_cast<uint32_t*>(&packed_2[0])), "r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 3])));
                }
                __syncwarp();
                if (elect_sync()) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        void* _cpbulk_dst_4 = reinterpret_cast<void*>(output + ((unsigned long long)token_8 * 6144));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_4), "r"(combine_storage_addr + (warp_0_2 + 16) * 6144), "r"((uint32_t)(6144))
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                __syncwarp();
            }
        }
    }
    // ---- Role: acts ----
    if (warp == 4) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        { // acts_main
            unsigned int a_stage = 0;
            unsigned int task_stage_1 = 0;
            int running_1 = 1;
            unsigned int _phase_task_full_1 = 0;
            unsigned int _phase_empty = 1;
            while (running_1 != 0) {
                mbarrier_wait(task_full_addr + (task_stage_1) * 8, _phase_task_full_1);
                unsigned int _tasks_reg_0[8];
                {
                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(tasks);
                    #pragma unroll
                    for (int _lr = 0; _lr < 8; _lr++)
                        _tasks_reg_0[_lr] = _smem_ptr[(task_stage_1 * 8) + _lr];
                }
                if (_tasks_reg_0[0] == 0) {
                    running_1 = 0;
                } else {
                    unsigned int phase_5 = _tasks_reg_0[0] - 1;
                    unsigned int tile_1 = _tasks_reg_0[3];
                    unsigned int expert_8 = _tasks_reg_0[1];
                    unsigned int logical_block_1 = _tasks_reg_0[4];
                    unsigned int pool_block_2 = logical_block_1 % 1284;
                    unsigned int ring_epoch_1 = logical_block_1 / 1284;
                    unsigned int task_valid_1 = _tasks_reg_0[5];
                    unsigned int task_n_1 = (task_valid_1 + 15) / 16 * 16;
                    unsigned int k_count_1 = _tasks_reg_0[7] / 128;
                    if (phase_5 == 0) {
                        {
                            unsigned int* _gca_p = reinterpret_cast<unsigned int*>(l1_full) + (pool_block_2);
                            while (true) {
                                unsigned int _gca_v;
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                if (_gca_v == (unsigned int)((ring_epoch_1 + 1) * 32)) break;
                            }
                        }
                    } else {
                        {
                            unsigned int* _gca_p = reinterpret_cast<unsigned int*>(l2_full) + (pool_block_2);
                            while (true) {
                                unsigned int _gca_v;
                                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                if (_gca_v == (unsigned int)((ring_epoch_1 + 1) * 80)) break;
                            }
                        }
                    }
                    #pragma unroll
                    for (int k_1 = 0; k_1 < k_count_1; k_1++) {
                        mbarrier_wait(empty_addr + (a_stage) * 8, _phase_empty);
                        if (elect_sync()) {
                            if (phase_5 == 0) {
                                tma_2d_gmem2smem_cta2(x_addr + a_stage * 2048, (&X1), k_1 * 128, pool_block_2 * 32 + (unsigned int)cta_rank * (task_n_1 / 2), ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                                tma_2d_gmem2smem_cta2(sx_addr + a_stage * 512, (&SX1), pool_block_2 * 128, k_1, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                            } else {
                                tma_2d_gmem2smem_cta2(x_addr + a_stage * 2048, (&X2), k_1 * 128, pool_block_2 * 32 + (unsigned int)cta_rank * (task_n_1 / 2), ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                                tma_2d_gmem2smem_cta2(sx_addr + a_stage * 512, (&SX2), pool_block_2 * 128, k_1, ((full_addr + (a_stage) * 8) & 0xFEFFFFFF));
                            }
                            if (cta_rank == 0) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((full_addr + (a_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(5120)) : "memory");
                            } else {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((full_addr + (a_stage) * 8) & 0xFEFFFFFF) : "memory");
                            }
                        }
                        a_stage += 1;
                        if (a_stage == 10) { a_stage = 0; _phase_empty ^= 1; }
                    }
                }
                task_stage_1 += 1;
                if (task_stage_1 == 2) { task_stage_1 = 0; _phase_task_full_1 ^= 1; }
            }
        }
    }
    // ---- Role: weights ----
    if (warp == 5) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        { // weights_main
            unsigned int w_stage = 0;
            unsigned int task_stage_2 = 0;
            int running_2 = 1;
            unsigned int _phase_task_full_2 = 0;
            unsigned int _phase_empty_1 = 1;
            while (running_2 != 0) {
                mbarrier_wait(task_full_addr + (task_stage_2) * 8, _phase_task_full_2);
                unsigned int _tasks_reg_1[8];
                {
                    const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(tasks);
                    #pragma unroll
                    for (int _lr = 0; _lr < 8; _lr++)
                        _tasks_reg_1[_lr] = _smem_ptr[(task_stage_2 * 8) + _lr];
                }
                if (_tasks_reg_1[0] == 0) {
                    running_2 = 0;
                } else {
                    unsigned int phase_6 = _tasks_reg_1[0] - 1;
                    unsigned int tile_2 = _tasks_reg_1[3];
                    unsigned int expert_9 = _tasks_reg_1[1];
                    unsigned int logical_block_2 = _tasks_reg_1[4];
                    unsigned int pool_block_3 = logical_block_2 % 1284;
                    unsigned int ring_epoch_2 = logical_block_2 / 1284;
                    unsigned int task_valid_2 = _tasks_reg_1[5];
                    unsigned int task_n_2 = (task_valid_2 + 15) / 16 * 16;
                    unsigned int k_count_2 = _tasks_reg_1[7] / 128;
                    #pragma unroll
                    for (int k_2 = 0; k_2 < k_count_2; k_2++) {
                        mbarrier_wait(empty_addr + (w_stage) * 8, _phase_empty_1);
                        if (elect_sync()) {
                            if (phase_6 == 0) {
                                tma_2d_gmem2smem_cta2(w_addr + w_stage * 16384, (&W1), k_2 * 128, expert_9 * 10240 + tile_2 * 256 + (unsigned int)(cta_rank * 128), ((full_addr + (w_stage) * 8) & 0xFEFFFFFF));
                                tma_2d_gmem2smem_cta2(sw_addr + w_stage * 512, (&SW1), tile_2 * 256 + (unsigned int)(cta_rank * 128), expert_9 * 24 + (unsigned int)k_2, ((full_addr + (w_stage) * 8) & 0xFEFFFFFF));
                            } else {
                                tma_2d_gmem2smem_cta2(w_addr + w_stage * 16384, (&W2), k_2 * 128, expert_9 * 3072 + tile_2 * 256 + (unsigned int)(cta_rank * 128), ((full_addr + (w_stage) * 8) & 0xFEFFFFFF));
                                tma_2d_gmem2smem_cta2(sw_addr + w_stage * 512, (&SW2), tile_2 * 256 + (unsigned int)(cta_rank * 128), expert_9 * 40 + (unsigned int)k_2, ((full_addr + (w_stage) * 8) & 0xFEFFFFFF));
                            }
                            if (cta_rank == 0) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((full_addr + (w_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(17408)) : "memory");
                            } else {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                    :: "r"((full_addr + (w_stage) * 8) & 0xFEFFFFFF) : "memory");
                            }
                        }
                        w_stage += 1;
                        if (w_stage == 10) { w_stage = 0; _phase_empty_1 ^= 1; }
                    }
                }
                task_stage_2 += 1;
                if (task_stage_2 == 2) { task_stage_2 = 0; _phase_task_full_2 ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 6) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        { // mma_main
            unsigned int m_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int _phase_task_full_3 = 0;
            unsigned int _phase_released = 1;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                unsigned int task_stage_3 = 0;
                int running_3 = 1;
                while (running_3 != 0) {
                    mbarrier_wait(task_full_addr + (task_stage_3) * 8, _phase_task_full_3);
                    unsigned int _tasks_reg_2[8];
                    {
                        const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(tasks);
                        #pragma unroll
                        for (int _lr = 0; _lr < 8; _lr++)
                            _tasks_reg_2[_lr] = _smem_ptr[(task_stage_3 * 8) + _lr];
                    }
                    if (_tasks_reg_2[0] == 0) {
                        running_3 = 0;
                    } else {
                        unsigned int phase_7 = _tasks_reg_2[0] - 1;
                        unsigned int tile_3 = _tasks_reg_2[3];
                        unsigned int expert_10 = _tasks_reg_2[1];
                        unsigned int logical_block_3 = _tasks_reg_2[4];
                        unsigned int pool_block_4 = logical_block_3 % 1284;
                        unsigned int ring_epoch_3 = logical_block_3 / 1284;
                        unsigned int task_valid_3 = _tasks_reg_2[5];
                        unsigned int task_n_3 = (task_valid_3 + 15) / 16 * 16;
                        unsigned int k_count_3 = _tasks_reg_2[7] / 128;
                        mbarrier_wait(released_addr + (acc_stage) * 8, _phase_released);
                        #pragma unroll
                        for (int k_3 = 0; k_3 < k_count_3; k_3++) {
                            mbarrier_wait(full_addr + (m_stage) * 8, _phase_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init = ((k_3 == 0) ? 1 : 0);
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sfw, make_sf_cp_desc_lo_sbo128((((sw_addr) >> 4) + (m_stage) * 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_sfx, make_sf_cp_desc_lo_sbo128((((sx_addr) >> 4) + (m_stage) * 32)));
                                int _mma_a_lo_0 = (((w_addr) >> 4) & 0x3FFF) + (m_stage) * 1024;
                                int _mma_b_lo_0 = (((x_addr) >> 4) & 0x3FFF) + (m_stage) * 128;
                                {
                                    const uint32_t mma_active_n = (uint32_t)(task_n_3);
                                    if (mma_active_n < 16U || mma_active_n > 32U || (mma_active_n & 15U)) { asm volatile("trap;"); }
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf8_bs_cta2((tmem_acc + (acc_stage * 32)), a_desc + 0, b_desc + 0,
                                        ((0x10880280U & ~0x007e0000U) | ((mma_active_n >> 3) << 17)), tmem_sfw, tmem_sfx, ((init) ? 0 : 1));
                                    tcgen05_mma_mxf8_bs_cta2((tmem_acc + (acc_stage * 32)), a_desc + 2, b_desc + 2,
                                        ((0x30880290U & ~0x007e0000U) | ((mma_active_n >> 3) << 17)), tmem_sfw, tmem_sfx, 1);
                                    tcgen05_mma_mxf8_bs_cta2((tmem_acc + (acc_stage * 32)), a_desc + 4, b_desc + 4,
                                        ((0x508802a0U & ~0x007e0000U) | ((mma_active_n >> 3) << 17)), tmem_sfw, tmem_sfx, 1);
                                    tcgen05_mma_mxf8_bs_cta2((tmem_acc + (acc_stage * 32)), a_desc + 6, b_desc + 6,
                                        ((0x708802b0U & ~0x007e0000U) | ((mma_active_n >> 3) << 17)), tmem_sfw, tmem_sfx, 1);
                                }
                            }
                            elect_commit_cg2_multicast(empty_addr + (m_stage) * 8, (uint16_t)(3));
                            m_stage += 1;
                            if (m_stage == 10) { m_stage = 0; _phase_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(done_addr + (acc_stage) * 8, (uint16_t)(3));
                        acc_stage += 1;
                        if (acc_stage == 2) { acc_stage = 0; _phase_released ^= 1; }
                    }
                    task_stage_3 += 1;
                    if (task_stage_3 == 2) { task_stage_3 = 0; _phase_task_full_3 ^= 1; }
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        { // scheduler_main
            unsigned int _phase_task_empty = 1;
            if (cta_rank == 0) {
                unsigned long long count_1 = 0;
                while (count_1 >> 32 != 2432) {
                    count_1 = reinterpret_cast<volatile unsigned long long*>(recv)[lane];
                }
                unsigned int tokens_1 = (unsigned int)count_1;
                __syncwarp();
                unsigned int blocks_1 = (tokens_1 + 32 - 1) / 32;
                unsigned int _warp_redux_u32_4;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_4) : "r"(blocks_1));
                unsigned int total = _warp_redux_u32_4;
                unsigned int clusters = num_bids / 2;
                unsigned int l1_waves = (total * 40 + clusters - 1) / clusters;
                unsigned int first_wave = ((clusters + 11) / 12 * 40 + clusters - 1) / clusters;
                int interleave = (40 + ((int)total - 1) * 28 + (int)clusters - 1) / (int)clusters + 1;
                unsigned int _max_2 = ((first_wave) > (interleave) ? (first_wave) : (interleave));
                unsigned int _min_1 = ((l1_waves) < (_max_2) ? (l1_waves) : (_max_2));
                unsigned int warmup = _min_1;
                unsigned int stage_1 = 0;
                int live = 1;
                while (live != 0) {
                    mbarrier_wait(task_empty_addr + (stage_1) * 8, _phase_task_empty);
                    int selected_1 = 0;
                    unsigned int phase_8 = 0;
                    unsigned int task_idx = 0;
                    while (selected_1 == 0) {
                        if (warmup != 4294967295 && warmup != 0) {
                            warmup = warmup - 1;
                            if (elect_sync()) {
                                unsigned int _atomic_old_9 = atomicAdd(&claims[0], 1);
                                task_idx = _atomic_old_9;
                            }
                            unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, task_idx, 0);
                            task_idx = _shfl_0;
                            if (task_idx >= total * 40) {
                                warmup = 4294967295;
                            } else {
                                phase_8 = 1;
                                selected_1 = 1;
                            }
                        } else {
                            if (elect_sync()) {
                                unsigned int _atomic_old_10 = atomicAdd(&claims[1], 1);
                                task_idx = _atomic_old_10;
                            }
                            unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, task_idx, 0);
                            task_idx = _shfl_1;
                            if (task_idx >= total * 12) {
                                live = 0;
                                selected_1 = 1;
                            } else {
                                if (warmup != 4294967295) {
                                    warmup = 1;
                                }
                                phase_8 = 2;
                                selected_1 = 1;
                            }
                        }
                    }
                    unsigned int expert_11 = 0;
                    unsigned int local_m = 0;
                    unsigned int n_cluster = 0;
                    unsigned int pool_block_5 = 0;
                    unsigned int valid_m_0 = 0;
                    unsigned int shape_n = 0;
                    unsigned int shape_k = 0;
                    if (live != 0) {
                        unsigned int n_clusters = ((phase_8 == 1) ? 40 : 12);
                        pool_block_5 = task_idx / n_clusters;
                        n_cluster = task_idx % n_clusters;
                        unsigned int inclusive = blocks_1;
                        unsigned int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive, 1, 32);
                        if (lane >= 1) {
                            inclusive = inclusive + _shfl_up_0;
                        }
                        unsigned int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, inclusive, 2, 32);
                        if (lane >= 2) {
                            inclusive = inclusive + _shfl_up_1;
                        }
                        unsigned int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, inclusive, 4, 32);
                        if (lane >= 4) {
                            inclusive = inclusive + _shfl_up_2;
                        }
                        unsigned int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, inclusive, 8, 32);
                        if (lane >= 8) {
                            inclusive = inclusive + _shfl_up_3;
                        }
                        unsigned int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, inclusive, 16, 32);
                        if (lane >= 16) {
                            inclusive = inclusive + _shfl_up_4;
                        }
                        unsigned int offset_4 = inclusive - blocks_1;
                        unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, pool_block_5 >= offset_4 && pool_block_5 < inclusive);
                        int _ffs_0 = __ffs(_vote_1);
                        unsigned int owner_lane = _ffs_0 - 1;
                        unsigned int lane_m = pool_block_5 - offset_4;
                        unsigned int _min_2 = ((tokens_1 - lane_m * 32) < (32) ? (tokens_1 - lane_m * 32) : (32));
                        unsigned int lane_valid = _min_2;
                        unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, lane, owner_lane);
                        expert_11 = _shfl_2;
                        unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, lane_m, owner_lane);
                        local_m = _shfl_3;
                        unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, lane_valid, owner_lane);
                        valid_m_0 = _shfl_4;
                        shape_n = ((phase_8 == 1) ? 10240 : 3072);
                        shape_k = ((phase_8 == 1) ? 3072 : 5120);
                        if (phase_8 == 2) {
                            unsigned int issued = 0;
                            while (issued < (pool_block_5 + 1) * 40) {
                                issued = reinterpret_cast<volatile unsigned int*>(claims)[0];
                            }
                        }
                    }
                    if (lane < 2) {
                        uint32_t _mapa_0;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_0) : "r"(task_full_addr + stage_1 * 8), "r"(lane));
                        uint32_t _mapa_1;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_1) : "r"(tasks_addr + stage_1 * 32), "r"(lane));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;"
                            :: "r"(_mapa_0), "r"((uint32_t)(32)) : "memory");
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_mapa_1), "r"(phase_8), "r"(expert_11), "r"(local_m), "r"(n_cluster), "r"(_mapa_0) : "memory");
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_mapa_1 + 16), "r"(pool_block_5), "r"(valid_m_0), "r"(shape_n), "r"(shape_k), "r"(_mapa_0) : "memory");
                    }
                    __syncwarp();
                    stage_1 += 1;
                    if (stage_1 == 2) { stage_1 = 0; _phase_task_empty ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
