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
#define TMEM_NCOLS 40
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O0_OFFSET 8
#define TMEM_TMEM_O1_OFFSET 16
#define TMEM_TMEM_O2_OFFSET 24
#define TMEM_TMEM_O3_OFFSET 32
#define NUM_KV_PIPE_STAGES 4
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 2048
#define SMEM_SMEM_Q_STRIDE 2048
#define SMEM_SMEM_KV_OFF 9216
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 9216
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P_OFF 140288
#define SMEM_SMEM_P_STAGE_BYTES 2048
#define SMEM_SMEM_P_STRIDE 2048
#define SMEM_SMEM_EXCH_OFF 142336
#define SMEM_SMEM_EXCH_STAGE_BYTES 128
#define SMEM_SMEM_EXCH_STRIDE 128
#define SMEM_SMEM_SUM_OFF 142464
#define SMEM_SMEM_SUM_STAGE_BYTES 32
#define SMEM_SMEM_SUM_STRIDE 32
#define SMEM_SMEM_INDICES_OFF 142528
#define SMEM_SMEM_INDICES_STAGE_BYTES 512
#define SMEM_SMEM_INDICES_STRIDE 512
#define SMEM_TOTAL 143104
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_cake_dsv4_bf16_h32_topk128x_early_v47(CakeTensorMap const* tmap_q, CakeTensorMap const* tmap_swa_kv, CakeTensorMap const* tmap_compressed_kv, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, unsigned int* __restrict__ partition_arrivals, int* __restrict__ sparse_indices, int* __restrict__ sparse_topk_lens, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int sparse_topk, int num_splits, int num_head_tiles, int has_sinks, int completion_base)
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
    #define kv_full_addr (mbar_base + 8)
    #define s_full_addr (mbar_base + 40)
    #define p_full_addr (mbar_base + 48)
    #define o_full_lo_addr (mbar_base + 56)
    #define o_full_hi_addr (mbar_base + 72)
    #define tmem_dealloc_addr (mbar_base + 88)
    #define partial_done_addr (mbar_base + 96)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_swa_kv)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_compressed_kv)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kv_addr = smem + 9216;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_v_addr = smem + 9216;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int smem_p_addr = smem + 140288;
    float* smem_exch = reinterpret_cast<float*>(smem_raw + 142336);
    const int smem_exch_addr = smem + 142336;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 142464);
    const int smem_sum_addr = smem + 142464;
    int* smem_indices = reinterpret_cast<int*>(smem_raw + 142528);
    const int smem_indices_addr = smem + 142528;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 13 barriers)
    // Mbarriers at smem_raw[0..104)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // p_full: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // o_full_lo: 2 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // o_full_hi: 2 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // tmem_dealloc: 1 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            // partial_done: 1 barriers, init_count=256
            mbarrier_init(smem + 96, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 40 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 104);
    if (warp == 0) {
        int _tmem_hold = smem + 104;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o0 = taddr + 8;
    const int tmem_tmem_o1 = taddr + 16;
    const int tmem_tmem_o2 = taddr + 24;
    const int tmem_tmem_o3 = taddr + 32;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int work_idx = blockIdx.x;
            int head_tile = work_idx % num_head_tiles;
            int split_work = work_idx / num_head_tiles;
            int split_idx = split_work % num_splits;
            int query_idx = split_work / num_splits;
            int active_topk = sparse_topk_lens[query_idx];
            int head_base = head_tile * 8;
            const int token_idx = warp * 32 + lane;
            int global_token = split_idx * 128 + token_idx;
            const int wg_tid = warp * 32 + lane;
            unsigned int _phase_s_full_0 = 0;
            mbarrier_wait(s_full_addr, _phase_s_full_0);
            _phase_s_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int score_addr = taddr + (warp * 32 << 16);
            float _tmem_load_0[8];
            tmem_ld_x8(&_tmem_load_0[0], score_addr);
            int token_valid = 1;
            if (global_token >= active_topk) {
                token_valid = 0;
            }
            int staged_index = smem_indices[token_idx];
            if (staged_index < 0) {
                token_valid = 0;
            }
            if (token_valid == 0) {
                #pragma unroll
                for (int h = 0; h < 8; h++) {
                    _tmem_load_0[h] = -CAKE_INF;
                }
            }
            #pragma unroll
            for (int h_1 = 0; h_1 < 8; h_1++) {
                if (head_base + h_1 >= num_heads) {
                    _tmem_load_0[h_1] = -CAKE_INF;
                }
            }
            float partial_max[8];
            #pragma unroll
            for (int h_2 = 0; h_2 < 8; h_2++) {
                float _warp_reduce_0 = _tmem_load_0[h_2];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                partial_max[h_2] = _warp_reduce_0;
            }
            if (lane < 8) {
                smem_exch[warp * 8 + lane] = partial_max[lane];
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            float tile_max[8];
            if (lane < 8) {
                int h_lane = lane;
                float _max_0 = max_noftz(smem_exch[h_lane], smem_exch[8 + h_lane]);
                float _max_1 = max_noftz(smem_exch[16 + h_lane], smem_exch[24 + h_lane]);
                float _max_2 = max_noftz(_max_0, _max_1);
                tile_max[h_lane] = _max_2;
            }
            #pragma unroll
            for (int h_3 = 0; h_3 < 8; h_3++) {
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, tile_max[h_3], h_3);
                tile_max[h_3] = _shfl_0;
            }
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            float row_max_values[8];
            float exp_values[8];
            float warp_sum[8];
            #pragma unroll
            for (int h_4 = 0; h_4 < 8; h_4++) {
                int global_head = head_base + h_4;
                float row_max = tile_max[h_4];
                if (has_sinks != 0 && split_idx == 0 && global_head < num_heads) {
                    float sink_unscaled = sinks[global_head] * 1.4426950408889634f / softmax_scale_log2;
                    float _max_3 = max_noftz(row_max, sink_unscaled);
                    row_max = _max_3;
                }
                row_max_values[h_4] = row_max;
                float safe_max = ((row_max == -CAKE_INF) ? 0.0f : row_max);
                float max_scaled = safe_max * softmax_scale_log2;
                float _exp2_0 = approx_exp2(_tmem_load_0[h_4] * softmax_scale_log2 - max_scaled);
                exp_values[h_4] = ((_tmem_load_0[h_4] != -CAKE_INF) ? _exp2_0 : 0.0f);
                float _warp_reduce_1 = exp_values[h_4];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                warp_sum[h_4] = _warp_reduce_1;
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            if (lane < 8) {
                smem_exch[warp * 8 + lane] = warp_sum[lane];
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            if (warp == 0) {
                if (lane < 8) {
                    int h_lane_1 = lane;
                    int global_head_1 = head_base + h_lane_1;
                    float row_sum = smem_exch[h_lane_1] + smem_exch[8 + h_lane_1] + smem_exch[16 + h_lane_1] + smem_exch[24 + h_lane_1];
                    float row_max_1 = row_max_values[h_lane_1];
                    if (has_sinks != 0 && split_idx == 0 && global_head_1 < num_heads) {
                        float sink_scaled = sinks[global_head_1] * 1.4426950408889634f;
                        float max_scaled_1 = row_max_1 * softmax_scale_log2;
                        float _exp2_1 = approx_exp2(sink_scaled - max_scaled_1);
                        row_sum = row_sum + _exp2_1;
                    }
                    smem_sum[lane] = row_sum;
                    if (global_head_1 < num_heads) {
                        int lse_offset = (query_idx * num_heads + global_head_1) * num_splits + split_idx;
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(row_sum));
                        partial_lse[lse_offset] = ((row_sum > 0.0f) ? row_max_1 * softmax_scale_log2 + _log2_0 : -CAKE_INF);
                    }
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            #pragma unroll
            for (int h_5 = 0; h_5 < 4; h_5++) {
                {
                    __nv_bfloat16 _bval_0 = __float2bfloat16_rn(exp_values[h_5]);
                    uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                    uint32_t _addr_0 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(wg_tid % 64 / 64 * 1024 + (wg_tid / 64 * 8 + h_5) * 128 + wg_tid % 64 % 64 * 2 ^ (wg_tid % 64 / 64 * 1024 + (wg_tid / 64 * 8 + h_5) * 128 + wg_tid % 64 % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                }
            }
            #pragma unroll
            for (int h_6 = 4; h_6 < 8; h_6++) {
                {
                    __nv_bfloat16 _bval_1 = __float2bfloat16_rn(exp_values[h_6]);
                    uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                    uint32_t _addr_1 = static_cast<uint32_t>((smem_p_addr + (unsigned int)(wg_tid % 64 / 64 * 1024 + (wg_tid / 64 * 8 + h_6) * 128 + wg_tid % 64 % 64 * 2 ^ (wg_tid % 64 / 64 * 1024 + (wg_tid / 64 * 8 + h_6) * 128 + wg_tid % 64 % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
                }
            }
            asm volatile("fence.proxy.async;");
            __threadfence();
            mbarrier_arrive(p_full_addr);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    }
    // ---- Role: correction_lo ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // correction_lo_main
            int work_idx_1 = blockIdx.x;
            int head_tile_1 = work_idx_1 % num_head_tiles;
            int split_work_1 = work_idx_1 / num_head_tiles;
            int split_idx_1 = split_work_1 % num_splits;
            int query_idx_1 = split_work_1 / num_splits;
            int head_base_1 = head_tile_1 * 8;
            const int local_warp = warp - 4;
            const int d_lane = (unsigned int)(local_warp * 32) + lane;
            const int row_addr = local_warp * 32 << 16;
            float output_scale = bmm2_scale[0];
            int merge_group = query_idx_1 * num_head_tiles + head_tile_1;
            unsigned int _phase_p_full_0 = 0;
            mbarrier_wait(p_full_addr, _phase_p_full_0);
            _phase_p_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int _phase_o_full_lo_0 = 0;
            mbarrier_wait(o_full_lo_addr, _phase_o_full_lo_0);
            _phase_o_full_lo_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_1[8];
            tmem_ld_x8(&_tmem_load_1[0], taddr + 8 + (unsigned int)row_addr);
            const int d_idx0 = d_lane;
            #pragma unroll
            for (int h_7 = 0; h_7 < 8; h_7++) {
                int global_head_2 = head_base_1 + h_7;
                if (global_head_2 < num_heads) {
                    float row_sum_1 = smem_sum[h_7];
                    float _rcp_0 = approx_rcp(row_sum_1);
                    float inv_sum = ((row_sum_1 > 0.0f) ? _rcp_0 : 0.0f);
                    int output_offset = ((query_idx_1 * num_heads + global_head_2) * num_splits + split_idx_1) * 512 + d_idx0;
                    partial_O[output_offset] = _tmem_load_1[h_7] * inv_sum * output_scale;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            unsigned int _phase_o_full_lo_1 = 0;
            mbarrier_wait(o_full_lo_addr + 8, _phase_o_full_lo_1);
            _phase_o_full_lo_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_2[8];
            tmem_ld_x8(&_tmem_load_2[0], taddr + 16 + (unsigned int)row_addr);
            const int d_idx1 = 128 + d_lane;
            #pragma unroll
            for (int h_8 = 0; h_8 < 8; h_8++) {
                int global_head_3 = head_base_1 + h_8;
                if (global_head_3 < num_heads) {
                    float row_sum_2 = smem_sum[h_8];
                    float _rcp_1 = approx_rcp(row_sum_2);
                    float inv_sum_1 = ((row_sum_2 > 0.0f) ? _rcp_1 : 0.0f);
                    int output_offset_1 = ((query_idx_1 * num_heads + global_head_3) * num_splits + split_idx_1) * 512 + d_idx1;
                    partial_O[output_offset_1] = _tmem_load_2[h_8] * inv_sum_1 * output_scale;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            __threadfence();
            unsigned int _phase_partial_done_0 = 0;
            if (num_splits > 1) {
                mbarrier_arrive(partial_done_addr);
                mbarrier_wait(partial_done_addr, _phase_partial_done_0);
                _phase_partial_done_0 ^= 1;
                if (warp == 4) {
                    if (elect_sync()) {
                        {
                            unsigned int* _gc_p = reinterpret_cast<unsigned int*>(partition_arrivals) + (merge_group);
                            unsigned int _gc_old;
                            asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(_gc_old) : "l"(_gc_p) : "memory");
                        }
                    }
                }
                if (split_idx_1 == 0) {
                    {
                        unsigned int* _gca_p = reinterpret_cast<unsigned int*>(partition_arrivals) + (merge_group);
                        while (true) {
                            unsigned int _gca_v;
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                            if (_gca_v >= (unsigned int)(completion_base + num_splits)) break;
                        }
                    }
                    #pragma unroll
                    for (int local_h = 0; local_h < 2; local_h++) {
                        int global_head_4 = head_base_1 + local_warp * 2 + local_h;
                        if (global_head_4 < num_heads) {
                            int lse_base = (query_idx_1 * num_heads + global_head_4) * num_splits;
                            float local_lse = -CAKE_INF;
                            if (lane < (unsigned int)num_splits) {
                                local_lse = partial_lse[(unsigned int)lse_base + lane];
                            }
                            float _warp_reduce_2 = local_lse;
                            #pragma unroll
                            for (int offset = 16; offset > 0; offset >>= 1)
                                _warp_reduce_2 = max_noftz(_warp_reduce_2, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset));
                            float global_max = _warp_reduce_2;
                            float local_weight = 0.0f;
                            if (lane < (unsigned int)num_splits) {
                                float _exp2_2 = approx_exp2(local_lse - global_max);
                                local_weight = ((local_lse == -CAKE_INF) ? 0.0f : _exp2_2);
                            }
                            float _warp_reduce_3 = local_weight;
                            #pragma unroll
                            for (int offset = 16; offset > 0; offset >>= 1)
                                _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
                            float global_sum = _warp_reduce_3;
                            float _rcp_2 = approx_rcp(global_sum);
                            float normalized_weight = ((global_sum > 0.0f) ? local_weight * _rcp_2 : 0.0f);
                            int partial_base = (query_idx_1 * num_heads + global_head_4) * num_splits * 512;
                            #pragma unroll
                            for (int merge_chunk = 0; merge_chunk < 2; merge_chunk++) {
                                int merge_d_base = (unsigned int)(merge_chunk * 256) + lane * 8;
                                float merged[8];
                                #pragma unroll
                                for (int elem = 0; elem < 8; elem++) {
                                    merged[elem] = 0.0f;
                                }
                                #pragma unroll 3
                                for (int split = 0; split < num_splits; split++) {
                                    float _vec_load_0[8];
                                    {
                                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_O + (partial_base + split * 512 + merge_d_base) + 0);
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
                                    float _shfl_1 = __shfl_sync(0xFFFFFFFF, normalized_weight, split);
                                    float weight = _shfl_1;
                                    #pragma unroll
                                    for (int elem_1 = 0; elem_1 < 8; elem_1++) {
                                        merged[elem_1] = merged[elem_1] + weight * _vec_load_0[elem_1];
                                    }
                                }
                                int output_base = (query_idx_1 * num_heads + global_head_4) * 512 + merge_d_base;
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(merged[0 + 0], merged[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(merged[0 + 2], merged[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(merged[0 + 4], merged[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(merged[0 + 6], merged[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + output_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: correction_hi ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // correction_hi_main
            int work_idx_2 = blockIdx.x;
            int head_tile_2 = work_idx_2 % num_head_tiles;
            int split_work_2 = work_idx_2 / num_head_tiles;
            int split_idx_2 = split_work_2 % num_splits;
            int query_idx_2 = split_work_2 / num_splits;
            int head_base_2 = head_tile_2 * 8;
            const int local_warp_1 = warp - 8;
            const int d_lane_1 = (unsigned int)(local_warp_1 * 32) + lane;
            const int row_addr_1 = local_warp_1 * 32 << 16;
            float output_scale_1 = bmm2_scale[0];
            unsigned int _phase_p_full_0_1 = 0;
            mbarrier_wait(p_full_addr, _phase_p_full_0_1);
            _phase_p_full_0_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int _phase_o_full_hi_0 = 0;
            mbarrier_wait(o_full_hi_addr, _phase_o_full_hi_0);
            _phase_o_full_hi_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_3[8];
            tmem_ld_x8(&_tmem_load_3[0], taddr + 24 + (unsigned int)row_addr_1);
            const int d_idx2 = 256 + d_lane_1;
            #pragma unroll
            for (int h_9 = 0; h_9 < 8; h_9++) {
                int global_head_5 = head_base_2 + h_9;
                if (global_head_5 < num_heads) {
                    float row_sum_3 = smem_sum[h_9];
                    float _rcp_3 = approx_rcp(row_sum_3);
                    float inv_sum_2 = ((row_sum_3 > 0.0f) ? _rcp_3 : 0.0f);
                    int output_offset_2 = ((query_idx_2 * num_heads + global_head_5) * num_splits + split_idx_2) * 512 + d_idx2;
                    partial_O[output_offset_2] = _tmem_load_3[h_9] * inv_sum_2 * output_scale_1;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            unsigned int _phase_o_full_hi_1 = 0;
            mbarrier_wait(o_full_hi_addr + 8, _phase_o_full_hi_1);
            _phase_o_full_hi_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_4[8];
            tmem_ld_x8(&_tmem_load_4[0], taddr + 32 + (unsigned int)row_addr_1);
            const int d_idx3 = 384 + d_lane_1;
            #pragma unroll
            for (int h_10 = 0; h_10 < 8; h_10++) {
                int global_head_6 = head_base_2 + h_10;
                if (global_head_6 < num_heads) {
                    float row_sum_4 = smem_sum[h_10];
                    float _rcp_4 = approx_rcp(row_sum_4);
                    float inv_sum_3 = ((row_sum_4 > 0.0f) ? _rcp_4 : 0.0f);
                    int output_offset_3 = ((query_idx_2 * num_heads + global_head_6) * num_splits + split_idx_2) * 512 + d_idx3;
                    partial_O[output_offset_3] = _tmem_load_4[h_10] * inv_sum_3 * output_scale_1;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
            __threadfence();
            if (num_splits > 1) {
                mbarrier_arrive(partial_done_addr);
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            unsigned int _phase_kv_full = 0;
            #pragma unroll
            for (int k_stage = 0; k_stage < 4; k_stage++) {
                mbarrier_wait(kv_full_addr + (k_stage) * 8, _phase_kv_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k_stage) * 2048);
                int _mma_b_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (k_stage) * 128);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, ((k_stage == 0) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                    }
                }
                if (k_stage == 3) {
                    elect_commit(s_full_addr);
                }
            }
            unsigned int _phase_p_full_0_2 = 0;
            mbarrier_wait(p_full_addr, _phase_p_full_0_2);
            _phase_p_full_0_2 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            mbarrier_wait(kv_full_addr, 0);
            int _mma_a_lo_1 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (0) * 2048);
            int _mma_b_lo_1 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) | 0x400000);
            {
                uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134382736, 1);
                }
            }
            elect_commit(o_full_lo_addr);
            mbarrier_wait(kv_full_addr + 8, 0);
            int _mma_a_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (1) * 2048);
            {
                uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                }
            }
            elect_commit(o_full_lo_addr + 8);
            mbarrier_wait(kv_full_addr + 16, 0);
            int _mma_a_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (2) * 2048);
            {
                uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o2, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                }
            }
            elect_commit(o_full_hi_addr);
            mbarrier_wait(kv_full_addr + 24, 0);
            int _mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (3) * 2048);
            {
                uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 58U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o3, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134382736, 1);
                }
            }
            elect_commit(o_full_hi_addr + 8);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(64));
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 13 && warp <= 15) {
        { // load_warp_main
            const int load_rank = warp - 13;
            int work_idx_3 = blockIdx.x;
            int head_tile_3 = work_idx_3 % num_head_tiles;
            int split_work_3 = work_idx_3 / num_head_tiles;
            int split_idx_3 = split_work_3 % num_splits;
            int query_idx_3 = split_work_3 / num_splits;
            int sparse_base = query_idx_3 * sparse_topk;
            int vector_offset = lane * 4;
            if (load_rank == 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 8192);
                    #pragma unroll
                    for (int q_stage = 0; q_stage < 4; q_stage++) {
                        tma_4d_gmem2smem(smem_q_addr + (unsigned int)(q_stage * 2048), tmap_q, 0, head_tile_3 * 8, q_stage * 2, query_idx_3, q_full_addr);
                    }
                }
                int global_vector_offset = split_idx_3 * 128 + vector_offset;
                if (global_vector_offset + 3 < sparse_topk) {
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16;"
                        :: "r"(smem_indices_addr + (unsigned int)(vector_offset * 4)), "l"(sparse_indices + (sparse_base + global_vector_offset)));
                } else {
                    int staged_rows[4];
                    #pragma unroll
                    for (int row_i = 0; row_i < 4; row_i++) {
                        staged_rows[row_i] = -1;
                        if (global_vector_offset + row_i < sparse_topk) {
                            staged_rows[row_i] = sparse_indices[sparse_base + global_vector_offset + row_i];
                        }
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_indices_addr + (unsigned int)(vector_offset * 4)), "r"(staged_rows[0]), "r"(staged_rows[1]), "r"(staged_rows[2]), "r"(staged_rows[3]) : "memory");
                }
                asm volatile("cp.async.commit_group;");
            }
            int active_topk_1 = sparse_topk_lens[query_idx_3];
            int valid_rows = active_topk_1 - split_idx_3 * 128;
            if (valid_rows < 0) {
                valid_rows = 0;
            }
            if (valid_rows > 128) {
                valid_rows = 128;
            }
            int valid_groups = (valid_rows + 3) / 4;
            int loader_thread = (unsigned int)(load_rank * 32) + lane;
            int tail_vecs_per_half = (32 - valid_groups) * 512 / 16;
            #pragma unroll
            for (int init_stage = 0; init_stage < 4; init_stage++) {
                #pragma unroll
                for (int init_half = 0; init_half < 2; init_half++) {
                    for (int tail_vec = loader_thread; tail_vec < tail_vecs_per_half; tail_vec += 96) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_kv_addr + (unsigned int)(init_stage * 32768 + init_half * 16384 + valid_groups * 512 + tail_vec * 16)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                    }
                }
            }
            if (load_rank == 0) {
                asm volatile("cp.async.wait_group 0;");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 8, 96;" ::: "memory");
            int raw_rows[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 3]))
                : "r"(smem_indices_addr + (unsigned int)(vector_offset * 4)));
            int row0 = ((raw_rows[0] >= 0) ? raw_rows[0] : 0);
            int row1 = ((raw_rows[1] >= 0) ? raw_rows[1] : 0);
            int row2 = ((raw_rows[2] >= 0) ? raw_rows[2] : 0);
            int row3 = ((raw_rows[3] >= 0) ? raw_rows[3] : 0);
            int gather_tx_bytes = valid_groups * 2 * 512;
            #pragma unroll
            for (int k_stage_1 = 0; k_stage_1 < 4; k_stage_1++) {
                if (k_stage_1 % 3 == load_rank) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (k_stage_1) * 8, gather_tx_bytes);
                    }
                    if ((unsigned int)valid_groups > lane) {
                        int dst_k = smem_kv_addr + (unsigned int)(k_stage_1 * 32768);
                        if (split_idx_3 == 0) {
                            tma_gather4_gmem2smem((unsigned int)dst_k + lane * 512, tmap_swa_kv, k_stage_1 * 128, row0, row1, row2, row3, kv_full_addr + (k_stage_1) * 8);
                            tma_gather4_gmem2smem((unsigned int)(dst_k + 16384) + lane * 512, tmap_swa_kv, k_stage_1 * 128 + 64, row0, row1, row2, row3, kv_full_addr + (k_stage_1) * 8);
                        } else {
                            tma_gather4_gmem2smem((unsigned int)dst_k + lane * 512, tmap_compressed_kv, k_stage_1 * 128, row0, row1, row2, row3, kv_full_addr + (k_stage_1) * 8);
                            tma_gather4_gmem2smem((unsigned int)(dst_k + 16384) + lane * 512, tmap_compressed_kv, k_stage_1 * 128 + 64, row0, row1, row2, row3, kv_full_addr + (k_stage_1) * 8);
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
