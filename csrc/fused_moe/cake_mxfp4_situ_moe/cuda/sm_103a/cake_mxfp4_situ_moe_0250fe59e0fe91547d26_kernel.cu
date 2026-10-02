/*
 * Copyright (c) 2026 by FlashInfer team.
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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 276
#define TMEM_ACC_OFFSET 0
#define TMEM_SF_A_OFFSET 256
#define TMEM_SF_B_OFFSET 272
#define NUM_PA_STAGES 6
#define NUM_PB_STAGES 6
#define NUM_PACC_STAGES 2
#define NUM_PTILE_STAGES 2
#define NUM_PC_STAGES 3
#define NUM_PACC_BUF_STAGES 2
#define SMEM_C_STAGE_OFF 1024
#define SMEM_C_STAGE_STAGE_BYTES 8192
#define SMEM_C_STAGE_STRIDE 8192
#define SMEM_ACT_OFF 25600
#define SMEM_ACT_STAGE_BYTES 16384
#define SMEM_ACT_STRIDE 16384
#define SMEM_W_OFF 123904
#define SMEM_W_STAGE_BYTES 16384
#define SMEM_W_STRIDE 16384
#define SMEM_ACT_SF_OFF 222208
#define SMEM_ACT_SF_STAGE_BYTES 512
#define SMEM_ACT_SF_STRIDE 512
#define SMEM_W_SF_OFF 225280
#define SMEM_W_SF_STAGE_BYTES 512
#define SMEM_W_SF_STRIDE 512
#define SMEM_SINFO_OFF 228352
#define SMEM_SINFO_STAGE_BYTES 40
#define SMEM_SINFO_STRIDE 40
#define SMEM_ZF_STAGE_OFF 1024
#define SMEM_ZF_STAGE_STAGE_BYTES 16384
#define SMEM_ZF_STAGE_STRIDE 16384
#define SMEM_TOTAL 228480
#define THREADS 384

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


__device__ __forceinline__ void tcgen05_mma_mxf8_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ void tcgen05_mma_mxf8_bs_elect(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
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


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
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


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384) void
kernel_cake_mxfp4_situ_moe_0250fe59e0fe91547d26(uint8_t* __restrict__ X, uint8_t* __restrict__ XSF, CakeTensorMap const* W, CakeTensorMap const* WSF, CakeTensorMap const* C, uint8_t* __restrict__ CSF, float* __restrict__ alpha, float* __restrict__ situ_beta, float* __restrict__ situ_linear_beta, int* __restrict__ tile_idx_to_expert_idx, int* __restrict__ tile_idx_to_mn_limit, int* __restrict__ tile_idx_to_row_group, int* __restrict__ token_id_mapping, int* __restrict__ num_non_exiting_tiles, unsigned int* __restrict__ zero_fill_words, int* __restrict__ zero_fill_counters, int* __restrict__ zero_fill_other_tiles, int zero_fill_num_words, int num_m_tiles, int n_tiles, int k_tiles, int k_cols, int sf_cols, int top_k, int sf_n_blocks, int beta_stride, int linear_beta_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define a_free_addr (mbar_base + 0)
    #define a_full_addr (mbar_base + 48)
    #define b_full_addr (mbar_base + 96)
    #define b_free_addr (mbar_base + 144)
    #define acc_full_addr (mbar_base + 192)
    #define acc_free_addr (mbar_base + 208)
    #define tile_full_addr (mbar_base + 224)
    #define tile_free_addr (mbar_base + 240)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(W)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(WSF)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(C)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* c_stage = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int c_stage_addr = smem + 1024;
    uint8_t* act = reinterpret_cast<uint8_t*>(smem_raw + 25600);
    const int act_addr = smem + 25600;
    uint8_t* w = reinterpret_cast<uint8_t*>(smem_raw + 123904);
    const int w_addr = smem + 123904;
    uint8_t* act_sf = reinterpret_cast<uint8_t*>(smem_raw + 222208);
    const int act_sf_addr = smem + 222208;
    uint8_t* w_sf = reinterpret_cast<uint8_t*>(smem_raw + 225280);
    const int w_sf_addr = smem + 225280;
    int* sinfo = reinterpret_cast<int*>(smem_raw + 228352);
    const int sinfo_addr = smem + 228352;
    unsigned int* zf_stage = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int zf_stage_addr = smem + 1024;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)

    if (warp == 0) {
        // --- pipeline 'pa' ---
        // a_free: 6 barriers, init_count=1
        // a_full: 6 barriers, init_count=128
        // --- pipeline 'pb' ---
        // b_full: 6 barriers, init_count=1
        // b_free: 6 barriers, init_count=1
        // --- pipeline 'pacc' ---
        // acc_full: 2 barriers, init_count=1
        // acc_free: 2 barriers, init_count=128
        // --- pipeline 'ptile' ---
        // tile_full: 2 barriers, init_count=32
        // tile_free: 2 barriers, init_count=320
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 320;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(30), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(28), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(26), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(12), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(6), "r"((uint32_t)(1)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 276 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
    if (warp == 0) {
        int _tmem_hold = smem + 256;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_sf_a = taddr + 256;
    const int tmem_sf_b = taddr + 272;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int epi_tidx = tid;
            int lane_0 = lane;
            int epi_warp = warp;
            unsigned int lane_base = (unsigned int)(epi_warp * 32) << 16;
            unsigned int acc_stage = 0;
            unsigned int acc_idx = 0;
            unsigned int tile_stage = 0;
            unsigned int c_buf = 1;
            int info[5];
            float vals[64];
            unsigned int words[16];
            float inv_fp8_max = 0.002232142857142857f;
            float one_f32 = 1.0f;
            unsigned int _phase_tile_full = 0;
            mbarrier_wait(tile_full_addr + (tile_stage) * 8, _phase_tile_full);
            info[0] = sinfo[tile_stage * 5];
            info[1] = sinfo[tile_stage * 5 + 1];
            info[2] = sinfo[tile_stage * 5 + 2];
            info[3] = sinfo[tile_stage * 5 + 3];
            info[4] = sinfo[tile_stage * 5 + 4];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage) * 8);
            tile_stage += 1;
            if (tile_stage == 2) { tile_stage = 0; _phase_tile_full ^= 1; }
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int _tile = 0; _tile < num_m_tiles * n_tiles + 1; _tile++) {
                if (info[3] == 0) {
                    break;
                }
                int m_tile = info[0];
                int n_tile = info[1];
                int expert_e = info[2];
                float alpha_val = alpha[expert_e];
                float beta = situ_beta[expert_e * beta_stride];
                float _fdiv_rn_0 = __fdiv_rn(1.0f, beta);
                float inv_beta = _fdiv_rn_0;
                float linear_beta = situ_linear_beta[expert_e * linear_beta_stride];
                float _fdiv_rn_1 = __fdiv_rn(1.0f, linear_beta);
                float inv_linear_beta = _fdiv_rn_1;
                int row_out = m_tile * 128;
                int col_out = n_tile * 64;
                int sf_base = lane_0 * 16 + epi_warp * 4 + n_tile / 2 * 512 + n_tile % 2 * 2 + m_tile * (512 * sf_n_blocks);
                int acc_col = (int)acc_stage * 128;
                int rev = 0;
                mbarrier_wait(acc_full_addr + (acc_stage) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll 1
                for (int step = 0; step < 1; step++) {
                    int real = rev ^ step;
                    unsigned int col0 = (unsigned int)(acc_col + real * 128);
                    float _tmem_load_0[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(taddr + lane_base + col0));
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31]), "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(taddr + lane_base + col0 + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float2 _f2_0 = make_float2(_tmem_load_0[0], _tmem_load_0[1]);
                    float2 _f2_1 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_0;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_1));
                    float2 _f2_2 = make_float2(_tmem_load_1[0], _tmem_load_1[1]);
                    float2 _f2_3 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_1;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_f2_2), "l"(*(const unsigned long long*)&_f2_3));
                    float g0 = _mul_f32x2_1.x;
                    float g1 = _mul_f32x2_1.y;
                    float _tanh_approx_0;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(g0 * inv_beta));
                    float _exp2_0 = approx_exp2(g0 * -1.4426950408889634f);
                    float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                    float s0 = beta * _tanh_approx_0 * _rcp_0;
                    float _tanh_approx_1;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(g1 * inv_beta));
                    float _exp2_1 = approx_exp2(g1 * -1.4426950408889634f);
                    float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                    float s1 = beta * _tanh_approx_1 * _rcp_1;
                    float _tanh_approx_2;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(_mul_f32x2_0.x * inv_linear_beta));
                    float u0 = linear_beta * _tanh_approx_2;
                    float _tanh_approx_3;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(_mul_f32x2_0.y * inv_linear_beta));
                    float u1 = linear_beta * _tanh_approx_3;
                    float2 _f2_4 = make_float2(u0, u1);
                    float2 _f2_5 = make_float2(s0, s1);
                    float2 _mul_f32x2_2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_4), "l"(*(const unsigned long long*)&_f2_5));
                    vals[0] = _mul_f32x2_2.x;
                    vals[1] = _mul_f32x2_2.y;
                    float2 _f2_6 = make_float2(_tmem_load_0[2], _tmem_load_0[3]);
                    float2 _f2_7 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_3;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_f2_6), "l"(*(const unsigned long long*)&_f2_7));
                    float2 _f2_8 = make_float2(_tmem_load_1[2], _tmem_load_1[3]);
                    float2 _f2_9 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_4;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_8), "l"(*(const unsigned long long*)&_f2_9));
                    float g0_0 = _mul_f32x2_4.x;
                    float g1_1 = _mul_f32x2_4.y;
                    float _tanh_approx_4;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_4) : "f"(g0_0 * inv_beta));
                    float _exp2_2 = approx_exp2(g0_0 * -1.4426950408889634f);
                    float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                    float s0_2 = beta * _tanh_approx_4 * _rcp_2;
                    float _tanh_approx_5;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_5) : "f"(g1_1 * inv_beta));
                    float _exp2_3 = approx_exp2(g1_1 * -1.4426950408889634f);
                    float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                    float s1_3 = beta * _tanh_approx_5 * _rcp_3;
                    float _tanh_approx_6;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_6) : "f"(_mul_f32x2_3.x * inv_linear_beta));
                    float u0_4 = linear_beta * _tanh_approx_6;
                    float _tanh_approx_7;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_7) : "f"(_mul_f32x2_3.y * inv_linear_beta));
                    float u1_5 = linear_beta * _tanh_approx_7;
                    float2 _f2_10 = make_float2(u0_4, u1_5);
                    float2 _f2_11 = make_float2(s0_2, s1_3);
                    float2 _mul_f32x2_5;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_f2_10), "l"(*(const unsigned long long*)&_f2_11));
                    vals[2] = _mul_f32x2_5.x;
                    vals[3] = _mul_f32x2_5.y;
                    float2 _f2_12 = make_float2(_tmem_load_0[4], _tmem_load_0[5]);
                    float2 _f2_13 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_6;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&_f2_12), "l"(*(const unsigned long long*)&_f2_13));
                    float2 _f2_14 = make_float2(_tmem_load_1[4], _tmem_load_1[5]);
                    float2 _f2_15 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_7;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&_f2_14), "l"(*(const unsigned long long*)&_f2_15));
                    float g0_6 = _mul_f32x2_7.x;
                    float g1_7 = _mul_f32x2_7.y;
                    float _tanh_approx_8;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_8) : "f"(g0_6 * inv_beta));
                    float _exp2_4 = approx_exp2(g0_6 * -1.4426950408889634f);
                    float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                    float s0_8 = beta * _tanh_approx_8 * _rcp_4;
                    float _tanh_approx_9;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_9) : "f"(g1_7 * inv_beta));
                    float _exp2_5 = approx_exp2(g1_7 * -1.4426950408889634f);
                    float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                    float s1_9 = beta * _tanh_approx_9 * _rcp_5;
                    float _tanh_approx_10;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_10) : "f"(_mul_f32x2_6.x * inv_linear_beta));
                    float u0_10 = linear_beta * _tanh_approx_10;
                    float _tanh_approx_11;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_11) : "f"(_mul_f32x2_6.y * inv_linear_beta));
                    float u1_11 = linear_beta * _tanh_approx_11;
                    float2 _f2_16 = make_float2(u0_10, u1_11);
                    float2 _f2_17 = make_float2(s0_8, s1_9);
                    float2 _mul_f32x2_8;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&_f2_16), "l"(*(const unsigned long long*)&_f2_17));
                    vals[4] = _mul_f32x2_8.x;
                    vals[5] = _mul_f32x2_8.y;
                    float2 _f2_18 = make_float2(_tmem_load_0[6], _tmem_load_0[7]);
                    float2 _f2_19 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_9;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_f2_18), "l"(*(const unsigned long long*)&_f2_19));
                    float2 _f2_20 = make_float2(_tmem_load_1[6], _tmem_load_1[7]);
                    float2 _f2_21 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_10;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&_f2_20), "l"(*(const unsigned long long*)&_f2_21));
                    float g0_12 = _mul_f32x2_10.x;
                    float g1_13 = _mul_f32x2_10.y;
                    float _tanh_approx_12;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_12) : "f"(g0_12 * inv_beta));
                    float _exp2_6 = approx_exp2(g0_12 * -1.4426950408889634f);
                    float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                    float s0_14 = beta * _tanh_approx_12 * _rcp_6;
                    float _tanh_approx_13;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_13) : "f"(g1_13 * inv_beta));
                    float _exp2_7 = approx_exp2(g1_13 * -1.4426950408889634f);
                    float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                    float s1_15 = beta * _tanh_approx_13 * _rcp_7;
                    float _tanh_approx_14;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_14) : "f"(_mul_f32x2_9.x * inv_linear_beta));
                    float u0_16 = linear_beta * _tanh_approx_14;
                    float _tanh_approx_15;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_15) : "f"(_mul_f32x2_9.y * inv_linear_beta));
                    float u1_17 = linear_beta * _tanh_approx_15;
                    float2 _f2_22 = make_float2(u0_16, u1_17);
                    float2 _f2_23 = make_float2(s0_14, s1_15);
                    float2 _mul_f32x2_11;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&_f2_22), "l"(*(const unsigned long long*)&_f2_23));
                    vals[6] = _mul_f32x2_11.x;
                    vals[7] = _mul_f32x2_11.y;
                    float2 _f2_24 = make_float2(_tmem_load_0[8], _tmem_load_0[9]);
                    float2 _f2_25 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_12;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_25));
                    float2 _f2_26 = make_float2(_tmem_load_1[8], _tmem_load_1[9]);
                    float2 _f2_27 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_13;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&_f2_26), "l"(*(const unsigned long long*)&_f2_27));
                    float g0_18 = _mul_f32x2_13.x;
                    float g1_19 = _mul_f32x2_13.y;
                    float _tanh_approx_16;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_16) : "f"(g0_18 * inv_beta));
                    float _exp2_8 = approx_exp2(g0_18 * -1.4426950408889634f);
                    float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                    float s0_20 = beta * _tanh_approx_16 * _rcp_8;
                    float _tanh_approx_17;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_17) : "f"(g1_19 * inv_beta));
                    float _exp2_9 = approx_exp2(g1_19 * -1.4426950408889634f);
                    float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                    float s1_21 = beta * _tanh_approx_17 * _rcp_9;
                    float _tanh_approx_18;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_18) : "f"(_mul_f32x2_12.x * inv_linear_beta));
                    float u0_22 = linear_beta * _tanh_approx_18;
                    float _tanh_approx_19;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_19) : "f"(_mul_f32x2_12.y * inv_linear_beta));
                    float u1_23 = linear_beta * _tanh_approx_19;
                    float2 _f2_28 = make_float2(u0_22, u1_23);
                    float2 _f2_29 = make_float2(s0_20, s1_21);
                    float2 _mul_f32x2_14;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_28), "l"(*(const unsigned long long*)&_f2_29));
                    vals[8] = _mul_f32x2_14.x;
                    vals[9] = _mul_f32x2_14.y;
                    float2 _f2_30 = make_float2(_tmem_load_0[10], _tmem_load_0[11]);
                    float2 _f2_31 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_15;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&_f2_30), "l"(*(const unsigned long long*)&_f2_31));
                    float2 _f2_32 = make_float2(_tmem_load_1[10], _tmem_load_1[11]);
                    float2 _f2_33 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_16;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&_f2_32), "l"(*(const unsigned long long*)&_f2_33));
                    float g0_24 = _mul_f32x2_16.x;
                    float g1_25 = _mul_f32x2_16.y;
                    float _tanh_approx_20;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_20) : "f"(g0_24 * inv_beta));
                    float _exp2_10 = approx_exp2(g0_24 * -1.4426950408889634f);
                    float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                    float s0_26 = beta * _tanh_approx_20 * _rcp_10;
                    float _tanh_approx_21;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_21) : "f"(g1_25 * inv_beta));
                    float _exp2_11 = approx_exp2(g1_25 * -1.4426950408889634f);
                    float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                    float s1_27 = beta * _tanh_approx_21 * _rcp_11;
                    float _tanh_approx_22;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_22) : "f"(_mul_f32x2_15.x * inv_linear_beta));
                    float u0_28 = linear_beta * _tanh_approx_22;
                    float _tanh_approx_23;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_23) : "f"(_mul_f32x2_15.y * inv_linear_beta));
                    float u1_29 = linear_beta * _tanh_approx_23;
                    float2 _f2_34 = make_float2(u0_28, u1_29);
                    float2 _f2_35 = make_float2(s0_26, s1_27);
                    float2 _mul_f32x2_17;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&_f2_34), "l"(*(const unsigned long long*)&_f2_35));
                    vals[10] = _mul_f32x2_17.x;
                    vals[11] = _mul_f32x2_17.y;
                    float2 _f2_36 = make_float2(_tmem_load_0[12], _tmem_load_0[13]);
                    float2 _f2_37 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_18;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&_f2_36), "l"(*(const unsigned long long*)&_f2_37));
                    float2 _f2_38 = make_float2(_tmem_load_1[12], _tmem_load_1[13]);
                    float2 _f2_39 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_19;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&_f2_38), "l"(*(const unsigned long long*)&_f2_39));
                    float g0_30 = _mul_f32x2_19.x;
                    float g1_31 = _mul_f32x2_19.y;
                    float _tanh_approx_24;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_24) : "f"(g0_30 * inv_beta));
                    float _exp2_12 = approx_exp2(g0_30 * -1.4426950408889634f);
                    float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                    float s0_32 = beta * _tanh_approx_24 * _rcp_12;
                    float _tanh_approx_25;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_25) : "f"(g1_31 * inv_beta));
                    float _exp2_13 = approx_exp2(g1_31 * -1.4426950408889634f);
                    float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                    float s1_33 = beta * _tanh_approx_25 * _rcp_13;
                    float _tanh_approx_26;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_26) : "f"(_mul_f32x2_18.x * inv_linear_beta));
                    float u0_34 = linear_beta * _tanh_approx_26;
                    float _tanh_approx_27;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_27) : "f"(_mul_f32x2_18.y * inv_linear_beta));
                    float u1_35 = linear_beta * _tanh_approx_27;
                    float2 _f2_40 = make_float2(u0_34, u1_35);
                    float2 _f2_41 = make_float2(s0_32, s1_33);
                    float2 _mul_f32x2_20;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_20) : "l"(*(const unsigned long long*)&_f2_40), "l"(*(const unsigned long long*)&_f2_41));
                    vals[12] = _mul_f32x2_20.x;
                    vals[13] = _mul_f32x2_20.y;
                    float2 _f2_42 = make_float2(_tmem_load_0[14], _tmem_load_0[15]);
                    float2 _f2_43 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_21;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_21) : "l"(*(const unsigned long long*)&_f2_42), "l"(*(const unsigned long long*)&_f2_43));
                    float2 _f2_44 = make_float2(_tmem_load_1[14], _tmem_load_1[15]);
                    float2 _f2_45 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_22;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_22) : "l"(*(const unsigned long long*)&_f2_44), "l"(*(const unsigned long long*)&_f2_45));
                    float g0_36 = _mul_f32x2_22.x;
                    float g1_37 = _mul_f32x2_22.y;
                    float _tanh_approx_28;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_28) : "f"(g0_36 * inv_beta));
                    float _exp2_14 = approx_exp2(g0_36 * -1.4426950408889634f);
                    float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                    float s0_38 = beta * _tanh_approx_28 * _rcp_14;
                    float _tanh_approx_29;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_29) : "f"(g1_37 * inv_beta));
                    float _exp2_15 = approx_exp2(g1_37 * -1.4426950408889634f);
                    float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                    float s1_39 = beta * _tanh_approx_29 * _rcp_15;
                    float _tanh_approx_30;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_30) : "f"(_mul_f32x2_21.x * inv_linear_beta));
                    float u0_40 = linear_beta * _tanh_approx_30;
                    float _tanh_approx_31;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_31) : "f"(_mul_f32x2_21.y * inv_linear_beta));
                    float u1_41 = linear_beta * _tanh_approx_31;
                    float2 _f2_46 = make_float2(u0_40, u1_41);
                    float2 _f2_47 = make_float2(s0_38, s1_39);
                    float2 _mul_f32x2_23;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_23) : "l"(*(const unsigned long long*)&_f2_46), "l"(*(const unsigned long long*)&_f2_47));
                    vals[14] = _mul_f32x2_23.x;
                    vals[15] = _mul_f32x2_23.y;
                    float2 _f2_48 = make_float2(_tmem_load_0[16], _tmem_load_0[17]);
                    float2 _f2_49 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_24;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_24) : "l"(*(const unsigned long long*)&_f2_48), "l"(*(const unsigned long long*)&_f2_49));
                    float2 _f2_50 = make_float2(_tmem_load_1[16], _tmem_load_1[17]);
                    float2 _f2_51 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_25;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_25) : "l"(*(const unsigned long long*)&_f2_50), "l"(*(const unsigned long long*)&_f2_51));
                    float g0_42 = _mul_f32x2_25.x;
                    float g1_43 = _mul_f32x2_25.y;
                    float _tanh_approx_32;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_32) : "f"(g0_42 * inv_beta));
                    float _exp2_16 = approx_exp2(g0_42 * -1.4426950408889634f);
                    float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                    float s0_44 = beta * _tanh_approx_32 * _rcp_16;
                    float _tanh_approx_33;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_33) : "f"(g1_43 * inv_beta));
                    float _exp2_17 = approx_exp2(g1_43 * -1.4426950408889634f);
                    float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                    float s1_45 = beta * _tanh_approx_33 * _rcp_17;
                    float _tanh_approx_34;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_34) : "f"(_mul_f32x2_24.x * inv_linear_beta));
                    float u0_46 = linear_beta * _tanh_approx_34;
                    float _tanh_approx_35;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_35) : "f"(_mul_f32x2_24.y * inv_linear_beta));
                    float u1_47 = linear_beta * _tanh_approx_35;
                    float2 _f2_52 = make_float2(u0_46, u1_47);
                    float2 _f2_53 = make_float2(s0_44, s1_45);
                    float2 _mul_f32x2_26;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_26) : "l"(*(const unsigned long long*)&_f2_52), "l"(*(const unsigned long long*)&_f2_53));
                    vals[16] = _mul_f32x2_26.x;
                    vals[17] = _mul_f32x2_26.y;
                    float2 _f2_54 = make_float2(_tmem_load_0[18], _tmem_load_0[19]);
                    float2 _f2_55 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_27;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_27) : "l"(*(const unsigned long long*)&_f2_54), "l"(*(const unsigned long long*)&_f2_55));
                    float2 _f2_56 = make_float2(_tmem_load_1[18], _tmem_load_1[19]);
                    float2 _f2_57 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_28;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_28) : "l"(*(const unsigned long long*)&_f2_56), "l"(*(const unsigned long long*)&_f2_57));
                    float g0_48 = _mul_f32x2_28.x;
                    float g1_49 = _mul_f32x2_28.y;
                    float _tanh_approx_36;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_36) : "f"(g0_48 * inv_beta));
                    float _exp2_18 = approx_exp2(g0_48 * -1.4426950408889634f);
                    float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                    float s0_50 = beta * _tanh_approx_36 * _rcp_18;
                    float _tanh_approx_37;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_37) : "f"(g1_49 * inv_beta));
                    float _exp2_19 = approx_exp2(g1_49 * -1.4426950408889634f);
                    float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                    float s1_51 = beta * _tanh_approx_37 * _rcp_19;
                    float _tanh_approx_38;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_38) : "f"(_mul_f32x2_27.x * inv_linear_beta));
                    float u0_52 = linear_beta * _tanh_approx_38;
                    float _tanh_approx_39;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_39) : "f"(_mul_f32x2_27.y * inv_linear_beta));
                    float u1_53 = linear_beta * _tanh_approx_39;
                    float2 _f2_58 = make_float2(u0_52, u1_53);
                    float2 _f2_59 = make_float2(s0_50, s1_51);
                    float2 _mul_f32x2_29;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_29) : "l"(*(const unsigned long long*)&_f2_58), "l"(*(const unsigned long long*)&_f2_59));
                    vals[18] = _mul_f32x2_29.x;
                    vals[19] = _mul_f32x2_29.y;
                    float2 _f2_60 = make_float2(_tmem_load_0[20], _tmem_load_0[21]);
                    float2 _f2_61 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_30;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_30) : "l"(*(const unsigned long long*)&_f2_60), "l"(*(const unsigned long long*)&_f2_61));
                    float2 _f2_62 = make_float2(_tmem_load_1[20], _tmem_load_1[21]);
                    float2 _f2_63 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_31;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_31) : "l"(*(const unsigned long long*)&_f2_62), "l"(*(const unsigned long long*)&_f2_63));
                    float g0_54 = _mul_f32x2_31.x;
                    float g1_55 = _mul_f32x2_31.y;
                    float _tanh_approx_40;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_40) : "f"(g0_54 * inv_beta));
                    float _exp2_20 = approx_exp2(g0_54 * -1.4426950408889634f);
                    float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                    float s0_56 = beta * _tanh_approx_40 * _rcp_20;
                    float _tanh_approx_41;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_41) : "f"(g1_55 * inv_beta));
                    float _exp2_21 = approx_exp2(g1_55 * -1.4426950408889634f);
                    float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                    float s1_57 = beta * _tanh_approx_41 * _rcp_21;
                    float _tanh_approx_42;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_42) : "f"(_mul_f32x2_30.x * inv_linear_beta));
                    float u0_58 = linear_beta * _tanh_approx_42;
                    float _tanh_approx_43;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_43) : "f"(_mul_f32x2_30.y * inv_linear_beta));
                    float u1_59 = linear_beta * _tanh_approx_43;
                    float2 _f2_64 = make_float2(u0_58, u1_59);
                    float2 _f2_65 = make_float2(s0_56, s1_57);
                    float2 _mul_f32x2_32;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_32) : "l"(*(const unsigned long long*)&_f2_64), "l"(*(const unsigned long long*)&_f2_65));
                    vals[20] = _mul_f32x2_32.x;
                    vals[21] = _mul_f32x2_32.y;
                    float2 _f2_66 = make_float2(_tmem_load_0[22], _tmem_load_0[23]);
                    float2 _f2_67 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_33;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_33) : "l"(*(const unsigned long long*)&_f2_66), "l"(*(const unsigned long long*)&_f2_67));
                    float2 _f2_68 = make_float2(_tmem_load_1[22], _tmem_load_1[23]);
                    float2 _f2_69 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_34;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_34) : "l"(*(const unsigned long long*)&_f2_68), "l"(*(const unsigned long long*)&_f2_69));
                    float g0_60 = _mul_f32x2_34.x;
                    float g1_61 = _mul_f32x2_34.y;
                    float _tanh_approx_44;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_44) : "f"(g0_60 * inv_beta));
                    float _exp2_22 = approx_exp2(g0_60 * -1.4426950408889634f);
                    float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                    float s0_62 = beta * _tanh_approx_44 * _rcp_22;
                    float _tanh_approx_45;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_45) : "f"(g1_61 * inv_beta));
                    float _exp2_23 = approx_exp2(g1_61 * -1.4426950408889634f);
                    float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                    float s1_63 = beta * _tanh_approx_45 * _rcp_23;
                    float _tanh_approx_46;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_46) : "f"(_mul_f32x2_33.x * inv_linear_beta));
                    float u0_64 = linear_beta * _tanh_approx_46;
                    float _tanh_approx_47;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_47) : "f"(_mul_f32x2_33.y * inv_linear_beta));
                    float u1_65 = linear_beta * _tanh_approx_47;
                    float2 _f2_70 = make_float2(u0_64, u1_65);
                    float2 _f2_71 = make_float2(s0_62, s1_63);
                    float2 _mul_f32x2_35;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_35) : "l"(*(const unsigned long long*)&_f2_70), "l"(*(const unsigned long long*)&_f2_71));
                    vals[22] = _mul_f32x2_35.x;
                    vals[23] = _mul_f32x2_35.y;
                    float2 _f2_72 = make_float2(_tmem_load_0[24], _tmem_load_0[25]);
                    float2 _f2_73 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_36;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_36) : "l"(*(const unsigned long long*)&_f2_72), "l"(*(const unsigned long long*)&_f2_73));
                    float2 _f2_74 = make_float2(_tmem_load_1[24], _tmem_load_1[25]);
                    float2 _f2_75 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_37;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_37) : "l"(*(const unsigned long long*)&_f2_74), "l"(*(const unsigned long long*)&_f2_75));
                    float g0_66 = _mul_f32x2_37.x;
                    float g1_67 = _mul_f32x2_37.y;
                    float _tanh_approx_48;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_48) : "f"(g0_66 * inv_beta));
                    float _exp2_24 = approx_exp2(g0_66 * -1.4426950408889634f);
                    float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                    float s0_68 = beta * _tanh_approx_48 * _rcp_24;
                    float _tanh_approx_49;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_49) : "f"(g1_67 * inv_beta));
                    float _exp2_25 = approx_exp2(g1_67 * -1.4426950408889634f);
                    float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                    float s1_69 = beta * _tanh_approx_49 * _rcp_25;
                    float _tanh_approx_50;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_50) : "f"(_mul_f32x2_36.x * inv_linear_beta));
                    float u0_70 = linear_beta * _tanh_approx_50;
                    float _tanh_approx_51;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_51) : "f"(_mul_f32x2_36.y * inv_linear_beta));
                    float u1_71 = linear_beta * _tanh_approx_51;
                    float2 _f2_76 = make_float2(u0_70, u1_71);
                    float2 _f2_77 = make_float2(s0_68, s1_69);
                    float2 _mul_f32x2_38;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_38) : "l"(*(const unsigned long long*)&_f2_76), "l"(*(const unsigned long long*)&_f2_77));
                    vals[24] = _mul_f32x2_38.x;
                    vals[25] = _mul_f32x2_38.y;
                    float2 _f2_78 = make_float2(_tmem_load_0[26], _tmem_load_0[27]);
                    float2 _f2_79 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_39;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_39) : "l"(*(const unsigned long long*)&_f2_78), "l"(*(const unsigned long long*)&_f2_79));
                    float2 _f2_80 = make_float2(_tmem_load_1[26], _tmem_load_1[27]);
                    float2 _f2_81 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_40;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_40) : "l"(*(const unsigned long long*)&_f2_80), "l"(*(const unsigned long long*)&_f2_81));
                    float g0_72 = _mul_f32x2_40.x;
                    float g1_73 = _mul_f32x2_40.y;
                    float _tanh_approx_52;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_52) : "f"(g0_72 * inv_beta));
                    float _exp2_26 = approx_exp2(g0_72 * -1.4426950408889634f);
                    float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                    float s0_74 = beta * _tanh_approx_52 * _rcp_26;
                    float _tanh_approx_53;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_53) : "f"(g1_73 * inv_beta));
                    float _exp2_27 = approx_exp2(g1_73 * -1.4426950408889634f);
                    float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                    float s1_75 = beta * _tanh_approx_53 * _rcp_27;
                    float _tanh_approx_54;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_54) : "f"(_mul_f32x2_39.x * inv_linear_beta));
                    float u0_76 = linear_beta * _tanh_approx_54;
                    float _tanh_approx_55;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_55) : "f"(_mul_f32x2_39.y * inv_linear_beta));
                    float u1_77 = linear_beta * _tanh_approx_55;
                    float2 _f2_82 = make_float2(u0_76, u1_77);
                    float2 _f2_83 = make_float2(s0_74, s1_75);
                    float2 _mul_f32x2_41;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_41) : "l"(*(const unsigned long long*)&_f2_82), "l"(*(const unsigned long long*)&_f2_83));
                    vals[26] = _mul_f32x2_41.x;
                    vals[27] = _mul_f32x2_41.y;
                    float2 _f2_84 = make_float2(_tmem_load_0[28], _tmem_load_0[29]);
                    float2 _f2_85 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_42;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_42) : "l"(*(const unsigned long long*)&_f2_84), "l"(*(const unsigned long long*)&_f2_85));
                    float2 _f2_86 = make_float2(_tmem_load_1[28], _tmem_load_1[29]);
                    float2 _f2_87 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_43;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_43) : "l"(*(const unsigned long long*)&_f2_86), "l"(*(const unsigned long long*)&_f2_87));
                    float g0_78 = _mul_f32x2_43.x;
                    float g1_79 = _mul_f32x2_43.y;
                    float _tanh_approx_56;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_56) : "f"(g0_78 * inv_beta));
                    float _exp2_28 = approx_exp2(g0_78 * -1.4426950408889634f);
                    float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                    float s0_80 = beta * _tanh_approx_56 * _rcp_28;
                    float _tanh_approx_57;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_57) : "f"(g1_79 * inv_beta));
                    float _exp2_29 = approx_exp2(g1_79 * -1.4426950408889634f);
                    float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                    float s1_81 = beta * _tanh_approx_57 * _rcp_29;
                    float _tanh_approx_58;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_58) : "f"(_mul_f32x2_42.x * inv_linear_beta));
                    float u0_82 = linear_beta * _tanh_approx_58;
                    float _tanh_approx_59;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_59) : "f"(_mul_f32x2_42.y * inv_linear_beta));
                    float u1_83 = linear_beta * _tanh_approx_59;
                    float2 _f2_88 = make_float2(u0_82, u1_83);
                    float2 _f2_89 = make_float2(s0_80, s1_81);
                    float2 _mul_f32x2_44;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_44) : "l"(*(const unsigned long long*)&_f2_88), "l"(*(const unsigned long long*)&_f2_89));
                    vals[28] = _mul_f32x2_44.x;
                    vals[29] = _mul_f32x2_44.y;
                    float2 _f2_90 = make_float2(_tmem_load_0[30], _tmem_load_0[31]);
                    float2 _f2_91 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_45;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_45) : "l"(*(const unsigned long long*)&_f2_90), "l"(*(const unsigned long long*)&_f2_91));
                    float2 _f2_92 = make_float2(_tmem_load_1[30], _tmem_load_1[31]);
                    float2 _f2_93 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_46;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_46) : "l"(*(const unsigned long long*)&_f2_92), "l"(*(const unsigned long long*)&_f2_93));
                    float g0_84 = _mul_f32x2_46.x;
                    float g1_85 = _mul_f32x2_46.y;
                    float _tanh_approx_60;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_60) : "f"(g0_84 * inv_beta));
                    float _exp2_30 = approx_exp2(g0_84 * -1.4426950408889634f);
                    float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                    float s0_86 = beta * _tanh_approx_60 * _rcp_30;
                    float _tanh_approx_61;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_61) : "f"(g1_85 * inv_beta));
                    float _exp2_31 = approx_exp2(g1_85 * -1.4426950408889634f);
                    float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                    float s1_87 = beta * _tanh_approx_61 * _rcp_31;
                    float _tanh_approx_62;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_62) : "f"(_mul_f32x2_45.x * inv_linear_beta));
                    float u0_88 = linear_beta * _tanh_approx_62;
                    float _tanh_approx_63;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_63) : "f"(_mul_f32x2_45.y * inv_linear_beta));
                    float u1_89 = linear_beta * _tanh_approx_63;
                    float2 _f2_94 = make_float2(u0_88, u1_89);
                    float2 _f2_95 = make_float2(s0_86, s1_87);
                    float2 _mul_f32x2_47;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_47) : "l"(*(const unsigned long long*)&_f2_94), "l"(*(const unsigned long long*)&_f2_95));
                    vals[30] = _mul_f32x2_47.x;
                    vals[31] = _mul_f32x2_47.y;
                    float2 _f2_96 = make_float2(_tmem_load_0[32], _tmem_load_0[33]);
                    float2 _f2_97 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_48;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_48) : "l"(*(const unsigned long long*)&_f2_96), "l"(*(const unsigned long long*)&_f2_97));
                    float2 _f2_98 = make_float2(_tmem_load_1[32], _tmem_load_1[33]);
                    float2 _f2_99 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_49;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_49) : "l"(*(const unsigned long long*)&_f2_98), "l"(*(const unsigned long long*)&_f2_99));
                    float g0_90 = _mul_f32x2_49.x;
                    float g1_91 = _mul_f32x2_49.y;
                    float _tanh_approx_64;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_64) : "f"(g0_90 * inv_beta));
                    float _exp2_32 = approx_exp2(g0_90 * -1.4426950408889634f);
                    float _rcp_32 = approx_rcp(1.0f + _exp2_32);
                    float s0_92 = beta * _tanh_approx_64 * _rcp_32;
                    float _tanh_approx_65;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_65) : "f"(g1_91 * inv_beta));
                    float _exp2_33 = approx_exp2(g1_91 * -1.4426950408889634f);
                    float _rcp_33 = approx_rcp(1.0f + _exp2_33);
                    float s1_93 = beta * _tanh_approx_65 * _rcp_33;
                    float _tanh_approx_66;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_66) : "f"(_mul_f32x2_48.x * inv_linear_beta));
                    float u0_94 = linear_beta * _tanh_approx_66;
                    float _tanh_approx_67;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_67) : "f"(_mul_f32x2_48.y * inv_linear_beta));
                    float u1_95 = linear_beta * _tanh_approx_67;
                    float2 _f2_100 = make_float2(u0_94, u1_95);
                    float2 _f2_101 = make_float2(s0_92, s1_93);
                    float2 _mul_f32x2_50;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_50) : "l"(*(const unsigned long long*)&_f2_100), "l"(*(const unsigned long long*)&_f2_101));
                    vals[32] = _mul_f32x2_50.x;
                    vals[33] = _mul_f32x2_50.y;
                    float2 _f2_102 = make_float2(_tmem_load_0[34], _tmem_load_0[35]);
                    float2 _f2_103 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_51;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_51) : "l"(*(const unsigned long long*)&_f2_102), "l"(*(const unsigned long long*)&_f2_103));
                    float2 _f2_104 = make_float2(_tmem_load_1[34], _tmem_load_1[35]);
                    float2 _f2_105 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_52;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_52) : "l"(*(const unsigned long long*)&_f2_104), "l"(*(const unsigned long long*)&_f2_105));
                    float g0_96 = _mul_f32x2_52.x;
                    float g1_97 = _mul_f32x2_52.y;
                    float _tanh_approx_68;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_68) : "f"(g0_96 * inv_beta));
                    float _exp2_34 = approx_exp2(g0_96 * -1.4426950408889634f);
                    float _rcp_34 = approx_rcp(1.0f + _exp2_34);
                    float s0_98 = beta * _tanh_approx_68 * _rcp_34;
                    float _tanh_approx_69;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_69) : "f"(g1_97 * inv_beta));
                    float _exp2_35 = approx_exp2(g1_97 * -1.4426950408889634f);
                    float _rcp_35 = approx_rcp(1.0f + _exp2_35);
                    float s1_99 = beta * _tanh_approx_69 * _rcp_35;
                    float _tanh_approx_70;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_70) : "f"(_mul_f32x2_51.x * inv_linear_beta));
                    float u0_100 = linear_beta * _tanh_approx_70;
                    float _tanh_approx_71;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_71) : "f"(_mul_f32x2_51.y * inv_linear_beta));
                    float u1_101 = linear_beta * _tanh_approx_71;
                    float2 _f2_106 = make_float2(u0_100, u1_101);
                    float2 _f2_107 = make_float2(s0_98, s1_99);
                    float2 _mul_f32x2_53;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_53) : "l"(*(const unsigned long long*)&_f2_106), "l"(*(const unsigned long long*)&_f2_107));
                    vals[34] = _mul_f32x2_53.x;
                    vals[35] = _mul_f32x2_53.y;
                    float2 _f2_108 = make_float2(_tmem_load_0[36], _tmem_load_0[37]);
                    float2 _f2_109 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_54;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_54) : "l"(*(const unsigned long long*)&_f2_108), "l"(*(const unsigned long long*)&_f2_109));
                    float2 _f2_110 = make_float2(_tmem_load_1[36], _tmem_load_1[37]);
                    float2 _f2_111 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_55;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_55) : "l"(*(const unsigned long long*)&_f2_110), "l"(*(const unsigned long long*)&_f2_111));
                    float g0_102 = _mul_f32x2_55.x;
                    float g1_103 = _mul_f32x2_55.y;
                    float _tanh_approx_72;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_72) : "f"(g0_102 * inv_beta));
                    float _exp2_36 = approx_exp2(g0_102 * -1.4426950408889634f);
                    float _rcp_36 = approx_rcp(1.0f + _exp2_36);
                    float s0_104 = beta * _tanh_approx_72 * _rcp_36;
                    float _tanh_approx_73;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_73) : "f"(g1_103 * inv_beta));
                    float _exp2_37 = approx_exp2(g1_103 * -1.4426950408889634f);
                    float _rcp_37 = approx_rcp(1.0f + _exp2_37);
                    float s1_105 = beta * _tanh_approx_73 * _rcp_37;
                    float _tanh_approx_74;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_74) : "f"(_mul_f32x2_54.x * inv_linear_beta));
                    float u0_106 = linear_beta * _tanh_approx_74;
                    float _tanh_approx_75;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_75) : "f"(_mul_f32x2_54.y * inv_linear_beta));
                    float u1_107 = linear_beta * _tanh_approx_75;
                    float2 _f2_112 = make_float2(u0_106, u1_107);
                    float2 _f2_113 = make_float2(s0_104, s1_105);
                    float2 _mul_f32x2_56;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_56) : "l"(*(const unsigned long long*)&_f2_112), "l"(*(const unsigned long long*)&_f2_113));
                    vals[36] = _mul_f32x2_56.x;
                    vals[37] = _mul_f32x2_56.y;
                    float2 _f2_114 = make_float2(_tmem_load_0[38], _tmem_load_0[39]);
                    float2 _f2_115 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_57;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_57) : "l"(*(const unsigned long long*)&_f2_114), "l"(*(const unsigned long long*)&_f2_115));
                    float2 _f2_116 = make_float2(_tmem_load_1[38], _tmem_load_1[39]);
                    float2 _f2_117 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_58;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_58) : "l"(*(const unsigned long long*)&_f2_116), "l"(*(const unsigned long long*)&_f2_117));
                    float g0_108 = _mul_f32x2_58.x;
                    float g1_109 = _mul_f32x2_58.y;
                    float _tanh_approx_76;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_76) : "f"(g0_108 * inv_beta));
                    float _exp2_38 = approx_exp2(g0_108 * -1.4426950408889634f);
                    float _rcp_38 = approx_rcp(1.0f + _exp2_38);
                    float s0_110 = beta * _tanh_approx_76 * _rcp_38;
                    float _tanh_approx_77;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_77) : "f"(g1_109 * inv_beta));
                    float _exp2_39 = approx_exp2(g1_109 * -1.4426950408889634f);
                    float _rcp_39 = approx_rcp(1.0f + _exp2_39);
                    float s1_111 = beta * _tanh_approx_77 * _rcp_39;
                    float _tanh_approx_78;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_78) : "f"(_mul_f32x2_57.x * inv_linear_beta));
                    float u0_112 = linear_beta * _tanh_approx_78;
                    float _tanh_approx_79;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_79) : "f"(_mul_f32x2_57.y * inv_linear_beta));
                    float u1_113 = linear_beta * _tanh_approx_79;
                    float2 _f2_118 = make_float2(u0_112, u1_113);
                    float2 _f2_119 = make_float2(s0_110, s1_111);
                    float2 _mul_f32x2_59;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_59) : "l"(*(const unsigned long long*)&_f2_118), "l"(*(const unsigned long long*)&_f2_119));
                    vals[38] = _mul_f32x2_59.x;
                    vals[39] = _mul_f32x2_59.y;
                    float2 _f2_120 = make_float2(_tmem_load_0[40], _tmem_load_0[41]);
                    float2 _f2_121 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_60;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_60) : "l"(*(const unsigned long long*)&_f2_120), "l"(*(const unsigned long long*)&_f2_121));
                    float2 _f2_122 = make_float2(_tmem_load_1[40], _tmem_load_1[41]);
                    float2 _f2_123 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_61;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_61) : "l"(*(const unsigned long long*)&_f2_122), "l"(*(const unsigned long long*)&_f2_123));
                    float g0_114 = _mul_f32x2_61.x;
                    float g1_115 = _mul_f32x2_61.y;
                    float _tanh_approx_80;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_80) : "f"(g0_114 * inv_beta));
                    float _exp2_40 = approx_exp2(g0_114 * -1.4426950408889634f);
                    float _rcp_40 = approx_rcp(1.0f + _exp2_40);
                    float s0_116 = beta * _tanh_approx_80 * _rcp_40;
                    float _tanh_approx_81;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_81) : "f"(g1_115 * inv_beta));
                    float _exp2_41 = approx_exp2(g1_115 * -1.4426950408889634f);
                    float _rcp_41 = approx_rcp(1.0f + _exp2_41);
                    float s1_117 = beta * _tanh_approx_81 * _rcp_41;
                    float _tanh_approx_82;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_82) : "f"(_mul_f32x2_60.x * inv_linear_beta));
                    float u0_118 = linear_beta * _tanh_approx_82;
                    float _tanh_approx_83;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_83) : "f"(_mul_f32x2_60.y * inv_linear_beta));
                    float u1_119 = linear_beta * _tanh_approx_83;
                    float2 _f2_124 = make_float2(u0_118, u1_119);
                    float2 _f2_125 = make_float2(s0_116, s1_117);
                    float2 _mul_f32x2_62;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_62) : "l"(*(const unsigned long long*)&_f2_124), "l"(*(const unsigned long long*)&_f2_125));
                    vals[40] = _mul_f32x2_62.x;
                    vals[41] = _mul_f32x2_62.y;
                    float2 _f2_126 = make_float2(_tmem_load_0[42], _tmem_load_0[43]);
                    float2 _f2_127 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_63;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_63) : "l"(*(const unsigned long long*)&_f2_126), "l"(*(const unsigned long long*)&_f2_127));
                    float2 _f2_128 = make_float2(_tmem_load_1[42], _tmem_load_1[43]);
                    float2 _f2_129 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_64;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_64) : "l"(*(const unsigned long long*)&_f2_128), "l"(*(const unsigned long long*)&_f2_129));
                    float g0_120 = _mul_f32x2_64.x;
                    float g1_121 = _mul_f32x2_64.y;
                    float _tanh_approx_84;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_84) : "f"(g0_120 * inv_beta));
                    float _exp2_42 = approx_exp2(g0_120 * -1.4426950408889634f);
                    float _rcp_42 = approx_rcp(1.0f + _exp2_42);
                    float s0_122 = beta * _tanh_approx_84 * _rcp_42;
                    float _tanh_approx_85;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_85) : "f"(g1_121 * inv_beta));
                    float _exp2_43 = approx_exp2(g1_121 * -1.4426950408889634f);
                    float _rcp_43 = approx_rcp(1.0f + _exp2_43);
                    float s1_123 = beta * _tanh_approx_85 * _rcp_43;
                    float _tanh_approx_86;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_86) : "f"(_mul_f32x2_63.x * inv_linear_beta));
                    float u0_124 = linear_beta * _tanh_approx_86;
                    float _tanh_approx_87;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_87) : "f"(_mul_f32x2_63.y * inv_linear_beta));
                    float u1_125 = linear_beta * _tanh_approx_87;
                    float2 _f2_130 = make_float2(u0_124, u1_125);
                    float2 _f2_131 = make_float2(s0_122, s1_123);
                    float2 _mul_f32x2_65;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_65) : "l"(*(const unsigned long long*)&_f2_130), "l"(*(const unsigned long long*)&_f2_131));
                    vals[42] = _mul_f32x2_65.x;
                    vals[43] = _mul_f32x2_65.y;
                    float2 _f2_132 = make_float2(_tmem_load_0[44], _tmem_load_0[45]);
                    float2 _f2_133 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_66;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_66) : "l"(*(const unsigned long long*)&_f2_132), "l"(*(const unsigned long long*)&_f2_133));
                    float2 _f2_134 = make_float2(_tmem_load_1[44], _tmem_load_1[45]);
                    float2 _f2_135 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_67;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_67) : "l"(*(const unsigned long long*)&_f2_134), "l"(*(const unsigned long long*)&_f2_135));
                    float g0_126 = _mul_f32x2_67.x;
                    float g1_127 = _mul_f32x2_67.y;
                    float _tanh_approx_88;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_88) : "f"(g0_126 * inv_beta));
                    float _exp2_44 = approx_exp2(g0_126 * -1.4426950408889634f);
                    float _rcp_44 = approx_rcp(1.0f + _exp2_44);
                    float s0_128 = beta * _tanh_approx_88 * _rcp_44;
                    float _tanh_approx_89;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_89) : "f"(g1_127 * inv_beta));
                    float _exp2_45 = approx_exp2(g1_127 * -1.4426950408889634f);
                    float _rcp_45 = approx_rcp(1.0f + _exp2_45);
                    float s1_129 = beta * _tanh_approx_89 * _rcp_45;
                    float _tanh_approx_90;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_90) : "f"(_mul_f32x2_66.x * inv_linear_beta));
                    float u0_130 = linear_beta * _tanh_approx_90;
                    float _tanh_approx_91;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_91) : "f"(_mul_f32x2_66.y * inv_linear_beta));
                    float u1_131 = linear_beta * _tanh_approx_91;
                    float2 _f2_136 = make_float2(u0_130, u1_131);
                    float2 _f2_137 = make_float2(s0_128, s1_129);
                    float2 _mul_f32x2_68;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_68) : "l"(*(const unsigned long long*)&_f2_136), "l"(*(const unsigned long long*)&_f2_137));
                    vals[44] = _mul_f32x2_68.x;
                    vals[45] = _mul_f32x2_68.y;
                    float2 _f2_138 = make_float2(_tmem_load_0[46], _tmem_load_0[47]);
                    float2 _f2_139 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_69;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_69) : "l"(*(const unsigned long long*)&_f2_138), "l"(*(const unsigned long long*)&_f2_139));
                    float2 _f2_140 = make_float2(_tmem_load_1[46], _tmem_load_1[47]);
                    float2 _f2_141 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_70;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_70) : "l"(*(const unsigned long long*)&_f2_140), "l"(*(const unsigned long long*)&_f2_141));
                    float g0_132 = _mul_f32x2_70.x;
                    float g1_133 = _mul_f32x2_70.y;
                    float _tanh_approx_92;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_92) : "f"(g0_132 * inv_beta));
                    float _exp2_46 = approx_exp2(g0_132 * -1.4426950408889634f);
                    float _rcp_46 = approx_rcp(1.0f + _exp2_46);
                    float s0_134 = beta * _tanh_approx_92 * _rcp_46;
                    float _tanh_approx_93;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_93) : "f"(g1_133 * inv_beta));
                    float _exp2_47 = approx_exp2(g1_133 * -1.4426950408889634f);
                    float _rcp_47 = approx_rcp(1.0f + _exp2_47);
                    float s1_135 = beta * _tanh_approx_93 * _rcp_47;
                    float _tanh_approx_94;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_94) : "f"(_mul_f32x2_69.x * inv_linear_beta));
                    float u0_136 = linear_beta * _tanh_approx_94;
                    float _tanh_approx_95;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_95) : "f"(_mul_f32x2_69.y * inv_linear_beta));
                    float u1_137 = linear_beta * _tanh_approx_95;
                    float2 _f2_142 = make_float2(u0_136, u1_137);
                    float2 _f2_143 = make_float2(s0_134, s1_135);
                    float2 _mul_f32x2_71;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_71) : "l"(*(const unsigned long long*)&_f2_142), "l"(*(const unsigned long long*)&_f2_143));
                    vals[46] = _mul_f32x2_71.x;
                    vals[47] = _mul_f32x2_71.y;
                    float2 _f2_144 = make_float2(_tmem_load_0[48], _tmem_load_0[49]);
                    float2 _f2_145 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_72;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_72) : "l"(*(const unsigned long long*)&_f2_144), "l"(*(const unsigned long long*)&_f2_145));
                    float2 _f2_146 = make_float2(_tmem_load_1[48], _tmem_load_1[49]);
                    float2 _f2_147 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_73;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_73) : "l"(*(const unsigned long long*)&_f2_146), "l"(*(const unsigned long long*)&_f2_147));
                    float g0_138 = _mul_f32x2_73.x;
                    float g1_139 = _mul_f32x2_73.y;
                    float _tanh_approx_96;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_96) : "f"(g0_138 * inv_beta));
                    float _exp2_48 = approx_exp2(g0_138 * -1.4426950408889634f);
                    float _rcp_48 = approx_rcp(1.0f + _exp2_48);
                    float s0_140 = beta * _tanh_approx_96 * _rcp_48;
                    float _tanh_approx_97;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_97) : "f"(g1_139 * inv_beta));
                    float _exp2_49 = approx_exp2(g1_139 * -1.4426950408889634f);
                    float _rcp_49 = approx_rcp(1.0f + _exp2_49);
                    float s1_141 = beta * _tanh_approx_97 * _rcp_49;
                    float _tanh_approx_98;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_98) : "f"(_mul_f32x2_72.x * inv_linear_beta));
                    float u0_142 = linear_beta * _tanh_approx_98;
                    float _tanh_approx_99;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_99) : "f"(_mul_f32x2_72.y * inv_linear_beta));
                    float u1_143 = linear_beta * _tanh_approx_99;
                    float2 _f2_148 = make_float2(u0_142, u1_143);
                    float2 _f2_149 = make_float2(s0_140, s1_141);
                    float2 _mul_f32x2_74;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_74) : "l"(*(const unsigned long long*)&_f2_148), "l"(*(const unsigned long long*)&_f2_149));
                    vals[48] = _mul_f32x2_74.x;
                    vals[49] = _mul_f32x2_74.y;
                    float2 _f2_150 = make_float2(_tmem_load_0[50], _tmem_load_0[51]);
                    float2 _f2_151 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_75;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_75) : "l"(*(const unsigned long long*)&_f2_150), "l"(*(const unsigned long long*)&_f2_151));
                    float2 _f2_152 = make_float2(_tmem_load_1[50], _tmem_load_1[51]);
                    float2 _f2_153 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_76;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_76) : "l"(*(const unsigned long long*)&_f2_152), "l"(*(const unsigned long long*)&_f2_153));
                    float g0_144 = _mul_f32x2_76.x;
                    float g1_145 = _mul_f32x2_76.y;
                    float _tanh_approx_100;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_100) : "f"(g0_144 * inv_beta));
                    float _exp2_50 = approx_exp2(g0_144 * -1.4426950408889634f);
                    float _rcp_50 = approx_rcp(1.0f + _exp2_50);
                    float s0_146 = beta * _tanh_approx_100 * _rcp_50;
                    float _tanh_approx_101;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_101) : "f"(g1_145 * inv_beta));
                    float _exp2_51 = approx_exp2(g1_145 * -1.4426950408889634f);
                    float _rcp_51 = approx_rcp(1.0f + _exp2_51);
                    float s1_147 = beta * _tanh_approx_101 * _rcp_51;
                    float _tanh_approx_102;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_102) : "f"(_mul_f32x2_75.x * inv_linear_beta));
                    float u0_148 = linear_beta * _tanh_approx_102;
                    float _tanh_approx_103;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_103) : "f"(_mul_f32x2_75.y * inv_linear_beta));
                    float u1_149 = linear_beta * _tanh_approx_103;
                    float2 _f2_154 = make_float2(u0_148, u1_149);
                    float2 _f2_155 = make_float2(s0_146, s1_147);
                    float2 _mul_f32x2_77;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_77) : "l"(*(const unsigned long long*)&_f2_154), "l"(*(const unsigned long long*)&_f2_155));
                    vals[50] = _mul_f32x2_77.x;
                    vals[51] = _mul_f32x2_77.y;
                    float2 _f2_156 = make_float2(_tmem_load_0[52], _tmem_load_0[53]);
                    float2 _f2_157 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_78;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_78) : "l"(*(const unsigned long long*)&_f2_156), "l"(*(const unsigned long long*)&_f2_157));
                    float2 _f2_158 = make_float2(_tmem_load_1[52], _tmem_load_1[53]);
                    float2 _f2_159 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_79;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_79) : "l"(*(const unsigned long long*)&_f2_158), "l"(*(const unsigned long long*)&_f2_159));
                    float g0_150 = _mul_f32x2_79.x;
                    float g1_151 = _mul_f32x2_79.y;
                    float _tanh_approx_104;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_104) : "f"(g0_150 * inv_beta));
                    float _exp2_52 = approx_exp2(g0_150 * -1.4426950408889634f);
                    float _rcp_52 = approx_rcp(1.0f + _exp2_52);
                    float s0_152 = beta * _tanh_approx_104 * _rcp_52;
                    float _tanh_approx_105;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_105) : "f"(g1_151 * inv_beta));
                    float _exp2_53 = approx_exp2(g1_151 * -1.4426950408889634f);
                    float _rcp_53 = approx_rcp(1.0f + _exp2_53);
                    float s1_153 = beta * _tanh_approx_105 * _rcp_53;
                    float _tanh_approx_106;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_106) : "f"(_mul_f32x2_78.x * inv_linear_beta));
                    float u0_154 = linear_beta * _tanh_approx_106;
                    float _tanh_approx_107;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_107) : "f"(_mul_f32x2_78.y * inv_linear_beta));
                    float u1_155 = linear_beta * _tanh_approx_107;
                    float2 _f2_160 = make_float2(u0_154, u1_155);
                    float2 _f2_161 = make_float2(s0_152, s1_153);
                    float2 _mul_f32x2_80;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_80) : "l"(*(const unsigned long long*)&_f2_160), "l"(*(const unsigned long long*)&_f2_161));
                    vals[52] = _mul_f32x2_80.x;
                    vals[53] = _mul_f32x2_80.y;
                    float2 _f2_162 = make_float2(_tmem_load_0[54], _tmem_load_0[55]);
                    float2 _f2_163 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_81;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_81) : "l"(*(const unsigned long long*)&_f2_162), "l"(*(const unsigned long long*)&_f2_163));
                    float2 _f2_164 = make_float2(_tmem_load_1[54], _tmem_load_1[55]);
                    float2 _f2_165 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_82;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_82) : "l"(*(const unsigned long long*)&_f2_164), "l"(*(const unsigned long long*)&_f2_165));
                    float g0_156 = _mul_f32x2_82.x;
                    float g1_157 = _mul_f32x2_82.y;
                    float _tanh_approx_108;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_108) : "f"(g0_156 * inv_beta));
                    float _exp2_54 = approx_exp2(g0_156 * -1.4426950408889634f);
                    float _rcp_54 = approx_rcp(1.0f + _exp2_54);
                    float s0_158 = beta * _tanh_approx_108 * _rcp_54;
                    float _tanh_approx_109;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_109) : "f"(g1_157 * inv_beta));
                    float _exp2_55 = approx_exp2(g1_157 * -1.4426950408889634f);
                    float _rcp_55 = approx_rcp(1.0f + _exp2_55);
                    float s1_159 = beta * _tanh_approx_109 * _rcp_55;
                    float _tanh_approx_110;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_110) : "f"(_mul_f32x2_81.x * inv_linear_beta));
                    float u0_160 = linear_beta * _tanh_approx_110;
                    float _tanh_approx_111;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_111) : "f"(_mul_f32x2_81.y * inv_linear_beta));
                    float u1_161 = linear_beta * _tanh_approx_111;
                    float2 _f2_166 = make_float2(u0_160, u1_161);
                    float2 _f2_167 = make_float2(s0_158, s1_159);
                    float2 _mul_f32x2_83;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_83) : "l"(*(const unsigned long long*)&_f2_166), "l"(*(const unsigned long long*)&_f2_167));
                    vals[54] = _mul_f32x2_83.x;
                    vals[55] = _mul_f32x2_83.y;
                    float2 _f2_168 = make_float2(_tmem_load_0[56], _tmem_load_0[57]);
                    float2 _f2_169 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_84;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_84) : "l"(*(const unsigned long long*)&_f2_168), "l"(*(const unsigned long long*)&_f2_169));
                    float2 _f2_170 = make_float2(_tmem_load_1[56], _tmem_load_1[57]);
                    float2 _f2_171 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_85;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_85) : "l"(*(const unsigned long long*)&_f2_170), "l"(*(const unsigned long long*)&_f2_171));
                    float g0_162 = _mul_f32x2_85.x;
                    float g1_163 = _mul_f32x2_85.y;
                    float _tanh_approx_112;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_112) : "f"(g0_162 * inv_beta));
                    float _exp2_56 = approx_exp2(g0_162 * -1.4426950408889634f);
                    float _rcp_56 = approx_rcp(1.0f + _exp2_56);
                    float s0_164 = beta * _tanh_approx_112 * _rcp_56;
                    float _tanh_approx_113;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_113) : "f"(g1_163 * inv_beta));
                    float _exp2_57 = approx_exp2(g1_163 * -1.4426950408889634f);
                    float _rcp_57 = approx_rcp(1.0f + _exp2_57);
                    float s1_165 = beta * _tanh_approx_113 * _rcp_57;
                    float _tanh_approx_114;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_114) : "f"(_mul_f32x2_84.x * inv_linear_beta));
                    float u0_166 = linear_beta * _tanh_approx_114;
                    float _tanh_approx_115;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_115) : "f"(_mul_f32x2_84.y * inv_linear_beta));
                    float u1_167 = linear_beta * _tanh_approx_115;
                    float2 _f2_172 = make_float2(u0_166, u1_167);
                    float2 _f2_173 = make_float2(s0_164, s1_165);
                    float2 _mul_f32x2_86;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_86) : "l"(*(const unsigned long long*)&_f2_172), "l"(*(const unsigned long long*)&_f2_173));
                    vals[56] = _mul_f32x2_86.x;
                    vals[57] = _mul_f32x2_86.y;
                    float2 _f2_174 = make_float2(_tmem_load_0[58], _tmem_load_0[59]);
                    float2 _f2_175 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_87;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_87) : "l"(*(const unsigned long long*)&_f2_174), "l"(*(const unsigned long long*)&_f2_175));
                    float2 _f2_176 = make_float2(_tmem_load_1[58], _tmem_load_1[59]);
                    float2 _f2_177 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_88;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_88) : "l"(*(const unsigned long long*)&_f2_176), "l"(*(const unsigned long long*)&_f2_177));
                    float g0_168 = _mul_f32x2_88.x;
                    float g1_169 = _mul_f32x2_88.y;
                    float _tanh_approx_116;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_116) : "f"(g0_168 * inv_beta));
                    float _exp2_58 = approx_exp2(g0_168 * -1.4426950408889634f);
                    float _rcp_58 = approx_rcp(1.0f + _exp2_58);
                    float s0_170 = beta * _tanh_approx_116 * _rcp_58;
                    float _tanh_approx_117;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_117) : "f"(g1_169 * inv_beta));
                    float _exp2_59 = approx_exp2(g1_169 * -1.4426950408889634f);
                    float _rcp_59 = approx_rcp(1.0f + _exp2_59);
                    float s1_171 = beta * _tanh_approx_117 * _rcp_59;
                    float _tanh_approx_118;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_118) : "f"(_mul_f32x2_87.x * inv_linear_beta));
                    float u0_172 = linear_beta * _tanh_approx_118;
                    float _tanh_approx_119;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_119) : "f"(_mul_f32x2_87.y * inv_linear_beta));
                    float u1_173 = linear_beta * _tanh_approx_119;
                    float2 _f2_178 = make_float2(u0_172, u1_173);
                    float2 _f2_179 = make_float2(s0_170, s1_171);
                    float2 _mul_f32x2_89;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_89) : "l"(*(const unsigned long long*)&_f2_178), "l"(*(const unsigned long long*)&_f2_179));
                    vals[58] = _mul_f32x2_89.x;
                    vals[59] = _mul_f32x2_89.y;
                    float2 _f2_180 = make_float2(_tmem_load_0[60], _tmem_load_0[61]);
                    float2 _f2_181 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_90;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_90) : "l"(*(const unsigned long long*)&_f2_180), "l"(*(const unsigned long long*)&_f2_181));
                    float2 _f2_182 = make_float2(_tmem_load_1[60], _tmem_load_1[61]);
                    float2 _f2_183 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_91;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_91) : "l"(*(const unsigned long long*)&_f2_182), "l"(*(const unsigned long long*)&_f2_183));
                    float g0_174 = _mul_f32x2_91.x;
                    float g1_175 = _mul_f32x2_91.y;
                    float _tanh_approx_120;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_120) : "f"(g0_174 * inv_beta));
                    float _exp2_60 = approx_exp2(g0_174 * -1.4426950408889634f);
                    float _rcp_60 = approx_rcp(1.0f + _exp2_60);
                    float s0_176 = beta * _tanh_approx_120 * _rcp_60;
                    float _tanh_approx_121;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_121) : "f"(g1_175 * inv_beta));
                    float _exp2_61 = approx_exp2(g1_175 * -1.4426950408889634f);
                    float _rcp_61 = approx_rcp(1.0f + _exp2_61);
                    float s1_177 = beta * _tanh_approx_121 * _rcp_61;
                    float _tanh_approx_122;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_122) : "f"(_mul_f32x2_90.x * inv_linear_beta));
                    float u0_178 = linear_beta * _tanh_approx_122;
                    float _tanh_approx_123;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_123) : "f"(_mul_f32x2_90.y * inv_linear_beta));
                    float u1_179 = linear_beta * _tanh_approx_123;
                    float2 _f2_184 = make_float2(u0_178, u1_179);
                    float2 _f2_185 = make_float2(s0_176, s1_177);
                    float2 _mul_f32x2_92;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_92) : "l"(*(const unsigned long long*)&_f2_184), "l"(*(const unsigned long long*)&_f2_185));
                    vals[60] = _mul_f32x2_92.x;
                    vals[61] = _mul_f32x2_92.y;
                    float2 _f2_186 = make_float2(_tmem_load_0[62], _tmem_load_0[63]);
                    float2 _f2_187 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_93;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_93) : "l"(*(const unsigned long long*)&_f2_186), "l"(*(const unsigned long long*)&_f2_187));
                    float2 _f2_188 = make_float2(_tmem_load_1[62], _tmem_load_1[63]);
                    float2 _f2_189 = make_float2(alpha_val, alpha_val);
                    float2 _mul_f32x2_94;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_94) : "l"(*(const unsigned long long*)&_f2_188), "l"(*(const unsigned long long*)&_f2_189));
                    float g0_180 = _mul_f32x2_94.x;
                    float g1_181 = _mul_f32x2_94.y;
                    float _tanh_approx_124;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_124) : "f"(g0_180 * inv_beta));
                    float _exp2_62 = approx_exp2(g0_180 * -1.4426950408889634f);
                    float _rcp_62 = approx_rcp(1.0f + _exp2_62);
                    float s0_182 = beta * _tanh_approx_124 * _rcp_62;
                    float _tanh_approx_125;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_125) : "f"(g1_181 * inv_beta));
                    float _exp2_63 = approx_exp2(g1_181 * -1.4426950408889634f);
                    float _rcp_63 = approx_rcp(1.0f + _exp2_63);
                    float s1_183 = beta * _tanh_approx_125 * _rcp_63;
                    float _tanh_approx_126;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_126) : "f"(_mul_f32x2_93.x * inv_linear_beta));
                    float u0_184 = linear_beta * _tanh_approx_126;
                    float _tanh_approx_127;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_127) : "f"(_mul_f32x2_93.y * inv_linear_beta));
                    float u1_185 = linear_beta * _tanh_approx_127;
                    float2 _f2_190 = make_float2(u0_184, u1_185);
                    float2 _f2_191 = make_float2(s0_182, s1_183);
                    float2 _mul_f32x2_95;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_95) : "l"(*(const unsigned long long*)&_f2_190), "l"(*(const unsigned long long*)&_f2_191));
                    vals[62] = _mul_f32x2_95.x;
                    vals[63] = _mul_f32x2_95.y;
                    float amax0 = 0.0f;
                    float amax1 = 0.0f;
                    float _fabs_0 = fabsf(vals[0]);
                    float _fmax_0 = fmaxf(amax0, _fabs_0);
                    amax0 = _fmax_0;
                    float _fabs_1 = fabsf(vals[32]);
                    float _fmax_1 = fmaxf(amax1, _fabs_1);
                    amax1 = _fmax_1;
                    float _fabs_2 = fabsf(vals[1]);
                    float _fmax_2 = fmaxf(amax0, _fabs_2);
                    amax0 = _fmax_2;
                    float _fabs_3 = fabsf(vals[33]);
                    float _fmax_3 = fmaxf(amax1, _fabs_3);
                    amax1 = _fmax_3;
                    float _fabs_4 = fabsf(vals[2]);
                    float _fmax_4 = fmaxf(amax0, _fabs_4);
                    amax0 = _fmax_4;
                    float _fabs_5 = fabsf(vals[34]);
                    float _fmax_5 = fmaxf(amax1, _fabs_5);
                    amax1 = _fmax_5;
                    float _fabs_6 = fabsf(vals[3]);
                    float _fmax_6 = fmaxf(amax0, _fabs_6);
                    amax0 = _fmax_6;
                    float _fabs_7 = fabsf(vals[35]);
                    float _fmax_7 = fmaxf(amax1, _fabs_7);
                    amax1 = _fmax_7;
                    float _fabs_8 = fabsf(vals[4]);
                    float _fmax_8 = fmaxf(amax0, _fabs_8);
                    amax0 = _fmax_8;
                    float _fabs_9 = fabsf(vals[36]);
                    float _fmax_9 = fmaxf(amax1, _fabs_9);
                    amax1 = _fmax_9;
                    float _fabs_10 = fabsf(vals[5]);
                    float _fmax_10 = fmaxf(amax0, _fabs_10);
                    amax0 = _fmax_10;
                    float _fabs_11 = fabsf(vals[37]);
                    float _fmax_11 = fmaxf(amax1, _fabs_11);
                    amax1 = _fmax_11;
                    float _fabs_12 = fabsf(vals[6]);
                    float _fmax_12 = fmaxf(amax0, _fabs_12);
                    amax0 = _fmax_12;
                    float _fabs_13 = fabsf(vals[38]);
                    float _fmax_13 = fmaxf(amax1, _fabs_13);
                    amax1 = _fmax_13;
                    float _fabs_14 = fabsf(vals[7]);
                    float _fmax_14 = fmaxf(amax0, _fabs_14);
                    amax0 = _fmax_14;
                    float _fabs_15 = fabsf(vals[39]);
                    float _fmax_15 = fmaxf(amax1, _fabs_15);
                    amax1 = _fmax_15;
                    float _fabs_16 = fabsf(vals[8]);
                    float _fmax_16 = fmaxf(amax0, _fabs_16);
                    amax0 = _fmax_16;
                    float _fabs_17 = fabsf(vals[40]);
                    float _fmax_17 = fmaxf(amax1, _fabs_17);
                    amax1 = _fmax_17;
                    float _fabs_18 = fabsf(vals[9]);
                    float _fmax_18 = fmaxf(amax0, _fabs_18);
                    amax0 = _fmax_18;
                    float _fabs_19 = fabsf(vals[41]);
                    float _fmax_19 = fmaxf(amax1, _fabs_19);
                    amax1 = _fmax_19;
                    float _fabs_20 = fabsf(vals[10]);
                    float _fmax_20 = fmaxf(amax0, _fabs_20);
                    amax0 = _fmax_20;
                    float _fabs_21 = fabsf(vals[42]);
                    float _fmax_21 = fmaxf(amax1, _fabs_21);
                    amax1 = _fmax_21;
                    float _fabs_22 = fabsf(vals[11]);
                    float _fmax_22 = fmaxf(amax0, _fabs_22);
                    amax0 = _fmax_22;
                    float _fabs_23 = fabsf(vals[43]);
                    float _fmax_23 = fmaxf(amax1, _fabs_23);
                    amax1 = _fmax_23;
                    float _fabs_24 = fabsf(vals[12]);
                    float _fmax_24 = fmaxf(amax0, _fabs_24);
                    amax0 = _fmax_24;
                    float _fabs_25 = fabsf(vals[44]);
                    float _fmax_25 = fmaxf(amax1, _fabs_25);
                    amax1 = _fmax_25;
                    float _fabs_26 = fabsf(vals[13]);
                    float _fmax_26 = fmaxf(amax0, _fabs_26);
                    amax0 = _fmax_26;
                    float _fabs_27 = fabsf(vals[45]);
                    float _fmax_27 = fmaxf(amax1, _fabs_27);
                    amax1 = _fmax_27;
                    float _fabs_28 = fabsf(vals[14]);
                    float _fmax_28 = fmaxf(amax0, _fabs_28);
                    amax0 = _fmax_28;
                    float _fabs_29 = fabsf(vals[46]);
                    float _fmax_29 = fmaxf(amax1, _fabs_29);
                    amax1 = _fmax_29;
                    float _fabs_30 = fabsf(vals[15]);
                    float _fmax_30 = fmaxf(amax0, _fabs_30);
                    amax0 = _fmax_30;
                    float _fabs_31 = fabsf(vals[47]);
                    float _fmax_31 = fmaxf(amax1, _fabs_31);
                    amax1 = _fmax_31;
                    float _fabs_32 = fabsf(vals[16]);
                    float _fmax_32 = fmaxf(amax0, _fabs_32);
                    amax0 = _fmax_32;
                    float _fabs_33 = fabsf(vals[48]);
                    float _fmax_33 = fmaxf(amax1, _fabs_33);
                    amax1 = _fmax_33;
                    float _fabs_34 = fabsf(vals[17]);
                    float _fmax_34 = fmaxf(amax0, _fabs_34);
                    amax0 = _fmax_34;
                    float _fabs_35 = fabsf(vals[49]);
                    float _fmax_35 = fmaxf(amax1, _fabs_35);
                    amax1 = _fmax_35;
                    float _fabs_36 = fabsf(vals[18]);
                    float _fmax_36 = fmaxf(amax0, _fabs_36);
                    amax0 = _fmax_36;
                    float _fabs_37 = fabsf(vals[50]);
                    float _fmax_37 = fmaxf(amax1, _fabs_37);
                    amax1 = _fmax_37;
                    float _fabs_38 = fabsf(vals[19]);
                    float _fmax_38 = fmaxf(amax0, _fabs_38);
                    amax0 = _fmax_38;
                    float _fabs_39 = fabsf(vals[51]);
                    float _fmax_39 = fmaxf(amax1, _fabs_39);
                    amax1 = _fmax_39;
                    float _fabs_40 = fabsf(vals[20]);
                    float _fmax_40 = fmaxf(amax0, _fabs_40);
                    amax0 = _fmax_40;
                    float _fabs_41 = fabsf(vals[52]);
                    float _fmax_41 = fmaxf(amax1, _fabs_41);
                    amax1 = _fmax_41;
                    float _fabs_42 = fabsf(vals[21]);
                    float _fmax_42 = fmaxf(amax0, _fabs_42);
                    amax0 = _fmax_42;
                    float _fabs_43 = fabsf(vals[53]);
                    float _fmax_43 = fmaxf(amax1, _fabs_43);
                    amax1 = _fmax_43;
                    float _fabs_44 = fabsf(vals[22]);
                    float _fmax_44 = fmaxf(amax0, _fabs_44);
                    amax0 = _fmax_44;
                    float _fabs_45 = fabsf(vals[54]);
                    float _fmax_45 = fmaxf(amax1, _fabs_45);
                    amax1 = _fmax_45;
                    float _fabs_46 = fabsf(vals[23]);
                    float _fmax_46 = fmaxf(amax0, _fabs_46);
                    amax0 = _fmax_46;
                    float _fabs_47 = fabsf(vals[55]);
                    float _fmax_47 = fmaxf(amax1, _fabs_47);
                    amax1 = _fmax_47;
                    float _fabs_48 = fabsf(vals[24]);
                    float _fmax_48 = fmaxf(amax0, _fabs_48);
                    amax0 = _fmax_48;
                    float _fabs_49 = fabsf(vals[56]);
                    float _fmax_49 = fmaxf(amax1, _fabs_49);
                    amax1 = _fmax_49;
                    float _fabs_50 = fabsf(vals[25]);
                    float _fmax_50 = fmaxf(amax0, _fabs_50);
                    amax0 = _fmax_50;
                    float _fabs_51 = fabsf(vals[57]);
                    float _fmax_51 = fmaxf(amax1, _fabs_51);
                    amax1 = _fmax_51;
                    float _fabs_52 = fabsf(vals[26]);
                    float _fmax_52 = fmaxf(amax0, _fabs_52);
                    amax0 = _fmax_52;
                    float _fabs_53 = fabsf(vals[58]);
                    float _fmax_53 = fmaxf(amax1, _fabs_53);
                    amax1 = _fmax_53;
                    float _fabs_54 = fabsf(vals[27]);
                    float _fmax_54 = fmaxf(amax0, _fabs_54);
                    amax0 = _fmax_54;
                    float _fabs_55 = fabsf(vals[59]);
                    float _fmax_55 = fmaxf(amax1, _fabs_55);
                    amax1 = _fmax_55;
                    float _fabs_56 = fabsf(vals[28]);
                    float _fmax_56 = fmaxf(amax0, _fabs_56);
                    amax0 = _fmax_56;
                    float _fabs_57 = fabsf(vals[60]);
                    float _fmax_57 = fmaxf(amax1, _fabs_57);
                    amax1 = _fmax_57;
                    float _fabs_58 = fabsf(vals[29]);
                    float _fmax_58 = fmaxf(amax0, _fabs_58);
                    amax0 = _fmax_58;
                    float _fabs_59 = fabsf(vals[61]);
                    float _fmax_59 = fmaxf(amax1, _fabs_59);
                    amax1 = _fmax_59;
                    float _fabs_60 = fabsf(vals[30]);
                    float _fmax_60 = fmaxf(amax0, _fabs_60);
                    amax0 = _fmax_60;
                    float _fabs_61 = fabsf(vals[62]);
                    float _fmax_61 = fmaxf(amax1, _fabs_61);
                    amax1 = _fmax_61;
                    float _fabs_62 = fabsf(vals[31]);
                    float _fmax_62 = fmaxf(amax0, _fabs_62);
                    amax0 = _fmax_62;
                    float _fabs_63 = fabsf(vals[63]);
                    float _fmax_63 = fmaxf(amax1, _fabs_63);
                    amax1 = _fmax_63;
                    float2 _f2_192 = make_float2(amax0, amax1);
                    float2 _f2_193 = make_float2(inv_fp8_max, inv_fp8_max);
                    float2 _mul_f32x2_96;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_96) : "l"(*(const unsigned long long*)&_f2_192), "l"(*(const unsigned long long*)&_f2_193));
                    float2 _f2_194 = make_float2(one_f32, one_f32);
                    float2 _mul_f32x2_97;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_97) : "l"(*(const unsigned long long*)&_mul_f32x2_96), "l"(*(const unsigned long long*)&_f2_194));
                    uint16_t _ue8m0x2_f32_0;
                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_0) : "f"(_mul_f32x2_97.y), "f"(_mul_f32x2_97.x));
                    int code_full = (int)_ue8m0x2_f32_0;
                    int code0 = code_full & 255;
                    int code1 = code_full >> 8 & 255;
                    int sf_off = sf_base + real * 2;
                    *(reinterpret_cast<unsigned char*>(CSF + sf_off) + (0)) = (unsigned char)((unsigned int)code0);
                    *(reinterpret_cast<unsigned char*>(CSF + (sf_off + 1)) + (0)) = (unsigned char)((unsigned int)code1);
                    int _max_0 = ((254 - code0) > (0) ? (254 - code0) : (0));
                    unsigned int inv0_bits = (unsigned int)(_max_0 << 23);
                    int _max_1 = ((254 - code1) > (0) ? (254 - code1) : (0));
                    unsigned int inv1_bits = (unsigned int)(_max_1 << 23);
                    float inv0 = __uint_as_float(inv0_bits) * (float)(code0 != 0);
                    float inv1 = __uint_as_float(inv1_bits) * (float)(code1 != 0);
                    float2 _f2_195 = make_float2(vals[0], vals[32]);
                    float2 _f2_196 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_98;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_98) : "l"(*(const unsigned long long*)&_f2_195), "l"(*(const unsigned long long*)&_f2_196));
                    vals[0] = _mul_f32x2_98.x;
                    vals[32] = _mul_f32x2_98.y;
                    float2 _f2_197 = make_float2(vals[1], vals[33]);
                    float2 _f2_198 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_99;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_99) : "l"(*(const unsigned long long*)&_f2_197), "l"(*(const unsigned long long*)&_f2_198));
                    vals[1] = _mul_f32x2_99.x;
                    vals[33] = _mul_f32x2_99.y;
                    float2 _f2_199 = make_float2(vals[2], vals[34]);
                    float2 _f2_200 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_100;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_100) : "l"(*(const unsigned long long*)&_f2_199), "l"(*(const unsigned long long*)&_f2_200));
                    vals[2] = _mul_f32x2_100.x;
                    vals[34] = _mul_f32x2_100.y;
                    float2 _f2_201 = make_float2(vals[3], vals[35]);
                    float2 _f2_202 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_101;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_101) : "l"(*(const unsigned long long*)&_f2_201), "l"(*(const unsigned long long*)&_f2_202));
                    vals[3] = _mul_f32x2_101.x;
                    vals[35] = _mul_f32x2_101.y;
                    float2 _f2_203 = make_float2(vals[4], vals[36]);
                    float2 _f2_204 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_102;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_102) : "l"(*(const unsigned long long*)&_f2_203), "l"(*(const unsigned long long*)&_f2_204));
                    vals[4] = _mul_f32x2_102.x;
                    vals[36] = _mul_f32x2_102.y;
                    float2 _f2_205 = make_float2(vals[5], vals[37]);
                    float2 _f2_206 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_103;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_103) : "l"(*(const unsigned long long*)&_f2_205), "l"(*(const unsigned long long*)&_f2_206));
                    vals[5] = _mul_f32x2_103.x;
                    vals[37] = _mul_f32x2_103.y;
                    float2 _f2_207 = make_float2(vals[6], vals[38]);
                    float2 _f2_208 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_104;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_104) : "l"(*(const unsigned long long*)&_f2_207), "l"(*(const unsigned long long*)&_f2_208));
                    vals[6] = _mul_f32x2_104.x;
                    vals[38] = _mul_f32x2_104.y;
                    float2 _f2_209 = make_float2(vals[7], vals[39]);
                    float2 _f2_210 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_105;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_105) : "l"(*(const unsigned long long*)&_f2_209), "l"(*(const unsigned long long*)&_f2_210));
                    vals[7] = _mul_f32x2_105.x;
                    vals[39] = _mul_f32x2_105.y;
                    float2 _f2_211 = make_float2(vals[8], vals[40]);
                    float2 _f2_212 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_106;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_106) : "l"(*(const unsigned long long*)&_f2_211), "l"(*(const unsigned long long*)&_f2_212));
                    vals[8] = _mul_f32x2_106.x;
                    vals[40] = _mul_f32x2_106.y;
                    float2 _f2_213 = make_float2(vals[9], vals[41]);
                    float2 _f2_214 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_107;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_107) : "l"(*(const unsigned long long*)&_f2_213), "l"(*(const unsigned long long*)&_f2_214));
                    vals[9] = _mul_f32x2_107.x;
                    vals[41] = _mul_f32x2_107.y;
                    float2 _f2_215 = make_float2(vals[10], vals[42]);
                    float2 _f2_216 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_108;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_108) : "l"(*(const unsigned long long*)&_f2_215), "l"(*(const unsigned long long*)&_f2_216));
                    vals[10] = _mul_f32x2_108.x;
                    vals[42] = _mul_f32x2_108.y;
                    float2 _f2_217 = make_float2(vals[11], vals[43]);
                    float2 _f2_218 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_109;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_109) : "l"(*(const unsigned long long*)&_f2_217), "l"(*(const unsigned long long*)&_f2_218));
                    vals[11] = _mul_f32x2_109.x;
                    vals[43] = _mul_f32x2_109.y;
                    float2 _f2_219 = make_float2(vals[12], vals[44]);
                    float2 _f2_220 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_110;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_110) : "l"(*(const unsigned long long*)&_f2_219), "l"(*(const unsigned long long*)&_f2_220));
                    vals[12] = _mul_f32x2_110.x;
                    vals[44] = _mul_f32x2_110.y;
                    float2 _f2_221 = make_float2(vals[13], vals[45]);
                    float2 _f2_222 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_111;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_111) : "l"(*(const unsigned long long*)&_f2_221), "l"(*(const unsigned long long*)&_f2_222));
                    vals[13] = _mul_f32x2_111.x;
                    vals[45] = _mul_f32x2_111.y;
                    float2 _f2_223 = make_float2(vals[14], vals[46]);
                    float2 _f2_224 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_112;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_112) : "l"(*(const unsigned long long*)&_f2_223), "l"(*(const unsigned long long*)&_f2_224));
                    vals[14] = _mul_f32x2_112.x;
                    vals[46] = _mul_f32x2_112.y;
                    float2 _f2_225 = make_float2(vals[15], vals[47]);
                    float2 _f2_226 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_113;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_113) : "l"(*(const unsigned long long*)&_f2_225), "l"(*(const unsigned long long*)&_f2_226));
                    vals[15] = _mul_f32x2_113.x;
                    vals[47] = _mul_f32x2_113.y;
                    float2 _f2_227 = make_float2(vals[16], vals[48]);
                    float2 _f2_228 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_114;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_114) : "l"(*(const unsigned long long*)&_f2_227), "l"(*(const unsigned long long*)&_f2_228));
                    vals[16] = _mul_f32x2_114.x;
                    vals[48] = _mul_f32x2_114.y;
                    float2 _f2_229 = make_float2(vals[17], vals[49]);
                    float2 _f2_230 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_115;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_115) : "l"(*(const unsigned long long*)&_f2_229), "l"(*(const unsigned long long*)&_f2_230));
                    vals[17] = _mul_f32x2_115.x;
                    vals[49] = _mul_f32x2_115.y;
                    float2 _f2_231 = make_float2(vals[18], vals[50]);
                    float2 _f2_232 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_116;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_116) : "l"(*(const unsigned long long*)&_f2_231), "l"(*(const unsigned long long*)&_f2_232));
                    vals[18] = _mul_f32x2_116.x;
                    vals[50] = _mul_f32x2_116.y;
                    float2 _f2_233 = make_float2(vals[19], vals[51]);
                    float2 _f2_234 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_117;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_117) : "l"(*(const unsigned long long*)&_f2_233), "l"(*(const unsigned long long*)&_f2_234));
                    vals[19] = _mul_f32x2_117.x;
                    vals[51] = _mul_f32x2_117.y;
                    float2 _f2_235 = make_float2(vals[20], vals[52]);
                    float2 _f2_236 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_118;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_118) : "l"(*(const unsigned long long*)&_f2_235), "l"(*(const unsigned long long*)&_f2_236));
                    vals[20] = _mul_f32x2_118.x;
                    vals[52] = _mul_f32x2_118.y;
                    float2 _f2_237 = make_float2(vals[21], vals[53]);
                    float2 _f2_238 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_119;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_119) : "l"(*(const unsigned long long*)&_f2_237), "l"(*(const unsigned long long*)&_f2_238));
                    vals[21] = _mul_f32x2_119.x;
                    vals[53] = _mul_f32x2_119.y;
                    float2 _f2_239 = make_float2(vals[22], vals[54]);
                    float2 _f2_240 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_120;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_120) : "l"(*(const unsigned long long*)&_f2_239), "l"(*(const unsigned long long*)&_f2_240));
                    vals[22] = _mul_f32x2_120.x;
                    vals[54] = _mul_f32x2_120.y;
                    float2 _f2_241 = make_float2(vals[23], vals[55]);
                    float2 _f2_242 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_121;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_121) : "l"(*(const unsigned long long*)&_f2_241), "l"(*(const unsigned long long*)&_f2_242));
                    vals[23] = _mul_f32x2_121.x;
                    vals[55] = _mul_f32x2_121.y;
                    float2 _f2_243 = make_float2(vals[24], vals[56]);
                    float2 _f2_244 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_122;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_122) : "l"(*(const unsigned long long*)&_f2_243), "l"(*(const unsigned long long*)&_f2_244));
                    vals[24] = _mul_f32x2_122.x;
                    vals[56] = _mul_f32x2_122.y;
                    float2 _f2_245 = make_float2(vals[25], vals[57]);
                    float2 _f2_246 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_123;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_123) : "l"(*(const unsigned long long*)&_f2_245), "l"(*(const unsigned long long*)&_f2_246));
                    vals[25] = _mul_f32x2_123.x;
                    vals[57] = _mul_f32x2_123.y;
                    float2 _f2_247 = make_float2(vals[26], vals[58]);
                    float2 _f2_248 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_124;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_124) : "l"(*(const unsigned long long*)&_f2_247), "l"(*(const unsigned long long*)&_f2_248));
                    vals[26] = _mul_f32x2_124.x;
                    vals[58] = _mul_f32x2_124.y;
                    float2 _f2_249 = make_float2(vals[27], vals[59]);
                    float2 _f2_250 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_125;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_125) : "l"(*(const unsigned long long*)&_f2_249), "l"(*(const unsigned long long*)&_f2_250));
                    vals[27] = _mul_f32x2_125.x;
                    vals[59] = _mul_f32x2_125.y;
                    float2 _f2_251 = make_float2(vals[28], vals[60]);
                    float2 _f2_252 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_126;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_126) : "l"(*(const unsigned long long*)&_f2_251), "l"(*(const unsigned long long*)&_f2_252));
                    vals[28] = _mul_f32x2_126.x;
                    vals[60] = _mul_f32x2_126.y;
                    float2 _f2_253 = make_float2(vals[29], vals[61]);
                    float2 _f2_254 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_127;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_127) : "l"(*(const unsigned long long*)&_f2_253), "l"(*(const unsigned long long*)&_f2_254));
                    vals[29] = _mul_f32x2_127.x;
                    vals[61] = _mul_f32x2_127.y;
                    float2 _f2_255 = make_float2(vals[30], vals[62]);
                    float2 _f2_256 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_128;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_128) : "l"(*(const unsigned long long*)&_f2_255), "l"(*(const unsigned long long*)&_f2_256));
                    vals[30] = _mul_f32x2_128.x;
                    vals[62] = _mul_f32x2_128.y;
                    float2 _f2_257 = make_float2(vals[31], vals[63]);
                    float2 _f2_258 = make_float2(inv0, inv1);
                    float2 _mul_f32x2_129;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_129) : "l"(*(const unsigned long long*)&_f2_257), "l"(*(const unsigned long long*)&_f2_258));
                    vals[31] = _mul_f32x2_129.x;
                    vals[63] = _mul_f32x2_129.y;
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[0]), "f"(vals[1]),
                                               "f"(vals[2]), "f"(vals[3]));
                        words[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[4]), "f"(vals[5]),
                                               "f"(vals[6]), "f"(vals[7]));
                        words[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[8]), "f"(vals[9]),
                                               "f"(vals[10]), "f"(vals[11]));
                        words[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[12]), "f"(vals[13]),
                                               "f"(vals[14]), "f"(vals[15]));
                        words[3] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[16]), "f"(vals[17]),
                                               "f"(vals[18]), "f"(vals[19]));
                        words[4] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[20]), "f"(vals[21]),
                                               "f"(vals[22]), "f"(vals[23]));
                        words[5] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[24]), "f"(vals[25]),
                                               "f"(vals[26]), "f"(vals[27]));
                        words[6] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[28]), "f"(vals[29]),
                                               "f"(vals[30]), "f"(vals[31]));
                        words[7] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[32]), "f"(vals[33]),
                                               "f"(vals[34]), "f"(vals[35]));
                        words[8] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[36]), "f"(vals[37]),
                                               "f"(vals[38]), "f"(vals[39]));
                        words[9] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[40]), "f"(vals[41]),
                                               "f"(vals[42]), "f"(vals[43]));
                        words[10] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[44]), "f"(vals[45]),
                                               "f"(vals[46]), "f"(vals[47]));
                        words[11] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[48]), "f"(vals[49]),
                                               "f"(vals[50]), "f"(vals[51]));
                        words[12] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[52]), "f"(vals[53]),
                                               "f"(vals[54]), "f"(vals[55]));
                        words[13] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[56]), "f"(vals[57]),
                                               "f"(vals[58]), "f"(vals[59]));
                        words[14] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(vals[60]), "f"(vals[61]),
                                               "f"(vals[62]), "f"(vals[63]));
                        words[15] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((c_stage_addr + c_buf * 8192 + (unsigned int)(epi_tidx * 64 ^ (epi_tidx * 64 >> 7 & 3) << 4))), "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((c_stage_addr + c_buf * 8192 + (unsigned int)(epi_tidx * 64 + 16 ^ (epi_tidx * 64 + 16 >> 7 & 3) << 4))), "r"(words[4]), "r"(words[5]), "r"(words[6]), "r"(words[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((c_stage_addr + c_buf * 8192 + (unsigned int)(epi_tidx * 64 + 32 ^ (epi_tidx * 64 + 32 >> 7 & 3) << 4))), "r"(words[8]), "r"(words[9]), "r"(words[10]), "r"(words[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((c_stage_addr + c_buf * 8192 + (unsigned int)(epi_tidx * 64 + 48 ^ (epi_tidx * 64 + 48 >> 7 & 3) << 4))), "r"(words[12]), "r"(words[13]), "r"(words[14]), "r"(words[15]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            tma_store_2d(C, col_out + real * 64, row_out, c_stage_addr + c_buf * 8192);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    c_buf += 1;
                    if (c_buf == 3) { c_buf = 0; }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(acc_free_addr + (acc_stage) * 8);
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_acc_full ^= 1; }
                mbarrier_wait(tile_full_addr + (tile_stage) * 8, _phase_tile_full);
                info[0] = sinfo[tile_stage * 5];
                info[1] = sinfo[tile_stage * 5 + 1];
                info[2] = sinfo[tile_stage * 5 + 2];
                info[3] = sinfo[tile_stage * 5 + 3];
                info[4] = sinfo[tile_stage * 5 + 4];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage) * 8);
                tile_stage += 1;
                if (tile_stage == 2) { tile_stage = 0; _phase_tile_full ^= 1; }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }
    // ---- Role: gather ----
    if (warp >= 4 && warp <= 7) {
        { // gather_main
            unsigned int stage = 0;
            unsigned int tile_stage_1 = 0;
            int info_1[5];
            unsigned int _phase_tile_full_1 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_1) * 8, _phase_tile_full_1);
            info_1[0] = sinfo[tile_stage_1 * 5];
            info_1[1] = sinfo[tile_stage_1 * 5 + 1];
            info_1[2] = sinfo[tile_stage_1 * 5 + 2];
            info_1[3] = sinfo[tile_stage_1 * 5 + 3];
            info_1[4] = sinfo[tile_stage_1 * 5 + 4];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_1) * 8);
            tile_stage_1 += 1;
            if (tile_stage_1 == 2) { tile_stage_1 = 0; _phase_tile_full_1 ^= 1; }
            int t = (warp - 4) * 32 + lane;
            int row_in_pass = t / 8;
            int chunk = t % 8;
            int chunk_off = chunk * 16;
            int sf_row = 8 * (t / 32) + 32 * (t % 32 / 8) + t % 8;
            int sf_dst = (8 * (t / 32) + t % 8) * 16 + t % 32 / 8 * 4;
            int tok[8];
            int ok[8];
            unsigned int _phase_a_free = 1;
            #pragma unroll 1
            for (int _tile_1 = 0; _tile_1 < num_m_tiles * n_tiles + 1; _tile_1++) {
                if (info_1[3] == 0) {
                    break;
                }
                int m_base = info_1[0] * 128;
                int mn_limit = info_1[4];
                int row_i = row_in_pass;
                int prow_i = m_base + row_i;
                int valid_i = (int)(prow_i < mn_limit);
                int expanded_i = token_id_mapping[prow_i];
                tok[0] = expanded_i / top_k * valid_i;
                ok[0] = valid_i;
                int row_i_0 = row_in_pass + 16;
                int prow_i_1 = m_base + row_i_0;
                int valid_i_2 = (int)(prow_i_1 < mn_limit);
                int expanded_i_3 = token_id_mapping[prow_i_1];
                tok[1] = expanded_i_3 / top_k * valid_i_2;
                ok[1] = valid_i_2;
                int row_i_4 = row_in_pass + 32;
                int prow_i_5 = m_base + row_i_4;
                int valid_i_6 = (int)(prow_i_5 < mn_limit);
                int expanded_i_7 = token_id_mapping[prow_i_5];
                tok[2] = expanded_i_7 / top_k * valid_i_6;
                ok[2] = valid_i_6;
                int row_i_8 = row_in_pass + 48;
                int prow_i_9 = m_base + row_i_8;
                int valid_i_10 = (int)(prow_i_9 < mn_limit);
                int expanded_i_11 = token_id_mapping[prow_i_9];
                tok[3] = expanded_i_11 / top_k * valid_i_10;
                ok[3] = valid_i_10;
                int row_i_12 = row_in_pass + 64;
                int prow_i_13 = m_base + row_i_12;
                int valid_i_14 = (int)(prow_i_13 < mn_limit);
                int expanded_i_15 = token_id_mapping[prow_i_13];
                tok[4] = expanded_i_15 / top_k * valid_i_14;
                ok[4] = valid_i_14;
                int row_i_16 = row_in_pass + 80;
                int prow_i_17 = m_base + row_i_16;
                int valid_i_18 = (int)(prow_i_17 < mn_limit);
                int expanded_i_19 = token_id_mapping[prow_i_17];
                tok[5] = expanded_i_19 / top_k * valid_i_18;
                ok[5] = valid_i_18;
                int row_i_20 = row_in_pass + 96;
                int prow_i_21 = m_base + row_i_20;
                int valid_i_22 = (int)(prow_i_21 < mn_limit);
                int expanded_i_23 = token_id_mapping[prow_i_21];
                tok[6] = expanded_i_23 / top_k * valid_i_22;
                ok[6] = valid_i_22;
                int row_i_24 = row_in_pass + 112;
                int prow_i_25 = m_base + row_i_24;
                int valid_i_26 = (int)(prow_i_25 < mn_limit);
                int expanded_i_27 = token_id_mapping[prow_i_25];
                tok[7] = expanded_i_27 / top_k * valid_i_26;
                ok[7] = valid_i_26;
                int sf_prow = m_base + sf_row;
                int sf_valid = (int)(sf_prow < mn_limit);
                int sf_tok = token_id_mapping[sf_prow] / top_k;
                #pragma unroll 1
                for (int k = 0; k < k_tiles; k++) {
                    mbarrier_wait(a_free_addr + (stage) * 8, _phase_a_free);
                    int k0 = k * 128;
                    int col_ok = (int)(k0 + chunk_off < k_cols);
                    int row_c = row_in_pass;
                    int dst_off = row_c * 128 + (chunk ^ row_c % 8) * 16;
                    int src_off = tok[0] * k_cols + k0 + chunk_off;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(act_addr + stage * 16384 + (unsigned int)dst_off), "l"(X + src_off), "r"((ok[0] * col_ok != 0) ? 16 : 0));
                    int row_c_0 = row_in_pass + 16;
                    int dst_off_1 = row_c_0 * 128 + (chunk ^ row_c_0 % 8) * 16;
                    int src_off_2 = tok[1] * k_cols + k0 + chunk_off;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(act_addr + stage * 16384 + (unsigned int)dst_off_1), "l"(X + src_off_2), "r"((ok[1] * col_ok != 0) ? 16 : 0));
                    int row_c_3 = row_in_pass + 32;
                    int dst_off_4 = row_c_3 * 128 + (chunk ^ row_c_3 % 8) * 16;
                    int src_off_5 = tok[2] * k_cols + k0 + chunk_off;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(act_addr + stage * 16384 + (unsigned int)dst_off_4), "l"(X + src_off_5), "r"((ok[2] * col_ok != 0) ? 16 : 0));
                    int row_c_6 = row_in_pass + 48;
                    int dst_off_7 = row_c_6 * 128 + (chunk ^ row_c_6 % 8) * 16;
                    int src_off_8 = tok[3] * k_cols + k0 + chunk_off;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(act_addr + stage * 16384 + (unsigned int)dst_off_7), "l"(X + src_off_8), "r"((ok[3] * col_ok != 0) ? 16 : 0));
                    int row_c_9 = row_in_pass + 64;
                    int dst_off_10 = row_c_9 * 128 + (chunk ^ row_c_9 % 8) * 16;
                    int src_off_11 = tok[4] * k_cols + k0 + chunk_off;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(act_addr + stage * 16384 + (unsigned int)dst_off_10), "l"(X + src_off_11), "r"((ok[4] * col_ok != 0) ? 16 : 0));
                    int row_c_12 = row_in_pass + 80;
                    int dst_off_13 = row_c_12 * 128 + (chunk ^ row_c_12 % 8) * 16;
                    int src_off_14 = tok[5] * k_cols + k0 + chunk_off;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(act_addr + stage * 16384 + (unsigned int)dst_off_13), "l"(X + src_off_14), "r"((ok[5] * col_ok != 0) ? 16 : 0));
                    int row_c_15 = row_in_pass + 96;
                    int dst_off_16 = row_c_15 * 128 + (chunk ^ row_c_15 % 8) * 16;
                    int src_off_17 = tok[6] * k_cols + k0 + chunk_off;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(act_addr + stage * 16384 + (unsigned int)dst_off_16), "l"(X + src_off_17), "r"((ok[6] * col_ok != 0) ? 16 : 0));
                    int row_c_18 = row_in_pass + 112;
                    int dst_off_19 = row_c_18 * 128 + (chunk ^ row_c_18 % 8) * 16;
                    int src_off_20 = tok[7] * k_cols + k0 + chunk_off;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(act_addr + stage * 16384 + (unsigned int)dst_off_19), "l"(X + src_off_20), "r"((ok[7] * col_ok != 0) ? 16 : 0));
                    int sf_k = k * 4;
                    int sf_ok = (int)(sf_k < sf_cols);
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                        :: "r"(act_sf_addr + stage * 512 + (unsigned int)sf_dst), "l"(XSF + (sf_tok * sf_cols + sf_k)), "r"((sf_valid * sf_ok != 0) ? 4 : 0));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(a_full_addr + (stage) * 8) : "memory");
                    stage += 1;
                    if (stage == 6) { stage = 0; _phase_a_free ^= 1; }
                }
                mbarrier_wait(tile_full_addr + (tile_stage_1) * 8, _phase_tile_full_1);
                info_1[0] = sinfo[tile_stage_1 * 5];
                info_1[1] = sinfo[tile_stage_1 * 5 + 1];
                info_1[2] = sinfo[tile_stage_1 * 5 + 2];
                info_1[3] = sinfo[tile_stage_1 * 5 + 3];
                info_1[4] = sinfo[tile_stage_1 * 5 + 4];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_1) * 8);
                tile_stage_1 += 1;
                if (tile_stage_1 == 2) { tile_stage_1 = 0; _phase_tile_full_1 ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            unsigned int sa = 0;
            unsigned int sb = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int acc_idx_1 = 0;
            unsigned int tile_stage_2 = 0;
            int info_2[5];
            unsigned int _phase_tile_full_2 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_2) * 8, _phase_tile_full_2);
            info_2[0] = sinfo[tile_stage_2 * 5];
            info_2[1] = sinfo[tile_stage_2 * 5 + 1];
            info_2[2] = sinfo[tile_stage_2 * 5 + 2];
            info_2[3] = sinfo[tile_stage_2 * 5 + 3];
            info_2[4] = sinfo[tile_stage_2 * 5 + 4];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_2) * 8);
            tile_stage_2 += 1;
            if (tile_stage_2 == 2) { tile_stage_2 = 0; _phase_tile_full_2 ^= 1; }
            unsigned int _phase_acc_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            #pragma unroll 1
            for (int _tile_2 = 0; _tile_2 < num_m_tiles * n_tiles + 1; _tile_2++) {
                if (info_2[3] == 0) {
                    break;
                }
                int acc_col_1 = (int)acc_stage_1 * 128;
                mbarrier_wait(acc_free_addr + (acc_stage_1) * 8, _phase_acc_free);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(a_full_addr + (sa) * 8, _phase_a_full);
                mbarrier_wait(b_full_addr + (sb) * 8, _phase_b_full);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((act_sf_addr) >> 4) + (sa) * 32)));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((w_sf_addr) >> 4) + (sb) * 32)));
                }
                int _mma_a_lo_0 = make_warp_uniform((((act_addr) >> 4) & 0x3FFF) + (sa) * 1024);
                int _mma_b_lo_0 = make_warp_uniform((((w_addr) >> 4) & 0x3FFF) + (sb) * 1024);
                {
                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                    tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_col_1)), a_desc + 0, b_desc + 0,
                        0x8a01400U, tmem_sf_a, tmem_sf_b, 0);
                    tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_col_1)), a_desc + 2, b_desc + 2,
                        0x28a01410U, tmem_sf_a, tmem_sf_b, 1);
                    tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_col_1)), a_desc + 4, b_desc + 4,
                        0x48a01420U, tmem_sf_a, tmem_sf_b, 1);
                    tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_col_1)), a_desc + 6, b_desc + 6,
                        0x68a01430U, tmem_sf_a, tmem_sf_b, 1);
                }
                elect_commit(a_free_addr + (sa) * 8);
                elect_commit(b_free_addr + (sb) * 8);
                sa += 1;
                if (sa == 6) { sa = 0; _phase_a_full ^= 1; }
                sb += 1;
                if (sb == 6) { sb = 0; _phase_b_full ^= 1; }
                #pragma unroll 1
                for (int _k = 1; _k < k_tiles; _k++) {
                    mbarrier_wait(a_full_addr + (sa) * 8, _phase_a_full);
                    mbarrier_wait(b_full_addr + (sb) * 8, _phase_b_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((act_sf_addr) >> 4) + (sa) * 32)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((w_sf_addr) >> 4) + (sb) * 32)));
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((act_addr) >> 4) & 0x3FFF) + (sa) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform((((w_addr) >> 4) & 0x3FFF) + (sb) * 1024);
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_col_1)), a_desc + 0, b_desc + 0,
                            0x8a01400U, tmem_sf_a, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_col_1)), a_desc + 2, b_desc + 2,
                            0x28a01410U, tmem_sf_a, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_col_1)), a_desc + 4, b_desc + 4,
                            0x48a01420U, tmem_sf_a, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_col_1)), a_desc + 6, b_desc + 6,
                            0x68a01430U, tmem_sf_a, tmem_sf_b, 1);
                    }
                    elect_commit(a_free_addr + (sa) * 8);
                    elect_commit(b_free_addr + (sb) * 8);
                    sa += 1;
                    if (sa == 6) { sa = 0; _phase_a_full ^= 1; }
                    sb += 1;
                    if (sb == 6) { sb = 0; _phase_b_full ^= 1; }
                }
                elect_commit(acc_full_addr + (acc_stage_1) * 8);
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_acc_free ^= 1; }
                mbarrier_wait(tile_full_addr + (tile_stage_2) * 8, _phase_tile_full_2);
                info_2[0] = sinfo[tile_stage_2 * 5];
                info_2[1] = sinfo[tile_stage_2 * 5 + 1];
                info_2[2] = sinfo[tile_stage_2 * 5 + 2];
                info_2[3] = sinfo[tile_stage_2 * 5 + 3];
                info_2[4] = sinfo[tile_stage_2 * 5 + 4];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_2) * 8);
                tile_stage_2 += 1;
                if (tile_stage_2 == 2) { tile_stage_2 = 0; _phase_tile_full_2 ^= 1; }
            }
        }
    }
    // ---- Role: tma ----
    if (warp == 9) {
        { // tma_main
            unsigned int stage_1 = 0;
            unsigned int tile_stage_3 = 0;
            int info_3[5];
            unsigned int _phase_tile_full_3 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_3) * 8, _phase_tile_full_3);
            info_3[0] = sinfo[tile_stage_3 * 5];
            info_3[1] = sinfo[tile_stage_3 * 5 + 1];
            info_3[2] = sinfo[tile_stage_3 * 5 + 2];
            info_3[3] = sinfo[tile_stage_3 * 5 + 3];
            info_3[4] = sinfo[tile_stage_3 * 5 + 4];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_3) * 8);
            tile_stage_3 += 1;
            if (tile_stage_3 == 2) { tile_stage_3 = 0; _phase_tile_full_3 ^= 1; }
            int b_row0 = 0;
            unsigned int _phase_b_free = 1;
            #pragma unroll 1
            for (int _tile_3 = 0; _tile_3 < num_m_tiles * n_tiles + 1; _tile_3++) {
                if (info_3[3] == 0) {
                    break;
                }
                int expert_t = info_3[2];
                int n_row = info_3[1] * 128 + b_row0;
                int n_atom = info_3[1];
                #pragma unroll 1
                for (int k_1 = 0; k_1 < k_tiles; k_1++) {
                    mbarrier_wait(b_free_addr + (stage_1) * 8, _phase_b_free);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(b_full_addr + (stage_1) * 8, 8704);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(w_addr + stage_1 * 16384), "l"(W), "r"(k_1 * 128), "r"(n_row), "r"(expert_t),
                               "r"(b_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(w_sf_addr + stage_1 * 512), "l"(WSF), "r"(0), "r"(0), "r"(k_1), "r"(n_atom), "r"(expert_t),
                               "r"(b_full_addr + (stage_1) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    stage_1 += 1;
                    if (stage_1 == 6) { stage_1 = 0; _phase_b_free ^= 1; }
                }
                mbarrier_wait(tile_full_addr + (tile_stage_3) * 8, _phase_tile_full_3);
                info_3[0] = sinfo[tile_stage_3 * 5];
                info_3[1] = sinfo[tile_stage_3 * 5 + 1];
                info_3[2] = sinfo[tile_stage_3 * 5 + 2];
                info_3[3] = sinfo[tile_stage_3 * 5 + 3];
                info_3[4] = sinfo[tile_stage_3 * 5 + 4];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_3) * 8);
                tile_stage_3 += 1;
                if (tile_stage_3 == 2) { tile_stage_3 = 0; _phase_tile_full_3 ^= 1; }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        { // scheduler_main
            unsigned int tile_stage_4 = 0;
            int num_valid = num_non_exiting_tiles[0];
            int total_items = num_m_tiles * n_tiles;
            int sched_first = bid;
            int sched_step = num_bids;
            int cta_rank_m = 0;
            unsigned int _phase_tile_free = 1;
            #pragma unroll 1
            for (int item = sched_first; item < total_items; item += sched_step) {
                int m_tile_1 = item / n_tiles;
                int n_tile_1 = item - m_tile_1 * n_tiles;
                if (m_tile_1 >= num_valid) {
                    break;
                }
                mbarrier_wait(tile_free_addr + (tile_stage_4) * 8, _phase_tile_free);
                int sched_group = tile_idx_to_row_group[m_tile_1];
                int sched_coord_m = sched_group + cta_rank_m;
                int expert = tile_idx_to_expert_idx[sched_group];
                int mn_limit_1 = tile_idx_to_mn_limit[sched_group];
                if (elect_sync()) {
                    sinfo[tile_stage_4 * 5] = sched_coord_m;
                    sinfo[tile_stage_4 * 5 + 1] = n_tile_1;
                    sinfo[tile_stage_4 * 5 + 2] = expert;
                    sinfo[tile_stage_4 * 5 + 3] = 1;
                    sinfo[tile_stage_4 * 5 + 4] = mn_limit_1;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 4, 32;" ::: "memory");
                mbarrier_arrive(tile_full_addr + (tile_stage_4) * 8);
                tile_stage_4 += 1;
                if (tile_stage_4 == 2) { tile_stage_4 = 0; _phase_tile_free ^= 1; }
            }
            mbarrier_wait(tile_free_addr + (tile_stage_4) * 8, _phase_tile_free);
            if (elect_sync()) {
                sinfo[tile_stage_4 * 5] = 0;
                sinfo[tile_stage_4 * 5 + 1] = 0;
                sinfo[tile_stage_4 * 5 + 2] = -1;
                sinfo[tile_stage_4 * 5 + 3] = 0;
                sinfo[tile_stage_4 * 5 + 4] = -1;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 4, 32;" ::: "memory");
            mbarrier_arrive(tile_full_addr + (tile_stage_4) * 8);
            tile_stage_4 += 1;
            if (tile_stage_4 == 2) { tile_stage_4 = 0; _phase_tile_free ^= 1; }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
