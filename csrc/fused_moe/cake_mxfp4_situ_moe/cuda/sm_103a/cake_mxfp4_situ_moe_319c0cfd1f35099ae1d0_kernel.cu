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
#define TMEM_NCOLS 72
#define TMEM_ACC_OFFSET 0
#define TMEM_SF_A_OFFSET 64
#define TMEM_SF_B_OFFSET 68
#define NUM_PAB_STAGES 9
#define NUM_PB_STAGES 9
#define NUM_PACC_STAGES 2
#define NUM_PTILE_STAGES 8
#define SMEM_A_OFF 1024
#define SMEM_A_STAGE_BYTES 16384
#define SMEM_A_STRIDE 16384
#define SMEM_B_OFF 148480
#define SMEM_B_STAGE_BYTES 4096
#define SMEM_B_STRIDE 4096
#define SMEM_SFA_OFF 185344
#define SMEM_SFA_STAGE_BYTES 512
#define SMEM_SFA_STRIDE 512
#define SMEM_SFB_OFF 189952
#define SMEM_SFB_STAGE_BYTES 512
#define SMEM_SFB_STRIDE 512
#define SMEM_SINFO_OFF 194560
#define SMEM_SINFO_STAGE_BYTES 224
#define SMEM_SINFO_STRIDE 224
#define SMEM_STOK_OFF 194784
#define SMEM_STOK_STAGE_BYTES 1024
#define SMEM_STOK_STRIDE 1024
#define SMEM_SSCALE_OFF 195808
#define SMEM_SSCALE_STAGE_BYTES 1056
#define SMEM_SSCALE_STRIDE 1056
#define SMEM_SEXCH_OFF 196864
#define SMEM_SEXCH_STAGE_BYTES 8448
#define SMEM_SEXCH_STRIDE 8448
#define SMEM_TOTAL 205312
#define THREADS 256

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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mxfp4_situ_moe_319c0cfd1f35099ae1d0(CakeTensorMap const* A, CakeTensorMap const* SFA, uint8_t* __restrict__ B, uint8_t* __restrict__ SFB, uint8_t* __restrict__ out, int* __restrict__ tile_idx_to_expert_idx, int* __restrict__ tile_idx_to_mn_limit, int* __restrict__ num_non_exiting_tiles, int* __restrict__ tile_idx_to_row_group, float* __restrict__ alpha, int* __restrict__ permuted_idx_to_expanded_idx, float* __restrict__ token_final_scales, int num_m_tiles, int group_capacity, int k_tiles, int k_cols, int sf_cols, int out_cols, int top_k, float* __restrict__ situ_beta, float* __restrict__ situ_linear_beta, uint8_t* __restrict__ act_sf, float* __restrict__ zero_buf, int zero_words, int num_rows_b, int act_cols, int act_sf_cols, int* __restrict__ dbg)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_free_addr (mbar_base + 72)
    #define b_full_addr (mbar_base + 144)
    #define b_free_addr (mbar_base + 216)
    #define acc_full_addr (mbar_base + 288)
    #define acc_free_addr (mbar_base + 304)
    #define tile_full_addr (mbar_base + 320)
    #define tile_free_addr (mbar_base + 384)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFA)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int a_addr = smem + 1024;
    uint8_t* b = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int b_addr = smem + 148480;
    uint8_t* sfa = reinterpret_cast<uint8_t*>(smem_raw + 185344);
    const int sfa_addr = smem + 185344;
    uint8_t* sfb = reinterpret_cast<uint8_t*>(smem_raw + 189952);
    const int sfb_addr = smem + 189952;
    int* sinfo = reinterpret_cast<int*>(smem_raw + 194560);
    const int sinfo_addr = smem + 194560;
    int* stok = reinterpret_cast<int*>(smem_raw + 194784);
    const int stok_addr = smem + 194784;
    float* sscale = reinterpret_cast<float*>(smem_raw + 195808);
    const int sscale_addr = smem + 195808;
    float* sexch = reinterpret_cast<float*>(smem_raw + 196864);
    const int sexch_addr = smem + 196864;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 56 barriers)
    // Mbarriers at smem_raw[0..448)

    if (warp == 0) {
        // --- pipeline 'pab' ---
        // ab_full: 9 barriers, init_count=1
        // ab_free: 9 barriers, init_count=1
        // --- pipeline 'pb' ---
        // b_full: 9 barriers, init_count=32
        // b_free: 9 barriers, init_count=1
        // --- pipeline 'pacc' ---
        // acc_full: 2 barriers, init_count=1
        // acc_free: 2 barriers, init_count=128
        // --- pipeline 'ptile' ---
        // tile_full: 8 barriers, init_count=32
        // tile_free: 8 barriers, init_count=224
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(27), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(18), "r"((uint32_t)(1)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        uint32_t _mbarrier_init_count_0_32 = 224;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(16), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(8), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(6), "r"((uint32_t)(1)));
        if (lane < 24) {
            mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (128 columns, 72 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 448);
    if (warp == 0) {
        int _tmem_hold = smem + 448;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_sf_a = taddr + 64;
    const int tmem_sf_b = taddr + 68;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int nrec_epilogue = 0;
            int nst_epilogue = 0;
            int epi_tidx = tid;
            int lane_0 = lane;
            int epi_warp = warp;
            unsigned int acc_stage = 0;
            unsigned int tile_stage = 0;
            int info[7];
            int cur_tok[32];
            float cur_scale[32];
            float meta_alpha = 0.0f;
            float vals[32];
            float amax[32];
            float zero_pair[2];
            zero_pair[0] = 0.0f;
            zero_pair[1] = 0.0f;
            #pragma unroll 1
            for (int zi = bid * 128 + epi_tidx; zi < zero_words; zi += num_bids * 128) {
                {
                    float2 _v2 = make_float2(zero_pair[0 + 0], zero_pair[0 + 1]);
                    *reinterpret_cast<float2*>(zero_buf + (zi * 2) + 0) = _v2;
                }
            }
            unsigned int _phase_tile_full = 0;
            mbarrier_wait(tile_full_addr + (tile_stage) * 8, _phase_tile_full);
            info[0] = sinfo[tile_stage * 7];
            info[1] = sinfo[tile_stage * 7 + 1];
            info[2] = sinfo[tile_stage * 7 + 2];
            info[3] = sinfo[tile_stage * 7 + 3];
            info[4] = sinfo[tile_stage * 7 + 4];
            info[5] = sinfo[tile_stage * 7 + 5];
            info[6] = sinfo[tile_stage * 7 + 6];
            meta_alpha = sscale[tile_stage * 33 + 32];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage) * 8);
            tile_stage += 1;
            if (tile_stage == 8) { tile_stage = 0; _phase_tile_full ^= 1; }
            nrec_epilogue = nrec_epilogue + 1;
            int is_even_lane = (int)(lane_0 % 2 == 0);
            int is_gate_lane = (int)(epi_warp >= 2);
            int exch_row = epi_tidx & 63;
            float inv_fp8_max = 0.002232142857142857f;
            float zero_f32 = 0.0f;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int _tile = 0; _tile < num_m_tiles * group_capacity + 1; _tile++) {
                if (info[3] == 0) {
                    break;
                }
                int row_base = info[1] * 32;
                int mn_limit = info[4];
                int h0 = info[0] * 128;
                int h = h0 + epi_tidx;
                int expert_e = info[2];
                float beta = situ_beta[expert_e];
                float _fdiv_rn_0 = __fdiv_rn(1.0f, beta);
                float inv_beta = _fdiv_rn_0;
                float linear_beta = situ_linear_beta[expert_e];
                float _fdiv_rn_1 = __fdiv_rn(1.0f, linear_beta);
                float inv_linear_beta = _fdiv_rn_1;
                int j_col = info[0] * 64 + epi_tidx;
                mbarrier_wait(acc_full_addr + (acc_stage) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                {
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    vals[0] = _tmem_load_0[0] * meta_alpha;
                    vals[1] = _tmem_load_0[1] * meta_alpha;
                    vals[2] = _tmem_load_0[2] * meta_alpha;
                    vals[3] = _tmem_load_0[3] * meta_alpha;
                    vals[4] = _tmem_load_0[4] * meta_alpha;
                    vals[5] = _tmem_load_0[5] * meta_alpha;
                    vals[6] = _tmem_load_0[6] * meta_alpha;
                    vals[7] = _tmem_load_0[7] * meta_alpha;
                    vals[8] = _tmem_load_0[8] * meta_alpha;
                    vals[9] = _tmem_load_0[9] * meta_alpha;
                    vals[10] = _tmem_load_0[10] * meta_alpha;
                    vals[11] = _tmem_load_0[11] * meta_alpha;
                    vals[12] = _tmem_load_0[12] * meta_alpha;
                    vals[13] = _tmem_load_0[13] * meta_alpha;
                    vals[14] = _tmem_load_0[14] * meta_alpha;
                    vals[15] = _tmem_load_0[15] * meta_alpha;
                    vals[16] = _tmem_load_0[16] * meta_alpha;
                    vals[17] = _tmem_load_0[17] * meta_alpha;
                    vals[18] = _tmem_load_0[18] * meta_alpha;
                    vals[19] = _tmem_load_0[19] * meta_alpha;
                    vals[20] = _tmem_load_0[20] * meta_alpha;
                    vals[21] = _tmem_load_0[21] * meta_alpha;
                    vals[22] = _tmem_load_0[22] * meta_alpha;
                    vals[23] = _tmem_load_0[23] * meta_alpha;
                    vals[24] = _tmem_load_0[24] * meta_alpha;
                    vals[25] = _tmem_load_0[25] * meta_alpha;
                    vals[26] = _tmem_load_0[26] * meta_alpha;
                    vals[27] = _tmem_load_0[27] * meta_alpha;
                    vals[28] = _tmem_load_0[28] * meta_alpha;
                    vals[29] = _tmem_load_0[29] * meta_alpha;
                    vals[30] = _tmem_load_0[30] * meta_alpha;
                    vals[31] = _tmem_load_0[31] * meta_alpha;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    {
                        if (is_gate_lane != 0) {
                            float x_g = vals[0];
                            float _exp2_0 = approx_exp2(x_g * -1.4426950408889634f);
                            float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                            float sig_g = _rcp_0;
                            float _tanh_approx_0;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(x_g * inv_beta));
                            sexch[exch_row * 33] = beta * _tanh_approx_0 * sig_g;
                            float x_g_0 = vals[1];
                            float _exp2_1 = approx_exp2(x_g_0 * -1.4426950408889634f);
                            float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                            float sig_g_1 = _rcp_1;
                            float _tanh_approx_1;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(x_g_0 * inv_beta));
                            sexch[exch_row * 33 + 1] = beta * _tanh_approx_1 * sig_g_1;
                            float x_g_2 = vals[2];
                            float _exp2_2 = approx_exp2(x_g_2 * -1.4426950408889634f);
                            float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                            float sig_g_3 = _rcp_2;
                            float _tanh_approx_2;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(x_g_2 * inv_beta));
                            sexch[exch_row * 33 + 2] = beta * _tanh_approx_2 * sig_g_3;
                            float x_g_4 = vals[3];
                            float _exp2_3 = approx_exp2(x_g_4 * -1.4426950408889634f);
                            float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                            float sig_g_5 = _rcp_3;
                            float _tanh_approx_3;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(x_g_4 * inv_beta));
                            sexch[exch_row * 33 + 3] = beta * _tanh_approx_3 * sig_g_5;
                            float x_g_6 = vals[4];
                            float _exp2_4 = approx_exp2(x_g_6 * -1.4426950408889634f);
                            float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                            float sig_g_7 = _rcp_4;
                            float _tanh_approx_4;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_4) : "f"(x_g_6 * inv_beta));
                            sexch[exch_row * 33 + 4] = beta * _tanh_approx_4 * sig_g_7;
                            float x_g_8 = vals[5];
                            float _exp2_5 = approx_exp2(x_g_8 * -1.4426950408889634f);
                            float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                            float sig_g_9 = _rcp_5;
                            float _tanh_approx_5;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_5) : "f"(x_g_8 * inv_beta));
                            sexch[exch_row * 33 + 5] = beta * _tanh_approx_5 * sig_g_9;
                            float x_g_10 = vals[6];
                            float _exp2_6 = approx_exp2(x_g_10 * -1.4426950408889634f);
                            float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                            float sig_g_11 = _rcp_6;
                            float _tanh_approx_6;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_6) : "f"(x_g_10 * inv_beta));
                            sexch[exch_row * 33 + 6] = beta * _tanh_approx_6 * sig_g_11;
                            float x_g_12 = vals[7];
                            float _exp2_7 = approx_exp2(x_g_12 * -1.4426950408889634f);
                            float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                            float sig_g_13 = _rcp_7;
                            float _tanh_approx_7;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_7) : "f"(x_g_12 * inv_beta));
                            sexch[exch_row * 33 + 7] = beta * _tanh_approx_7 * sig_g_13;
                            float x_g_14 = vals[8];
                            float _exp2_8 = approx_exp2(x_g_14 * -1.4426950408889634f);
                            float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                            float sig_g_15 = _rcp_8;
                            float _tanh_approx_8;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_8) : "f"(x_g_14 * inv_beta));
                            sexch[exch_row * 33 + 8] = beta * _tanh_approx_8 * sig_g_15;
                            float x_g_16 = vals[9];
                            float _exp2_9 = approx_exp2(x_g_16 * -1.4426950408889634f);
                            float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                            float sig_g_17 = _rcp_9;
                            float _tanh_approx_9;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_9) : "f"(x_g_16 * inv_beta));
                            sexch[exch_row * 33 + 9] = beta * _tanh_approx_9 * sig_g_17;
                            float x_g_18 = vals[10];
                            float _exp2_10 = approx_exp2(x_g_18 * -1.4426950408889634f);
                            float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                            float sig_g_19 = _rcp_10;
                            float _tanh_approx_10;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_10) : "f"(x_g_18 * inv_beta));
                            sexch[exch_row * 33 + 10] = beta * _tanh_approx_10 * sig_g_19;
                            float x_g_20 = vals[11];
                            float _exp2_11 = approx_exp2(x_g_20 * -1.4426950408889634f);
                            float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                            float sig_g_21 = _rcp_11;
                            float _tanh_approx_11;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_11) : "f"(x_g_20 * inv_beta));
                            sexch[exch_row * 33 + 11] = beta * _tanh_approx_11 * sig_g_21;
                            float x_g_22 = vals[12];
                            float _exp2_12 = approx_exp2(x_g_22 * -1.4426950408889634f);
                            float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                            float sig_g_23 = _rcp_12;
                            float _tanh_approx_12;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_12) : "f"(x_g_22 * inv_beta));
                            sexch[exch_row * 33 + 12] = beta * _tanh_approx_12 * sig_g_23;
                            float x_g_24 = vals[13];
                            float _exp2_13 = approx_exp2(x_g_24 * -1.4426950408889634f);
                            float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                            float sig_g_25 = _rcp_13;
                            float _tanh_approx_13;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_13) : "f"(x_g_24 * inv_beta));
                            sexch[exch_row * 33 + 13] = beta * _tanh_approx_13 * sig_g_25;
                            float x_g_26 = vals[14];
                            float _exp2_14 = approx_exp2(x_g_26 * -1.4426950408889634f);
                            float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                            float sig_g_27 = _rcp_14;
                            float _tanh_approx_14;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_14) : "f"(x_g_26 * inv_beta));
                            sexch[exch_row * 33 + 14] = beta * _tanh_approx_14 * sig_g_27;
                            float x_g_28 = vals[15];
                            float _exp2_15 = approx_exp2(x_g_28 * -1.4426950408889634f);
                            float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                            float sig_g_29 = _rcp_15;
                            float _tanh_approx_15;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_15) : "f"(x_g_28 * inv_beta));
                            sexch[exch_row * 33 + 15] = beta * _tanh_approx_15 * sig_g_29;
                            float x_g_30 = vals[16];
                            float _exp2_16 = approx_exp2(x_g_30 * -1.4426950408889634f);
                            float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                            float sig_g_31 = _rcp_16;
                            float _tanh_approx_16;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_16) : "f"(x_g_30 * inv_beta));
                            sexch[exch_row * 33 + 16] = beta * _tanh_approx_16 * sig_g_31;
                            float x_g_32 = vals[17];
                            float _exp2_17 = approx_exp2(x_g_32 * -1.4426950408889634f);
                            float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                            float sig_g_33 = _rcp_17;
                            float _tanh_approx_17;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_17) : "f"(x_g_32 * inv_beta));
                            sexch[exch_row * 33 + 17] = beta * _tanh_approx_17 * sig_g_33;
                            float x_g_34 = vals[18];
                            float _exp2_18 = approx_exp2(x_g_34 * -1.4426950408889634f);
                            float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                            float sig_g_35 = _rcp_18;
                            float _tanh_approx_18;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_18) : "f"(x_g_34 * inv_beta));
                            sexch[exch_row * 33 + 18] = beta * _tanh_approx_18 * sig_g_35;
                            float x_g_36 = vals[19];
                            float _exp2_19 = approx_exp2(x_g_36 * -1.4426950408889634f);
                            float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                            float sig_g_37 = _rcp_19;
                            float _tanh_approx_19;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_19) : "f"(x_g_36 * inv_beta));
                            sexch[exch_row * 33 + 19] = beta * _tanh_approx_19 * sig_g_37;
                            float x_g_38 = vals[20];
                            float _exp2_20 = approx_exp2(x_g_38 * -1.4426950408889634f);
                            float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                            float sig_g_39 = _rcp_20;
                            float _tanh_approx_20;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_20) : "f"(x_g_38 * inv_beta));
                            sexch[exch_row * 33 + 20] = beta * _tanh_approx_20 * sig_g_39;
                            float x_g_40 = vals[21];
                            float _exp2_21 = approx_exp2(x_g_40 * -1.4426950408889634f);
                            float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                            float sig_g_41 = _rcp_21;
                            float _tanh_approx_21;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_21) : "f"(x_g_40 * inv_beta));
                            sexch[exch_row * 33 + 21] = beta * _tanh_approx_21 * sig_g_41;
                            float x_g_42 = vals[22];
                            float _exp2_22 = approx_exp2(x_g_42 * -1.4426950408889634f);
                            float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                            float sig_g_43 = _rcp_22;
                            float _tanh_approx_22;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_22) : "f"(x_g_42 * inv_beta));
                            sexch[exch_row * 33 + 22] = beta * _tanh_approx_22 * sig_g_43;
                            float x_g_44 = vals[23];
                            float _exp2_23 = approx_exp2(x_g_44 * -1.4426950408889634f);
                            float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                            float sig_g_45 = _rcp_23;
                            float _tanh_approx_23;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_23) : "f"(x_g_44 * inv_beta));
                            sexch[exch_row * 33 + 23] = beta * _tanh_approx_23 * sig_g_45;
                            float x_g_46 = vals[24];
                            float _exp2_24 = approx_exp2(x_g_46 * -1.4426950408889634f);
                            float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                            float sig_g_47 = _rcp_24;
                            float _tanh_approx_24;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_24) : "f"(x_g_46 * inv_beta));
                            sexch[exch_row * 33 + 24] = beta * _tanh_approx_24 * sig_g_47;
                            float x_g_48 = vals[25];
                            float _exp2_25 = approx_exp2(x_g_48 * -1.4426950408889634f);
                            float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                            float sig_g_49 = _rcp_25;
                            float _tanh_approx_25;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_25) : "f"(x_g_48 * inv_beta));
                            sexch[exch_row * 33 + 25] = beta * _tanh_approx_25 * sig_g_49;
                            float x_g_50 = vals[26];
                            float _exp2_26 = approx_exp2(x_g_50 * -1.4426950408889634f);
                            float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                            float sig_g_51 = _rcp_26;
                            float _tanh_approx_26;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_26) : "f"(x_g_50 * inv_beta));
                            sexch[exch_row * 33 + 26] = beta * _tanh_approx_26 * sig_g_51;
                            float x_g_52 = vals[27];
                            float _exp2_27 = approx_exp2(x_g_52 * -1.4426950408889634f);
                            float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                            float sig_g_53 = _rcp_27;
                            float _tanh_approx_27;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_27) : "f"(x_g_52 * inv_beta));
                            sexch[exch_row * 33 + 27] = beta * _tanh_approx_27 * sig_g_53;
                            float x_g_54 = vals[28];
                            float _exp2_28 = approx_exp2(x_g_54 * -1.4426950408889634f);
                            float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                            float sig_g_55 = _rcp_28;
                            float _tanh_approx_28;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_28) : "f"(x_g_54 * inv_beta));
                            sexch[exch_row * 33 + 28] = beta * _tanh_approx_28 * sig_g_55;
                            float x_g_56 = vals[29];
                            float _exp2_29 = approx_exp2(x_g_56 * -1.4426950408889634f);
                            float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                            float sig_g_57 = _rcp_29;
                            float _tanh_approx_29;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_29) : "f"(x_g_56 * inv_beta));
                            sexch[exch_row * 33 + 29] = beta * _tanh_approx_29 * sig_g_57;
                            float x_g_58 = vals[30];
                            float _exp2_30 = approx_exp2(x_g_58 * -1.4426950408889634f);
                            float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                            float sig_g_59 = _rcp_30;
                            float _tanh_approx_30;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_30) : "f"(x_g_58 * inv_beta));
                            sexch[exch_row * 33 + 30] = beta * _tanh_approx_30 * sig_g_59;
                            float x_g_60 = vals[31];
                            float _exp2_31 = approx_exp2(x_g_60 * -1.4426950408889634f);
                            float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                            float sig_g_61 = _rcp_31;
                            float _tanh_approx_31;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_31) : "f"(x_g_60 * inv_beta));
                            sexch[exch_row * 33 + 31] = beta * _tanh_approx_31 * sig_g_61;
                        } else {
                            float _tanh_approx_32;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_32) : "f"(vals[0] * inv_linear_beta));
                            vals[0] = linear_beta * _tanh_approx_32;
                            float _tanh_approx_33;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_33) : "f"(vals[1] * inv_linear_beta));
                            vals[1] = linear_beta * _tanh_approx_33;
                            float _tanh_approx_34;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_34) : "f"(vals[2] * inv_linear_beta));
                            vals[2] = linear_beta * _tanh_approx_34;
                            float _tanh_approx_35;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_35) : "f"(vals[3] * inv_linear_beta));
                            vals[3] = linear_beta * _tanh_approx_35;
                            float _tanh_approx_36;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_36) : "f"(vals[4] * inv_linear_beta));
                            vals[4] = linear_beta * _tanh_approx_36;
                            float _tanh_approx_37;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_37) : "f"(vals[5] * inv_linear_beta));
                            vals[5] = linear_beta * _tanh_approx_37;
                            float _tanh_approx_38;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_38) : "f"(vals[6] * inv_linear_beta));
                            vals[6] = linear_beta * _tanh_approx_38;
                            float _tanh_approx_39;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_39) : "f"(vals[7] * inv_linear_beta));
                            vals[7] = linear_beta * _tanh_approx_39;
                            float _tanh_approx_40;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_40) : "f"(vals[8] * inv_linear_beta));
                            vals[8] = linear_beta * _tanh_approx_40;
                            float _tanh_approx_41;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_41) : "f"(vals[9] * inv_linear_beta));
                            vals[9] = linear_beta * _tanh_approx_41;
                            float _tanh_approx_42;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_42) : "f"(vals[10] * inv_linear_beta));
                            vals[10] = linear_beta * _tanh_approx_42;
                            float _tanh_approx_43;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_43) : "f"(vals[11] * inv_linear_beta));
                            vals[11] = linear_beta * _tanh_approx_43;
                            float _tanh_approx_44;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_44) : "f"(vals[12] * inv_linear_beta));
                            vals[12] = linear_beta * _tanh_approx_44;
                            float _tanh_approx_45;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_45) : "f"(vals[13] * inv_linear_beta));
                            vals[13] = linear_beta * _tanh_approx_45;
                            float _tanh_approx_46;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_46) : "f"(vals[14] * inv_linear_beta));
                            vals[14] = linear_beta * _tanh_approx_46;
                            float _tanh_approx_47;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_47) : "f"(vals[15] * inv_linear_beta));
                            vals[15] = linear_beta * _tanh_approx_47;
                            float _tanh_approx_48;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_48) : "f"(vals[16] * inv_linear_beta));
                            vals[16] = linear_beta * _tanh_approx_48;
                            float _tanh_approx_49;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_49) : "f"(vals[17] * inv_linear_beta));
                            vals[17] = linear_beta * _tanh_approx_49;
                            float _tanh_approx_50;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_50) : "f"(vals[18] * inv_linear_beta));
                            vals[18] = linear_beta * _tanh_approx_50;
                            float _tanh_approx_51;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_51) : "f"(vals[19] * inv_linear_beta));
                            vals[19] = linear_beta * _tanh_approx_51;
                            float _tanh_approx_52;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_52) : "f"(vals[20] * inv_linear_beta));
                            vals[20] = linear_beta * _tanh_approx_52;
                            float _tanh_approx_53;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_53) : "f"(vals[21] * inv_linear_beta));
                            vals[21] = linear_beta * _tanh_approx_53;
                            float _tanh_approx_54;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_54) : "f"(vals[22] * inv_linear_beta));
                            vals[22] = linear_beta * _tanh_approx_54;
                            float _tanh_approx_55;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_55) : "f"(vals[23] * inv_linear_beta));
                            vals[23] = linear_beta * _tanh_approx_55;
                            float _tanh_approx_56;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_56) : "f"(vals[24] * inv_linear_beta));
                            vals[24] = linear_beta * _tanh_approx_56;
                            float _tanh_approx_57;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_57) : "f"(vals[25] * inv_linear_beta));
                            vals[25] = linear_beta * _tanh_approx_57;
                            float _tanh_approx_58;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_58) : "f"(vals[26] * inv_linear_beta));
                            vals[26] = linear_beta * _tanh_approx_58;
                            float _tanh_approx_59;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_59) : "f"(vals[27] * inv_linear_beta));
                            vals[27] = linear_beta * _tanh_approx_59;
                            float _tanh_approx_60;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_60) : "f"(vals[28] * inv_linear_beta));
                            vals[28] = linear_beta * _tanh_approx_60;
                            float _tanh_approx_61;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_61) : "f"(vals[29] * inv_linear_beta));
                            vals[29] = linear_beta * _tanh_approx_61;
                            float _tanh_approx_62;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_62) : "f"(vals[30] * inv_linear_beta));
                            vals[30] = linear_beta * _tanh_approx_62;
                            float _tanh_approx_63;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_63) : "f"(vals[31] * inv_linear_beta));
                            vals[31] = linear_beta * _tanh_approx_63;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        if (is_gate_lane == 0) {
                            float v_u = vals[0] * sexch[exch_row * 33];
                            vals[0] = v_u;
                            float _fmax_0 = fmaxf(v_u, -v_u);
                            amax[0] = _fmax_0;
                            float v_u_0 = vals[1] * sexch[exch_row * 33 + 1];
                            vals[1] = v_u_0;
                            float _fmax_1 = fmaxf(v_u_0, -v_u_0);
                            amax[1] = _fmax_1;
                            float v_u_1 = vals[2] * sexch[exch_row * 33 + 2];
                            vals[2] = v_u_1;
                            float _fmax_2 = fmaxf(v_u_1, -v_u_1);
                            amax[2] = _fmax_2;
                            float v_u_2 = vals[3] * sexch[exch_row * 33 + 3];
                            vals[3] = v_u_2;
                            float _fmax_3 = fmaxf(v_u_2, -v_u_2);
                            amax[3] = _fmax_3;
                            float v_u_3 = vals[4] * sexch[exch_row * 33 + 4];
                            vals[4] = v_u_3;
                            float _fmax_4 = fmaxf(v_u_3, -v_u_3);
                            amax[4] = _fmax_4;
                            float v_u_4 = vals[5] * sexch[exch_row * 33 + 5];
                            vals[5] = v_u_4;
                            float _fmax_5 = fmaxf(v_u_4, -v_u_4);
                            amax[5] = _fmax_5;
                            float v_u_5 = vals[6] * sexch[exch_row * 33 + 6];
                            vals[6] = v_u_5;
                            float _fmax_6 = fmaxf(v_u_5, -v_u_5);
                            amax[6] = _fmax_6;
                            float v_u_6 = vals[7] * sexch[exch_row * 33 + 7];
                            vals[7] = v_u_6;
                            float _fmax_7 = fmaxf(v_u_6, -v_u_6);
                            amax[7] = _fmax_7;
                            float v_u_7 = vals[8] * sexch[exch_row * 33 + 8];
                            vals[8] = v_u_7;
                            float _fmax_8 = fmaxf(v_u_7, -v_u_7);
                            amax[8] = _fmax_8;
                            float v_u_8 = vals[9] * sexch[exch_row * 33 + 9];
                            vals[9] = v_u_8;
                            float _fmax_9 = fmaxf(v_u_8, -v_u_8);
                            amax[9] = _fmax_9;
                            float v_u_9 = vals[10] * sexch[exch_row * 33 + 10];
                            vals[10] = v_u_9;
                            float _fmax_10 = fmaxf(v_u_9, -v_u_9);
                            amax[10] = _fmax_10;
                            float v_u_10 = vals[11] * sexch[exch_row * 33 + 11];
                            vals[11] = v_u_10;
                            float _fmax_11 = fmaxf(v_u_10, -v_u_10);
                            amax[11] = _fmax_11;
                            float v_u_11 = vals[12] * sexch[exch_row * 33 + 12];
                            vals[12] = v_u_11;
                            float _fmax_12 = fmaxf(v_u_11, -v_u_11);
                            amax[12] = _fmax_12;
                            float v_u_12 = vals[13] * sexch[exch_row * 33 + 13];
                            vals[13] = v_u_12;
                            float _fmax_13 = fmaxf(v_u_12, -v_u_12);
                            amax[13] = _fmax_13;
                            float v_u_13 = vals[14] * sexch[exch_row * 33 + 14];
                            vals[14] = v_u_13;
                            float _fmax_14 = fmaxf(v_u_13, -v_u_13);
                            amax[14] = _fmax_14;
                            float v_u_14 = vals[15] * sexch[exch_row * 33 + 15];
                            vals[15] = v_u_14;
                            float _fmax_15 = fmaxf(v_u_14, -v_u_14);
                            amax[15] = _fmax_15;
                            float v_u_15 = vals[16] * sexch[exch_row * 33 + 16];
                            vals[16] = v_u_15;
                            float _fmax_16 = fmaxf(v_u_15, -v_u_15);
                            amax[16] = _fmax_16;
                            float v_u_16 = vals[17] * sexch[exch_row * 33 + 17];
                            vals[17] = v_u_16;
                            float _fmax_17 = fmaxf(v_u_16, -v_u_16);
                            amax[17] = _fmax_17;
                            float v_u_17 = vals[18] * sexch[exch_row * 33 + 18];
                            vals[18] = v_u_17;
                            float _fmax_18 = fmaxf(v_u_17, -v_u_17);
                            amax[18] = _fmax_18;
                            float v_u_18 = vals[19] * sexch[exch_row * 33 + 19];
                            vals[19] = v_u_18;
                            float _fmax_19 = fmaxf(v_u_18, -v_u_18);
                            amax[19] = _fmax_19;
                            float v_u_19 = vals[20] * sexch[exch_row * 33 + 20];
                            vals[20] = v_u_19;
                            float _fmax_20 = fmaxf(v_u_19, -v_u_19);
                            amax[20] = _fmax_20;
                            float v_u_20 = vals[21] * sexch[exch_row * 33 + 21];
                            vals[21] = v_u_20;
                            float _fmax_21 = fmaxf(v_u_20, -v_u_20);
                            amax[21] = _fmax_21;
                            float v_u_21 = vals[22] * sexch[exch_row * 33 + 22];
                            vals[22] = v_u_21;
                            float _fmax_22 = fmaxf(v_u_21, -v_u_21);
                            amax[22] = _fmax_22;
                            float v_u_22 = vals[23] * sexch[exch_row * 33 + 23];
                            vals[23] = v_u_22;
                            float _fmax_23 = fmaxf(v_u_22, -v_u_22);
                            amax[23] = _fmax_23;
                            float v_u_23 = vals[24] * sexch[exch_row * 33 + 24];
                            vals[24] = v_u_23;
                            float _fmax_24 = fmaxf(v_u_23, -v_u_23);
                            amax[24] = _fmax_24;
                            float v_u_24 = vals[25] * sexch[exch_row * 33 + 25];
                            vals[25] = v_u_24;
                            float _fmax_25 = fmaxf(v_u_24, -v_u_24);
                            amax[25] = _fmax_25;
                            float v_u_25 = vals[26] * sexch[exch_row * 33 + 26];
                            vals[26] = v_u_25;
                            float _fmax_26 = fmaxf(v_u_25, -v_u_25);
                            amax[26] = _fmax_26;
                            float v_u_26 = vals[27] * sexch[exch_row * 33 + 27];
                            vals[27] = v_u_26;
                            float _fmax_27 = fmaxf(v_u_26, -v_u_26);
                            amax[27] = _fmax_27;
                            float v_u_27 = vals[28] * sexch[exch_row * 33 + 28];
                            vals[28] = v_u_27;
                            float _fmax_28 = fmaxf(v_u_27, -v_u_27);
                            amax[28] = _fmax_28;
                            float v_u_28 = vals[29] * sexch[exch_row * 33 + 29];
                            vals[29] = v_u_28;
                            float _fmax_29 = fmaxf(v_u_28, -v_u_28);
                            amax[29] = _fmax_29;
                            float v_u_29 = vals[30] * sexch[exch_row * 33 + 30];
                            vals[30] = v_u_29;
                            float _fmax_30 = fmaxf(v_u_29, -v_u_29);
                            amax[30] = _fmax_30;
                            float v_u_30 = vals[31] * sexch[exch_row * 33 + 31];
                            vals[31] = v_u_30;
                            float _fmax_31 = fmaxf(v_u_30, -v_u_30);
                            amax[31] = _fmax_31;
                            float a_c = amax[0];
                            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, a_c, 1);
                            float _fmax_32 = fmaxf(a_c, _shfl_xor_0);
                            a_c = _fmax_32;
                            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, a_c, 2);
                            float _fmax_33 = fmaxf(a_c, _shfl_xor_1);
                            a_c = _fmax_33;
                            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, a_c, 4);
                            float _fmax_34 = fmaxf(a_c, _shfl_xor_2);
                            a_c = _fmax_34;
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, a_c, 8);
                            float _fmax_35 = fmaxf(a_c, _shfl_xor_3);
                            a_c = _fmax_35;
                            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, a_c, 16);
                            float _fmax_36 = fmaxf(a_c, _shfl_xor_4);
                            a_c = _fmax_36;
                            amax[0] = a_c;
                            float a_c_31 = amax[1];
                            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, a_c_31, 1);
                            float _fmax_37 = fmaxf(a_c_31, _shfl_xor_5);
                            a_c_31 = _fmax_37;
                            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, a_c_31, 2);
                            float _fmax_38 = fmaxf(a_c_31, _shfl_xor_6);
                            a_c_31 = _fmax_38;
                            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, a_c_31, 4);
                            float _fmax_39 = fmaxf(a_c_31, _shfl_xor_7);
                            a_c_31 = _fmax_39;
                            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, a_c_31, 8);
                            float _fmax_40 = fmaxf(a_c_31, _shfl_xor_8);
                            a_c_31 = _fmax_40;
                            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, a_c_31, 16);
                            float _fmax_41 = fmaxf(a_c_31, _shfl_xor_9);
                            a_c_31 = _fmax_41;
                            amax[1] = a_c_31;
                            float a_c_32 = amax[2];
                            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, a_c_32, 1);
                            float _fmax_42 = fmaxf(a_c_32, _shfl_xor_10);
                            a_c_32 = _fmax_42;
                            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, a_c_32, 2);
                            float _fmax_43 = fmaxf(a_c_32, _shfl_xor_11);
                            a_c_32 = _fmax_43;
                            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, a_c_32, 4);
                            float _fmax_44 = fmaxf(a_c_32, _shfl_xor_12);
                            a_c_32 = _fmax_44;
                            float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, a_c_32, 8);
                            float _fmax_45 = fmaxf(a_c_32, _shfl_xor_13);
                            a_c_32 = _fmax_45;
                            float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, a_c_32, 16);
                            float _fmax_46 = fmaxf(a_c_32, _shfl_xor_14);
                            a_c_32 = _fmax_46;
                            amax[2] = a_c_32;
                            float a_c_33 = amax[3];
                            float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, a_c_33, 1);
                            float _fmax_47 = fmaxf(a_c_33, _shfl_xor_15);
                            a_c_33 = _fmax_47;
                            float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, a_c_33, 2);
                            float _fmax_48 = fmaxf(a_c_33, _shfl_xor_16);
                            a_c_33 = _fmax_48;
                            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, a_c_33, 4);
                            float _fmax_49 = fmaxf(a_c_33, _shfl_xor_17);
                            a_c_33 = _fmax_49;
                            float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, a_c_33, 8);
                            float _fmax_50 = fmaxf(a_c_33, _shfl_xor_18);
                            a_c_33 = _fmax_50;
                            float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, a_c_33, 16);
                            float _fmax_51 = fmaxf(a_c_33, _shfl_xor_19);
                            a_c_33 = _fmax_51;
                            amax[3] = a_c_33;
                            float a_c_34 = amax[4];
                            float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, a_c_34, 1);
                            float _fmax_52 = fmaxf(a_c_34, _shfl_xor_20);
                            a_c_34 = _fmax_52;
                            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, a_c_34, 2);
                            float _fmax_53 = fmaxf(a_c_34, _shfl_xor_21);
                            a_c_34 = _fmax_53;
                            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, a_c_34, 4);
                            float _fmax_54 = fmaxf(a_c_34, _shfl_xor_22);
                            a_c_34 = _fmax_54;
                            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, a_c_34, 8);
                            float _fmax_55 = fmaxf(a_c_34, _shfl_xor_23);
                            a_c_34 = _fmax_55;
                            float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, a_c_34, 16);
                            float _fmax_56 = fmaxf(a_c_34, _shfl_xor_24);
                            a_c_34 = _fmax_56;
                            amax[4] = a_c_34;
                            float a_c_35 = amax[5];
                            float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, a_c_35, 1);
                            float _fmax_57 = fmaxf(a_c_35, _shfl_xor_25);
                            a_c_35 = _fmax_57;
                            float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, a_c_35, 2);
                            float _fmax_58 = fmaxf(a_c_35, _shfl_xor_26);
                            a_c_35 = _fmax_58;
                            float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, a_c_35, 4);
                            float _fmax_59 = fmaxf(a_c_35, _shfl_xor_27);
                            a_c_35 = _fmax_59;
                            float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, a_c_35, 8);
                            float _fmax_60 = fmaxf(a_c_35, _shfl_xor_28);
                            a_c_35 = _fmax_60;
                            float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, a_c_35, 16);
                            float _fmax_61 = fmaxf(a_c_35, _shfl_xor_29);
                            a_c_35 = _fmax_61;
                            amax[5] = a_c_35;
                            float a_c_36 = amax[6];
                            float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, a_c_36, 1);
                            float _fmax_62 = fmaxf(a_c_36, _shfl_xor_30);
                            a_c_36 = _fmax_62;
                            float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, a_c_36, 2);
                            float _fmax_63 = fmaxf(a_c_36, _shfl_xor_31);
                            a_c_36 = _fmax_63;
                            float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, a_c_36, 4);
                            float _fmax_64 = fmaxf(a_c_36, _shfl_xor_32);
                            a_c_36 = _fmax_64;
                            float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, a_c_36, 8);
                            float _fmax_65 = fmaxf(a_c_36, _shfl_xor_33);
                            a_c_36 = _fmax_65;
                            float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, a_c_36, 16);
                            float _fmax_66 = fmaxf(a_c_36, _shfl_xor_34);
                            a_c_36 = _fmax_66;
                            amax[6] = a_c_36;
                            float a_c_37 = amax[7];
                            float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, a_c_37, 1);
                            float _fmax_67 = fmaxf(a_c_37, _shfl_xor_35);
                            a_c_37 = _fmax_67;
                            float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, a_c_37, 2);
                            float _fmax_68 = fmaxf(a_c_37, _shfl_xor_36);
                            a_c_37 = _fmax_68;
                            float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, a_c_37, 4);
                            float _fmax_69 = fmaxf(a_c_37, _shfl_xor_37);
                            a_c_37 = _fmax_69;
                            float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, a_c_37, 8);
                            float _fmax_70 = fmaxf(a_c_37, _shfl_xor_38);
                            a_c_37 = _fmax_70;
                            float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, a_c_37, 16);
                            float _fmax_71 = fmaxf(a_c_37, _shfl_xor_39);
                            a_c_37 = _fmax_71;
                            amax[7] = a_c_37;
                            float a_c_38 = amax[8];
                            float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, a_c_38, 1);
                            float _fmax_72 = fmaxf(a_c_38, _shfl_xor_40);
                            a_c_38 = _fmax_72;
                            float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, a_c_38, 2);
                            float _fmax_73 = fmaxf(a_c_38, _shfl_xor_41);
                            a_c_38 = _fmax_73;
                            float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, a_c_38, 4);
                            float _fmax_74 = fmaxf(a_c_38, _shfl_xor_42);
                            a_c_38 = _fmax_74;
                            float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, a_c_38, 8);
                            float _fmax_75 = fmaxf(a_c_38, _shfl_xor_43);
                            a_c_38 = _fmax_75;
                            float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, a_c_38, 16);
                            float _fmax_76 = fmaxf(a_c_38, _shfl_xor_44);
                            a_c_38 = _fmax_76;
                            amax[8] = a_c_38;
                            float a_c_39 = amax[9];
                            float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, a_c_39, 1);
                            float _fmax_77 = fmaxf(a_c_39, _shfl_xor_45);
                            a_c_39 = _fmax_77;
                            float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, a_c_39, 2);
                            float _fmax_78 = fmaxf(a_c_39, _shfl_xor_46);
                            a_c_39 = _fmax_78;
                            float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, a_c_39, 4);
                            float _fmax_79 = fmaxf(a_c_39, _shfl_xor_47);
                            a_c_39 = _fmax_79;
                            float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, a_c_39, 8);
                            float _fmax_80 = fmaxf(a_c_39, _shfl_xor_48);
                            a_c_39 = _fmax_80;
                            float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, a_c_39, 16);
                            float _fmax_81 = fmaxf(a_c_39, _shfl_xor_49);
                            a_c_39 = _fmax_81;
                            amax[9] = a_c_39;
                            float a_c_40 = amax[10];
                            float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, a_c_40, 1);
                            float _fmax_82 = fmaxf(a_c_40, _shfl_xor_50);
                            a_c_40 = _fmax_82;
                            float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, a_c_40, 2);
                            float _fmax_83 = fmaxf(a_c_40, _shfl_xor_51);
                            a_c_40 = _fmax_83;
                            float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, a_c_40, 4);
                            float _fmax_84 = fmaxf(a_c_40, _shfl_xor_52);
                            a_c_40 = _fmax_84;
                            float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, a_c_40, 8);
                            float _fmax_85 = fmaxf(a_c_40, _shfl_xor_53);
                            a_c_40 = _fmax_85;
                            float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, a_c_40, 16);
                            float _fmax_86 = fmaxf(a_c_40, _shfl_xor_54);
                            a_c_40 = _fmax_86;
                            amax[10] = a_c_40;
                            float a_c_41 = amax[11];
                            float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, a_c_41, 1);
                            float _fmax_87 = fmaxf(a_c_41, _shfl_xor_55);
                            a_c_41 = _fmax_87;
                            float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, a_c_41, 2);
                            float _fmax_88 = fmaxf(a_c_41, _shfl_xor_56);
                            a_c_41 = _fmax_88;
                            float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, a_c_41, 4);
                            float _fmax_89 = fmaxf(a_c_41, _shfl_xor_57);
                            a_c_41 = _fmax_89;
                            float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, a_c_41, 8);
                            float _fmax_90 = fmaxf(a_c_41, _shfl_xor_58);
                            a_c_41 = _fmax_90;
                            float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, a_c_41, 16);
                            float _fmax_91 = fmaxf(a_c_41, _shfl_xor_59);
                            a_c_41 = _fmax_91;
                            amax[11] = a_c_41;
                            float a_c_42 = amax[12];
                            float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, a_c_42, 1);
                            float _fmax_92 = fmaxf(a_c_42, _shfl_xor_60);
                            a_c_42 = _fmax_92;
                            float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, a_c_42, 2);
                            float _fmax_93 = fmaxf(a_c_42, _shfl_xor_61);
                            a_c_42 = _fmax_93;
                            float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, a_c_42, 4);
                            float _fmax_94 = fmaxf(a_c_42, _shfl_xor_62);
                            a_c_42 = _fmax_94;
                            float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, a_c_42, 8);
                            float _fmax_95 = fmaxf(a_c_42, _shfl_xor_63);
                            a_c_42 = _fmax_95;
                            float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, a_c_42, 16);
                            float _fmax_96 = fmaxf(a_c_42, _shfl_xor_64);
                            a_c_42 = _fmax_96;
                            amax[12] = a_c_42;
                            float a_c_43 = amax[13];
                            float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, a_c_43, 1);
                            float _fmax_97 = fmaxf(a_c_43, _shfl_xor_65);
                            a_c_43 = _fmax_97;
                            float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, a_c_43, 2);
                            float _fmax_98 = fmaxf(a_c_43, _shfl_xor_66);
                            a_c_43 = _fmax_98;
                            float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, a_c_43, 4);
                            float _fmax_99 = fmaxf(a_c_43, _shfl_xor_67);
                            a_c_43 = _fmax_99;
                            float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, a_c_43, 8);
                            float _fmax_100 = fmaxf(a_c_43, _shfl_xor_68);
                            a_c_43 = _fmax_100;
                            float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, a_c_43, 16);
                            float _fmax_101 = fmaxf(a_c_43, _shfl_xor_69);
                            a_c_43 = _fmax_101;
                            amax[13] = a_c_43;
                            float a_c_44 = amax[14];
                            float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, a_c_44, 1);
                            float _fmax_102 = fmaxf(a_c_44, _shfl_xor_70);
                            a_c_44 = _fmax_102;
                            float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, a_c_44, 2);
                            float _fmax_103 = fmaxf(a_c_44, _shfl_xor_71);
                            a_c_44 = _fmax_103;
                            float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, a_c_44, 4);
                            float _fmax_104 = fmaxf(a_c_44, _shfl_xor_72);
                            a_c_44 = _fmax_104;
                            float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, a_c_44, 8);
                            float _fmax_105 = fmaxf(a_c_44, _shfl_xor_73);
                            a_c_44 = _fmax_105;
                            float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, a_c_44, 16);
                            float _fmax_106 = fmaxf(a_c_44, _shfl_xor_74);
                            a_c_44 = _fmax_106;
                            amax[14] = a_c_44;
                            float a_c_45 = amax[15];
                            float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, a_c_45, 1);
                            float _fmax_107 = fmaxf(a_c_45, _shfl_xor_75);
                            a_c_45 = _fmax_107;
                            float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, a_c_45, 2);
                            float _fmax_108 = fmaxf(a_c_45, _shfl_xor_76);
                            a_c_45 = _fmax_108;
                            float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, a_c_45, 4);
                            float _fmax_109 = fmaxf(a_c_45, _shfl_xor_77);
                            a_c_45 = _fmax_109;
                            float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, a_c_45, 8);
                            float _fmax_110 = fmaxf(a_c_45, _shfl_xor_78);
                            a_c_45 = _fmax_110;
                            float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, a_c_45, 16);
                            float _fmax_111 = fmaxf(a_c_45, _shfl_xor_79);
                            a_c_45 = _fmax_111;
                            amax[15] = a_c_45;
                            float a_c_46 = amax[16];
                            float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, a_c_46, 1);
                            float _fmax_112 = fmaxf(a_c_46, _shfl_xor_80);
                            a_c_46 = _fmax_112;
                            float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, a_c_46, 2);
                            float _fmax_113 = fmaxf(a_c_46, _shfl_xor_81);
                            a_c_46 = _fmax_113;
                            float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, a_c_46, 4);
                            float _fmax_114 = fmaxf(a_c_46, _shfl_xor_82);
                            a_c_46 = _fmax_114;
                            float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, a_c_46, 8);
                            float _fmax_115 = fmaxf(a_c_46, _shfl_xor_83);
                            a_c_46 = _fmax_115;
                            float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, a_c_46, 16);
                            float _fmax_116 = fmaxf(a_c_46, _shfl_xor_84);
                            a_c_46 = _fmax_116;
                            amax[16] = a_c_46;
                            float a_c_47 = amax[17];
                            float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, a_c_47, 1);
                            float _fmax_117 = fmaxf(a_c_47, _shfl_xor_85);
                            a_c_47 = _fmax_117;
                            float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, a_c_47, 2);
                            float _fmax_118 = fmaxf(a_c_47, _shfl_xor_86);
                            a_c_47 = _fmax_118;
                            float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, a_c_47, 4);
                            float _fmax_119 = fmaxf(a_c_47, _shfl_xor_87);
                            a_c_47 = _fmax_119;
                            float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, a_c_47, 8);
                            float _fmax_120 = fmaxf(a_c_47, _shfl_xor_88);
                            a_c_47 = _fmax_120;
                            float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, a_c_47, 16);
                            float _fmax_121 = fmaxf(a_c_47, _shfl_xor_89);
                            a_c_47 = _fmax_121;
                            amax[17] = a_c_47;
                            float a_c_48 = amax[18];
                            float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, a_c_48, 1);
                            float _fmax_122 = fmaxf(a_c_48, _shfl_xor_90);
                            a_c_48 = _fmax_122;
                            float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, a_c_48, 2);
                            float _fmax_123 = fmaxf(a_c_48, _shfl_xor_91);
                            a_c_48 = _fmax_123;
                            float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, a_c_48, 4);
                            float _fmax_124 = fmaxf(a_c_48, _shfl_xor_92);
                            a_c_48 = _fmax_124;
                            float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, a_c_48, 8);
                            float _fmax_125 = fmaxf(a_c_48, _shfl_xor_93);
                            a_c_48 = _fmax_125;
                            float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, a_c_48, 16);
                            float _fmax_126 = fmaxf(a_c_48, _shfl_xor_94);
                            a_c_48 = _fmax_126;
                            amax[18] = a_c_48;
                            float a_c_49 = amax[19];
                            float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, a_c_49, 1);
                            float _fmax_127 = fmaxf(a_c_49, _shfl_xor_95);
                            a_c_49 = _fmax_127;
                            float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, a_c_49, 2);
                            float _fmax_128 = fmaxf(a_c_49, _shfl_xor_96);
                            a_c_49 = _fmax_128;
                            float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, a_c_49, 4);
                            float _fmax_129 = fmaxf(a_c_49, _shfl_xor_97);
                            a_c_49 = _fmax_129;
                            float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, a_c_49, 8);
                            float _fmax_130 = fmaxf(a_c_49, _shfl_xor_98);
                            a_c_49 = _fmax_130;
                            float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, a_c_49, 16);
                            float _fmax_131 = fmaxf(a_c_49, _shfl_xor_99);
                            a_c_49 = _fmax_131;
                            amax[19] = a_c_49;
                            float a_c_50 = amax[20];
                            float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, a_c_50, 1);
                            float _fmax_132 = fmaxf(a_c_50, _shfl_xor_100);
                            a_c_50 = _fmax_132;
                            float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, a_c_50, 2);
                            float _fmax_133 = fmaxf(a_c_50, _shfl_xor_101);
                            a_c_50 = _fmax_133;
                            float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, a_c_50, 4);
                            float _fmax_134 = fmaxf(a_c_50, _shfl_xor_102);
                            a_c_50 = _fmax_134;
                            float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, a_c_50, 8);
                            float _fmax_135 = fmaxf(a_c_50, _shfl_xor_103);
                            a_c_50 = _fmax_135;
                            float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, a_c_50, 16);
                            float _fmax_136 = fmaxf(a_c_50, _shfl_xor_104);
                            a_c_50 = _fmax_136;
                            amax[20] = a_c_50;
                            float a_c_51 = amax[21];
                            float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, a_c_51, 1);
                            float _fmax_137 = fmaxf(a_c_51, _shfl_xor_105);
                            a_c_51 = _fmax_137;
                            float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, a_c_51, 2);
                            float _fmax_138 = fmaxf(a_c_51, _shfl_xor_106);
                            a_c_51 = _fmax_138;
                            float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, a_c_51, 4);
                            float _fmax_139 = fmaxf(a_c_51, _shfl_xor_107);
                            a_c_51 = _fmax_139;
                            float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, a_c_51, 8);
                            float _fmax_140 = fmaxf(a_c_51, _shfl_xor_108);
                            a_c_51 = _fmax_140;
                            float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, a_c_51, 16);
                            float _fmax_141 = fmaxf(a_c_51, _shfl_xor_109);
                            a_c_51 = _fmax_141;
                            amax[21] = a_c_51;
                            float a_c_52 = amax[22];
                            float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, a_c_52, 1);
                            float _fmax_142 = fmaxf(a_c_52, _shfl_xor_110);
                            a_c_52 = _fmax_142;
                            float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, a_c_52, 2);
                            float _fmax_143 = fmaxf(a_c_52, _shfl_xor_111);
                            a_c_52 = _fmax_143;
                            float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, a_c_52, 4);
                            float _fmax_144 = fmaxf(a_c_52, _shfl_xor_112);
                            a_c_52 = _fmax_144;
                            float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, a_c_52, 8);
                            float _fmax_145 = fmaxf(a_c_52, _shfl_xor_113);
                            a_c_52 = _fmax_145;
                            float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, a_c_52, 16);
                            float _fmax_146 = fmaxf(a_c_52, _shfl_xor_114);
                            a_c_52 = _fmax_146;
                            amax[22] = a_c_52;
                            float a_c_53 = amax[23];
                            float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, a_c_53, 1);
                            float _fmax_147 = fmaxf(a_c_53, _shfl_xor_115);
                            a_c_53 = _fmax_147;
                            float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, a_c_53, 2);
                            float _fmax_148 = fmaxf(a_c_53, _shfl_xor_116);
                            a_c_53 = _fmax_148;
                            float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, a_c_53, 4);
                            float _fmax_149 = fmaxf(a_c_53, _shfl_xor_117);
                            a_c_53 = _fmax_149;
                            float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, a_c_53, 8);
                            float _fmax_150 = fmaxf(a_c_53, _shfl_xor_118);
                            a_c_53 = _fmax_150;
                            float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, a_c_53, 16);
                            float _fmax_151 = fmaxf(a_c_53, _shfl_xor_119);
                            a_c_53 = _fmax_151;
                            amax[23] = a_c_53;
                            float a_c_54 = amax[24];
                            float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, a_c_54, 1);
                            float _fmax_152 = fmaxf(a_c_54, _shfl_xor_120);
                            a_c_54 = _fmax_152;
                            float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, a_c_54, 2);
                            float _fmax_153 = fmaxf(a_c_54, _shfl_xor_121);
                            a_c_54 = _fmax_153;
                            float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, a_c_54, 4);
                            float _fmax_154 = fmaxf(a_c_54, _shfl_xor_122);
                            a_c_54 = _fmax_154;
                            float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, a_c_54, 8);
                            float _fmax_155 = fmaxf(a_c_54, _shfl_xor_123);
                            a_c_54 = _fmax_155;
                            float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, a_c_54, 16);
                            float _fmax_156 = fmaxf(a_c_54, _shfl_xor_124);
                            a_c_54 = _fmax_156;
                            amax[24] = a_c_54;
                            float a_c_55 = amax[25];
                            float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, a_c_55, 1);
                            float _fmax_157 = fmaxf(a_c_55, _shfl_xor_125);
                            a_c_55 = _fmax_157;
                            float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, a_c_55, 2);
                            float _fmax_158 = fmaxf(a_c_55, _shfl_xor_126);
                            a_c_55 = _fmax_158;
                            float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, a_c_55, 4);
                            float _fmax_159 = fmaxf(a_c_55, _shfl_xor_127);
                            a_c_55 = _fmax_159;
                            float _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, a_c_55, 8);
                            float _fmax_160 = fmaxf(a_c_55, _shfl_xor_128);
                            a_c_55 = _fmax_160;
                            float _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, a_c_55, 16);
                            float _fmax_161 = fmaxf(a_c_55, _shfl_xor_129);
                            a_c_55 = _fmax_161;
                            amax[25] = a_c_55;
                            float a_c_56 = amax[26];
                            float _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, a_c_56, 1);
                            float _fmax_162 = fmaxf(a_c_56, _shfl_xor_130);
                            a_c_56 = _fmax_162;
                            float _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, a_c_56, 2);
                            float _fmax_163 = fmaxf(a_c_56, _shfl_xor_131);
                            a_c_56 = _fmax_163;
                            float _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, a_c_56, 4);
                            float _fmax_164 = fmaxf(a_c_56, _shfl_xor_132);
                            a_c_56 = _fmax_164;
                            float _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, a_c_56, 8);
                            float _fmax_165 = fmaxf(a_c_56, _shfl_xor_133);
                            a_c_56 = _fmax_165;
                            float _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, a_c_56, 16);
                            float _fmax_166 = fmaxf(a_c_56, _shfl_xor_134);
                            a_c_56 = _fmax_166;
                            amax[26] = a_c_56;
                            float a_c_57 = amax[27];
                            float _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, a_c_57, 1);
                            float _fmax_167 = fmaxf(a_c_57, _shfl_xor_135);
                            a_c_57 = _fmax_167;
                            float _shfl_xor_136 = __shfl_xor_sync(0xFFFFFFFF, a_c_57, 2);
                            float _fmax_168 = fmaxf(a_c_57, _shfl_xor_136);
                            a_c_57 = _fmax_168;
                            float _shfl_xor_137 = __shfl_xor_sync(0xFFFFFFFF, a_c_57, 4);
                            float _fmax_169 = fmaxf(a_c_57, _shfl_xor_137);
                            a_c_57 = _fmax_169;
                            float _shfl_xor_138 = __shfl_xor_sync(0xFFFFFFFF, a_c_57, 8);
                            float _fmax_170 = fmaxf(a_c_57, _shfl_xor_138);
                            a_c_57 = _fmax_170;
                            float _shfl_xor_139 = __shfl_xor_sync(0xFFFFFFFF, a_c_57, 16);
                            float _fmax_171 = fmaxf(a_c_57, _shfl_xor_139);
                            a_c_57 = _fmax_171;
                            amax[27] = a_c_57;
                            float a_c_58 = amax[28];
                            float _shfl_xor_140 = __shfl_xor_sync(0xFFFFFFFF, a_c_58, 1);
                            float _fmax_172 = fmaxf(a_c_58, _shfl_xor_140);
                            a_c_58 = _fmax_172;
                            float _shfl_xor_141 = __shfl_xor_sync(0xFFFFFFFF, a_c_58, 2);
                            float _fmax_173 = fmaxf(a_c_58, _shfl_xor_141);
                            a_c_58 = _fmax_173;
                            float _shfl_xor_142 = __shfl_xor_sync(0xFFFFFFFF, a_c_58, 4);
                            float _fmax_174 = fmaxf(a_c_58, _shfl_xor_142);
                            a_c_58 = _fmax_174;
                            float _shfl_xor_143 = __shfl_xor_sync(0xFFFFFFFF, a_c_58, 8);
                            float _fmax_175 = fmaxf(a_c_58, _shfl_xor_143);
                            a_c_58 = _fmax_175;
                            float _shfl_xor_144 = __shfl_xor_sync(0xFFFFFFFF, a_c_58, 16);
                            float _fmax_176 = fmaxf(a_c_58, _shfl_xor_144);
                            a_c_58 = _fmax_176;
                            amax[28] = a_c_58;
                            float a_c_59 = amax[29];
                            float _shfl_xor_145 = __shfl_xor_sync(0xFFFFFFFF, a_c_59, 1);
                            float _fmax_177 = fmaxf(a_c_59, _shfl_xor_145);
                            a_c_59 = _fmax_177;
                            float _shfl_xor_146 = __shfl_xor_sync(0xFFFFFFFF, a_c_59, 2);
                            float _fmax_178 = fmaxf(a_c_59, _shfl_xor_146);
                            a_c_59 = _fmax_178;
                            float _shfl_xor_147 = __shfl_xor_sync(0xFFFFFFFF, a_c_59, 4);
                            float _fmax_179 = fmaxf(a_c_59, _shfl_xor_147);
                            a_c_59 = _fmax_179;
                            float _shfl_xor_148 = __shfl_xor_sync(0xFFFFFFFF, a_c_59, 8);
                            float _fmax_180 = fmaxf(a_c_59, _shfl_xor_148);
                            a_c_59 = _fmax_180;
                            float _shfl_xor_149 = __shfl_xor_sync(0xFFFFFFFF, a_c_59, 16);
                            float _fmax_181 = fmaxf(a_c_59, _shfl_xor_149);
                            a_c_59 = _fmax_181;
                            amax[29] = a_c_59;
                            float a_c_60 = amax[30];
                            float _shfl_xor_150 = __shfl_xor_sync(0xFFFFFFFF, a_c_60, 1);
                            float _fmax_182 = fmaxf(a_c_60, _shfl_xor_150);
                            a_c_60 = _fmax_182;
                            float _shfl_xor_151 = __shfl_xor_sync(0xFFFFFFFF, a_c_60, 2);
                            float _fmax_183 = fmaxf(a_c_60, _shfl_xor_151);
                            a_c_60 = _fmax_183;
                            float _shfl_xor_152 = __shfl_xor_sync(0xFFFFFFFF, a_c_60, 4);
                            float _fmax_184 = fmaxf(a_c_60, _shfl_xor_152);
                            a_c_60 = _fmax_184;
                            float _shfl_xor_153 = __shfl_xor_sync(0xFFFFFFFF, a_c_60, 8);
                            float _fmax_185 = fmaxf(a_c_60, _shfl_xor_153);
                            a_c_60 = _fmax_185;
                            float _shfl_xor_154 = __shfl_xor_sync(0xFFFFFFFF, a_c_60, 16);
                            float _fmax_186 = fmaxf(a_c_60, _shfl_xor_154);
                            a_c_60 = _fmax_186;
                            amax[30] = a_c_60;
                            float a_c_61 = amax[31];
                            float _shfl_xor_155 = __shfl_xor_sync(0xFFFFFFFF, a_c_61, 1);
                            float _fmax_187 = fmaxf(a_c_61, _shfl_xor_155);
                            a_c_61 = _fmax_187;
                            float _shfl_xor_156 = __shfl_xor_sync(0xFFFFFFFF, a_c_61, 2);
                            float _fmax_188 = fmaxf(a_c_61, _shfl_xor_156);
                            a_c_61 = _fmax_188;
                            float _shfl_xor_157 = __shfl_xor_sync(0xFFFFFFFF, a_c_61, 4);
                            float _fmax_189 = fmaxf(a_c_61, _shfl_xor_157);
                            a_c_61 = _fmax_189;
                            float _shfl_xor_158 = __shfl_xor_sync(0xFFFFFFFF, a_c_61, 8);
                            float _fmax_190 = fmaxf(a_c_61, _shfl_xor_158);
                            a_c_61 = _fmax_190;
                            float _shfl_xor_159 = __shfl_xor_sync(0xFFFFFFFF, a_c_61, 16);
                            float _fmax_191 = fmaxf(a_c_61, _shfl_xor_159);
                            a_c_61 = _fmax_191;
                            amax[31] = a_c_61;
                            int prow_e = row_base;
                            if (prow_e < mn_limit) {
                                uint16_t _ue8m0x2_f32_0;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_0) : "f"(zero_f32), "f"(amax[0] * inv_fp8_max));
                                int code_full = (int)_ue8m0x2_f32_0;
                                int code = code_full & 255;
                                int _max_0 = ((254 - code) > (0) ? (254 - code) : (0));
                                unsigned int inv_bits = (unsigned int)(_max_0 << 23);
                                float inv_scale = __uint_as_float(inv_bits) * (float)(code != 0);
                                float q_val = vals[0] * inv_scale;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32 = (unsigned int)code;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32);
                                }
                            }
                            int prow_e_62 = row_base + 1;
                            if (prow_e_62 < mn_limit) {
                                uint16_t _ue8m0x2_f32_1;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_1) : "f"(zero_f32), "f"(amax[1] * inv_fp8_max));
                                int code_full_1 = (int)_ue8m0x2_f32_1;
                                int code_1 = code_full_1 & 255;
                                int _max_1 = ((254 - code_1) > (0) ? (254 - code_1) : (0));
                                unsigned int inv_bits_1 = (unsigned int)(_max_1 << 23);
                                float inv_scale_1 = __uint_as_float(inv_bits_1) * (float)(code_1 != 0);
                                float q_val_1 = vals[1] * inv_scale_1;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_1));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_62 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_1 = (unsigned int)code_1;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_62 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_1);
                                }
                            }
                            int prow_e_63 = row_base + 2;
                            if (prow_e_63 < mn_limit) {
                                uint16_t _ue8m0x2_f32_2;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_2) : "f"(zero_f32), "f"(amax[2] * inv_fp8_max));
                                int code_full_2 = (int)_ue8m0x2_f32_2;
                                int code_2 = code_full_2 & 255;
                                int _max_2 = ((254 - code_2) > (0) ? (254 - code_2) : (0));
                                unsigned int inv_bits_2 = (unsigned int)(_max_2 << 23);
                                float inv_scale_2 = __uint_as_float(inv_bits_2) * (float)(code_2 != 0);
                                float q_val_2 = vals[2] * inv_scale_2;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_2));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_63 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_2 = (unsigned int)code_2;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_63 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_2);
                                }
                            }
                            int prow_e_64 = row_base + 3;
                            if (prow_e_64 < mn_limit) {
                                uint16_t _ue8m0x2_f32_3;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_3) : "f"(zero_f32), "f"(amax[3] * inv_fp8_max));
                                int code_full_3 = (int)_ue8m0x2_f32_3;
                                int code_3 = code_full_3 & 255;
                                int _max_3 = ((254 - code_3) > (0) ? (254 - code_3) : (0));
                                unsigned int inv_bits_3 = (unsigned int)(_max_3 << 23);
                                float inv_scale_3 = __uint_as_float(inv_bits_3) * (float)(code_3 != 0);
                                float q_val_3 = vals[3] * inv_scale_3;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_3));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_64 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_3 = (unsigned int)code_3;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_64 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_3);
                                }
                            }
                            int prow_e_65 = row_base + 4;
                            if (prow_e_65 < mn_limit) {
                                uint16_t _ue8m0x2_f32_4;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_4) : "f"(zero_f32), "f"(amax[4] * inv_fp8_max));
                                int code_full_4 = (int)_ue8m0x2_f32_4;
                                int code_4 = code_full_4 & 255;
                                int _max_4 = ((254 - code_4) > (0) ? (254 - code_4) : (0));
                                unsigned int inv_bits_4 = (unsigned int)(_max_4 << 23);
                                float inv_scale_4 = __uint_as_float(inv_bits_4) * (float)(code_4 != 0);
                                float q_val_4 = vals[4] * inv_scale_4;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_4));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_65 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_4 = (unsigned int)code_4;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_65 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_4);
                                }
                            }
                            int prow_e_66 = row_base + 5;
                            if (prow_e_66 < mn_limit) {
                                uint16_t _ue8m0x2_f32_5;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_5) : "f"(zero_f32), "f"(amax[5] * inv_fp8_max));
                                int code_full_5 = (int)_ue8m0x2_f32_5;
                                int code_5 = code_full_5 & 255;
                                int _max_5 = ((254 - code_5) > (0) ? (254 - code_5) : (0));
                                unsigned int inv_bits_5 = (unsigned int)(_max_5 << 23);
                                float inv_scale_5 = __uint_as_float(inv_bits_5) * (float)(code_5 != 0);
                                float q_val_5 = vals[5] * inv_scale_5;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_5));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_66 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_5 = (unsigned int)code_5;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_66 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_5);
                                }
                            }
                            int prow_e_67 = row_base + 6;
                            if (prow_e_67 < mn_limit) {
                                uint16_t _ue8m0x2_f32_6;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_6) : "f"(zero_f32), "f"(amax[6] * inv_fp8_max));
                                int code_full_6 = (int)_ue8m0x2_f32_6;
                                int code_6 = code_full_6 & 255;
                                int _max_6 = ((254 - code_6) > (0) ? (254 - code_6) : (0));
                                unsigned int inv_bits_6 = (unsigned int)(_max_6 << 23);
                                float inv_scale_6 = __uint_as_float(inv_bits_6) * (float)(code_6 != 0);
                                float q_val_6 = vals[6] * inv_scale_6;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_6));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_67 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_6 = (unsigned int)code_6;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_67 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_6);
                                }
                            }
                            int prow_e_68 = row_base + 7;
                            if (prow_e_68 < mn_limit) {
                                uint16_t _ue8m0x2_f32_7;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_7) : "f"(zero_f32), "f"(amax[7] * inv_fp8_max));
                                int code_full_7 = (int)_ue8m0x2_f32_7;
                                int code_7 = code_full_7 & 255;
                                int _max_7 = ((254 - code_7) > (0) ? (254 - code_7) : (0));
                                unsigned int inv_bits_7 = (unsigned int)(_max_7 << 23);
                                float inv_scale_7 = __uint_as_float(inv_bits_7) * (float)(code_7 != 0);
                                float q_val_7 = vals[7] * inv_scale_7;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_7));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_68 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_7 = (unsigned int)code_7;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_68 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_7);
                                }
                            }
                            int prow_e_69 = row_base + 8;
                            if (prow_e_69 < mn_limit) {
                                uint16_t _ue8m0x2_f32_8;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_8) : "f"(zero_f32), "f"(amax[8] * inv_fp8_max));
                                int code_full_8 = (int)_ue8m0x2_f32_8;
                                int code_8 = code_full_8 & 255;
                                int _max_8 = ((254 - code_8) > (0) ? (254 - code_8) : (0));
                                unsigned int inv_bits_8 = (unsigned int)(_max_8 << 23);
                                float inv_scale_8 = __uint_as_float(inv_bits_8) * (float)(code_8 != 0);
                                float q_val_8 = vals[8] * inv_scale_8;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_8));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_69 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_8 = (unsigned int)code_8;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_69 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_8);
                                }
                            }
                            int prow_e_70 = row_base + 9;
                            if (prow_e_70 < mn_limit) {
                                uint16_t _ue8m0x2_f32_9;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_9) : "f"(zero_f32), "f"(amax[9] * inv_fp8_max));
                                int code_full_9 = (int)_ue8m0x2_f32_9;
                                int code_9 = code_full_9 & 255;
                                int _max_9 = ((254 - code_9) > (0) ? (254 - code_9) : (0));
                                unsigned int inv_bits_9 = (unsigned int)(_max_9 << 23);
                                float inv_scale_9 = __uint_as_float(inv_bits_9) * (float)(code_9 != 0);
                                float q_val_9 = vals[9] * inv_scale_9;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_9));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_70 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_9 = (unsigned int)code_9;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_70 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_9);
                                }
                            }
                            int prow_e_71 = row_base + 10;
                            if (prow_e_71 < mn_limit) {
                                uint16_t _ue8m0x2_f32_10;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_10) : "f"(zero_f32), "f"(amax[10] * inv_fp8_max));
                                int code_full_10 = (int)_ue8m0x2_f32_10;
                                int code_10 = code_full_10 & 255;
                                int _max_10 = ((254 - code_10) > (0) ? (254 - code_10) : (0));
                                unsigned int inv_bits_10 = (unsigned int)(_max_10 << 23);
                                float inv_scale_10 = __uint_as_float(inv_bits_10) * (float)(code_10 != 0);
                                float q_val_10 = vals[10] * inv_scale_10;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_10));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_71 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_10 = (unsigned int)code_10;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_71 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_10);
                                }
                            }
                            int prow_e_72 = row_base + 11;
                            if (prow_e_72 < mn_limit) {
                                uint16_t _ue8m0x2_f32_11;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_11) : "f"(zero_f32), "f"(amax[11] * inv_fp8_max));
                                int code_full_11 = (int)_ue8m0x2_f32_11;
                                int code_11 = code_full_11 & 255;
                                int _max_11 = ((254 - code_11) > (0) ? (254 - code_11) : (0));
                                unsigned int inv_bits_11 = (unsigned int)(_max_11 << 23);
                                float inv_scale_11 = __uint_as_float(inv_bits_11) * (float)(code_11 != 0);
                                float q_val_11 = vals[11] * inv_scale_11;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_11));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_72 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_11 = (unsigned int)code_11;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_72 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_11);
                                }
                            }
                            int prow_e_73 = row_base + 12;
                            if (prow_e_73 < mn_limit) {
                                uint16_t _ue8m0x2_f32_12;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_12) : "f"(zero_f32), "f"(amax[12] * inv_fp8_max));
                                int code_full_12 = (int)_ue8m0x2_f32_12;
                                int code_12 = code_full_12 & 255;
                                int _max_12 = ((254 - code_12) > (0) ? (254 - code_12) : (0));
                                unsigned int inv_bits_12 = (unsigned int)(_max_12 << 23);
                                float inv_scale_12 = __uint_as_float(inv_bits_12) * (float)(code_12 != 0);
                                float q_val_12 = vals[12] * inv_scale_12;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_12));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_73 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_12 = (unsigned int)code_12;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_73 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_12);
                                }
                            }
                            int prow_e_74 = row_base + 13;
                            if (prow_e_74 < mn_limit) {
                                uint16_t _ue8m0x2_f32_13;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_13) : "f"(zero_f32), "f"(amax[13] * inv_fp8_max));
                                int code_full_13 = (int)_ue8m0x2_f32_13;
                                int code_13 = code_full_13 & 255;
                                int _max_13 = ((254 - code_13) > (0) ? (254 - code_13) : (0));
                                unsigned int inv_bits_13 = (unsigned int)(_max_13 << 23);
                                float inv_scale_13 = __uint_as_float(inv_bits_13) * (float)(code_13 != 0);
                                float q_val_13 = vals[13] * inv_scale_13;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_13));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_74 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_13 = (unsigned int)code_13;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_74 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_13);
                                }
                            }
                            int prow_e_75 = row_base + 14;
                            if (prow_e_75 < mn_limit) {
                                uint16_t _ue8m0x2_f32_14;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_14) : "f"(zero_f32), "f"(amax[14] * inv_fp8_max));
                                int code_full_14 = (int)_ue8m0x2_f32_14;
                                int code_14 = code_full_14 & 255;
                                int _max_14 = ((254 - code_14) > (0) ? (254 - code_14) : (0));
                                unsigned int inv_bits_14 = (unsigned int)(_max_14 << 23);
                                float inv_scale_14 = __uint_as_float(inv_bits_14) * (float)(code_14 != 0);
                                float q_val_14 = vals[14] * inv_scale_14;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_14));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_75 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_14 = (unsigned int)code_14;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_75 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_14);
                                }
                            }
                            int prow_e_76 = row_base + 15;
                            if (prow_e_76 < mn_limit) {
                                uint16_t _ue8m0x2_f32_15;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_15) : "f"(zero_f32), "f"(amax[15] * inv_fp8_max));
                                int code_full_15 = (int)_ue8m0x2_f32_15;
                                int code_15 = code_full_15 & 255;
                                int _max_15 = ((254 - code_15) > (0) ? (254 - code_15) : (0));
                                unsigned int inv_bits_15 = (unsigned int)(_max_15 << 23);
                                float inv_scale_15 = __uint_as_float(inv_bits_15) * (float)(code_15 != 0);
                                float q_val_15 = vals[15] * inv_scale_15;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_15));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_76 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_15 = (unsigned int)code_15;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_76 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_15);
                                }
                            }
                            int prow_e_77 = row_base + 16;
                            if (prow_e_77 < mn_limit) {
                                uint16_t _ue8m0x2_f32_16;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_16) : "f"(zero_f32), "f"(amax[16] * inv_fp8_max));
                                int code_full_16 = (int)_ue8m0x2_f32_16;
                                int code_16 = code_full_16 & 255;
                                int _max_16 = ((254 - code_16) > (0) ? (254 - code_16) : (0));
                                unsigned int inv_bits_16 = (unsigned int)(_max_16 << 23);
                                float inv_scale_16 = __uint_as_float(inv_bits_16) * (float)(code_16 != 0);
                                float q_val_16 = vals[16] * inv_scale_16;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_16));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_77 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_16 = (unsigned int)code_16;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_77 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_16);
                                }
                            }
                            int prow_e_78 = row_base + 17;
                            if (prow_e_78 < mn_limit) {
                                uint16_t _ue8m0x2_f32_17;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_17) : "f"(zero_f32), "f"(amax[17] * inv_fp8_max));
                                int code_full_17 = (int)_ue8m0x2_f32_17;
                                int code_17 = code_full_17 & 255;
                                int _max_17 = ((254 - code_17) > (0) ? (254 - code_17) : (0));
                                unsigned int inv_bits_17 = (unsigned int)(_max_17 << 23);
                                float inv_scale_17 = __uint_as_float(inv_bits_17) * (float)(code_17 != 0);
                                float q_val_17 = vals[17] * inv_scale_17;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_17));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_78 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_17 = (unsigned int)code_17;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_78 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_17);
                                }
                            }
                            int prow_e_79 = row_base + 18;
                            if (prow_e_79 < mn_limit) {
                                uint16_t _ue8m0x2_f32_18;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_18) : "f"(zero_f32), "f"(amax[18] * inv_fp8_max));
                                int code_full_18 = (int)_ue8m0x2_f32_18;
                                int code_18 = code_full_18 & 255;
                                int _max_18 = ((254 - code_18) > (0) ? (254 - code_18) : (0));
                                unsigned int inv_bits_18 = (unsigned int)(_max_18 << 23);
                                float inv_scale_18 = __uint_as_float(inv_bits_18) * (float)(code_18 != 0);
                                float q_val_18 = vals[18] * inv_scale_18;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_18));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_79 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_18 = (unsigned int)code_18;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_79 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_18);
                                }
                            }
                            int prow_e_80 = row_base + 19;
                            if (prow_e_80 < mn_limit) {
                                uint16_t _ue8m0x2_f32_19;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_19) : "f"(zero_f32), "f"(amax[19] * inv_fp8_max));
                                int code_full_19 = (int)_ue8m0x2_f32_19;
                                int code_19 = code_full_19 & 255;
                                int _max_19 = ((254 - code_19) > (0) ? (254 - code_19) : (0));
                                unsigned int inv_bits_19 = (unsigned int)(_max_19 << 23);
                                float inv_scale_19 = __uint_as_float(inv_bits_19) * (float)(code_19 != 0);
                                float q_val_19 = vals[19] * inv_scale_19;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_19));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_80 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_19 = (unsigned int)code_19;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_80 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_19);
                                }
                            }
                            int prow_e_81 = row_base + 20;
                            if (prow_e_81 < mn_limit) {
                                uint16_t _ue8m0x2_f32_20;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_20) : "f"(zero_f32), "f"(amax[20] * inv_fp8_max));
                                int code_full_20 = (int)_ue8m0x2_f32_20;
                                int code_20 = code_full_20 & 255;
                                int _max_20 = ((254 - code_20) > (0) ? (254 - code_20) : (0));
                                unsigned int inv_bits_20 = (unsigned int)(_max_20 << 23);
                                float inv_scale_20 = __uint_as_float(inv_bits_20) * (float)(code_20 != 0);
                                float q_val_20 = vals[20] * inv_scale_20;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_20));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_81 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_20 = (unsigned int)code_20;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_81 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_20);
                                }
                            }
                            int prow_e_82 = row_base + 21;
                            if (prow_e_82 < mn_limit) {
                                uint16_t _ue8m0x2_f32_21;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_21) : "f"(zero_f32), "f"(amax[21] * inv_fp8_max));
                                int code_full_21 = (int)_ue8m0x2_f32_21;
                                int code_21 = code_full_21 & 255;
                                int _max_21 = ((254 - code_21) > (0) ? (254 - code_21) : (0));
                                unsigned int inv_bits_21 = (unsigned int)(_max_21 << 23);
                                float inv_scale_21 = __uint_as_float(inv_bits_21) * (float)(code_21 != 0);
                                float q_val_21 = vals[21] * inv_scale_21;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_21));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_82 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_21 = (unsigned int)code_21;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_82 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_21);
                                }
                            }
                            int prow_e_83 = row_base + 22;
                            if (prow_e_83 < mn_limit) {
                                uint16_t _ue8m0x2_f32_22;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_22) : "f"(zero_f32), "f"(amax[22] * inv_fp8_max));
                                int code_full_22 = (int)_ue8m0x2_f32_22;
                                int code_22 = code_full_22 & 255;
                                int _max_22 = ((254 - code_22) > (0) ? (254 - code_22) : (0));
                                unsigned int inv_bits_22 = (unsigned int)(_max_22 << 23);
                                float inv_scale_22 = __uint_as_float(inv_bits_22) * (float)(code_22 != 0);
                                float q_val_22 = vals[22] * inv_scale_22;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_22));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_83 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_22 = (unsigned int)code_22;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_83 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_22);
                                }
                            }
                            int prow_e_84 = row_base + 23;
                            if (prow_e_84 < mn_limit) {
                                uint16_t _ue8m0x2_f32_23;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_23) : "f"(zero_f32), "f"(amax[23] * inv_fp8_max));
                                int code_full_23 = (int)_ue8m0x2_f32_23;
                                int code_23 = code_full_23 & 255;
                                int _max_23 = ((254 - code_23) > (0) ? (254 - code_23) : (0));
                                unsigned int inv_bits_23 = (unsigned int)(_max_23 << 23);
                                float inv_scale_23 = __uint_as_float(inv_bits_23) * (float)(code_23 != 0);
                                float q_val_23 = vals[23] * inv_scale_23;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_23));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_84 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_23 = (unsigned int)code_23;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_84 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_23);
                                }
                            }
                            int prow_e_85 = row_base + 24;
                            if (prow_e_85 < mn_limit) {
                                uint16_t _ue8m0x2_f32_24;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_24) : "f"(zero_f32), "f"(amax[24] * inv_fp8_max));
                                int code_full_24 = (int)_ue8m0x2_f32_24;
                                int code_24 = code_full_24 & 255;
                                int _max_24 = ((254 - code_24) > (0) ? (254 - code_24) : (0));
                                unsigned int inv_bits_24 = (unsigned int)(_max_24 << 23);
                                float inv_scale_24 = __uint_as_float(inv_bits_24) * (float)(code_24 != 0);
                                float q_val_24 = vals[24] * inv_scale_24;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_24));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_85 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_24 = (unsigned int)code_24;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_85 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_24);
                                }
                            }
                            int prow_e_86 = row_base + 25;
                            if (prow_e_86 < mn_limit) {
                                uint16_t _ue8m0x2_f32_25;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_25) : "f"(zero_f32), "f"(amax[25] * inv_fp8_max));
                                int code_full_25 = (int)_ue8m0x2_f32_25;
                                int code_25 = code_full_25 & 255;
                                int _max_25 = ((254 - code_25) > (0) ? (254 - code_25) : (0));
                                unsigned int inv_bits_25 = (unsigned int)(_max_25 << 23);
                                float inv_scale_25 = __uint_as_float(inv_bits_25) * (float)(code_25 != 0);
                                float q_val_25 = vals[25] * inv_scale_25;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_25));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_86 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_25 = (unsigned int)code_25;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_86 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_25);
                                }
                            }
                            int prow_e_87 = row_base + 26;
                            if (prow_e_87 < mn_limit) {
                                uint16_t _ue8m0x2_f32_26;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_26) : "f"(zero_f32), "f"(amax[26] * inv_fp8_max));
                                int code_full_26 = (int)_ue8m0x2_f32_26;
                                int code_26 = code_full_26 & 255;
                                int _max_26 = ((254 - code_26) > (0) ? (254 - code_26) : (0));
                                unsigned int inv_bits_26 = (unsigned int)(_max_26 << 23);
                                float inv_scale_26 = __uint_as_float(inv_bits_26) * (float)(code_26 != 0);
                                float q_val_26 = vals[26] * inv_scale_26;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_26));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_87 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_26 = (unsigned int)code_26;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_87 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_26);
                                }
                            }
                            int prow_e_88 = row_base + 27;
                            if (prow_e_88 < mn_limit) {
                                uint16_t _ue8m0x2_f32_27;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_27) : "f"(zero_f32), "f"(amax[27] * inv_fp8_max));
                                int code_full_27 = (int)_ue8m0x2_f32_27;
                                int code_27 = code_full_27 & 255;
                                int _max_27 = ((254 - code_27) > (0) ? (254 - code_27) : (0));
                                unsigned int inv_bits_27 = (unsigned int)(_max_27 << 23);
                                float inv_scale_27 = __uint_as_float(inv_bits_27) * (float)(code_27 != 0);
                                float q_val_27 = vals[27] * inv_scale_27;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_27));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_88 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_27 = (unsigned int)code_27;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_88 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_27);
                                }
                            }
                            int prow_e_89 = row_base + 28;
                            if (prow_e_89 < mn_limit) {
                                uint16_t _ue8m0x2_f32_28;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_28) : "f"(zero_f32), "f"(amax[28] * inv_fp8_max));
                                int code_full_28 = (int)_ue8m0x2_f32_28;
                                int code_28 = code_full_28 & 255;
                                int _max_28 = ((254 - code_28) > (0) ? (254 - code_28) : (0));
                                unsigned int inv_bits_28 = (unsigned int)(_max_28 << 23);
                                float inv_scale_28 = __uint_as_float(inv_bits_28) * (float)(code_28 != 0);
                                float q_val_28 = vals[28] * inv_scale_28;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_28));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_89 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_28 = (unsigned int)code_28;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_89 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_28);
                                }
                            }
                            int prow_e_90 = row_base + 29;
                            if (prow_e_90 < mn_limit) {
                                uint16_t _ue8m0x2_f32_29;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_29) : "f"(zero_f32), "f"(amax[29] * inv_fp8_max));
                                int code_full_29 = (int)_ue8m0x2_f32_29;
                                int code_29 = code_full_29 & 255;
                                int _max_29 = ((254 - code_29) > (0) ? (254 - code_29) : (0));
                                unsigned int inv_bits_29 = (unsigned int)(_max_29 << 23);
                                float inv_scale_29 = __uint_as_float(inv_bits_29) * (float)(code_29 != 0);
                                float q_val_29 = vals[29] * inv_scale_29;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_29));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_90 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_29 = (unsigned int)code_29;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_90 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_29);
                                }
                            }
                            int prow_e_91 = row_base + 30;
                            if (prow_e_91 < mn_limit) {
                                uint16_t _ue8m0x2_f32_30;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_30) : "f"(zero_f32), "f"(amax[30] * inv_fp8_max));
                                int code_full_30 = (int)_ue8m0x2_f32_30;
                                int code_30 = code_full_30 & 255;
                                int _max_30 = ((254 - code_30) > (0) ? (254 - code_30) : (0));
                                unsigned int inv_bits_30 = (unsigned int)(_max_30 << 23);
                                float inv_scale_30 = __uint_as_float(inv_bits_30) * (float)(code_30 != 0);
                                float q_val_30 = vals[30] * inv_scale_30;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_30));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_91 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_30 = (unsigned int)code_30;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_91 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_30);
                                }
                            }
                            int prow_e_92 = row_base + 31;
                            if (prow_e_92 < mn_limit) {
                                uint16_t _ue8m0x2_f32_31;
                                asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_31) : "f"(zero_f32), "f"(amax[31] * inv_fp8_max));
                                int code_full_31 = (int)_ue8m0x2_f32_31;
                                int code_31 = code_full_31 & 255;
                                int _max_31 = ((254 - code_31) > (0) ? (254 - code_31) : (0));
                                unsigned int inv_bits_31 = (unsigned int)(_max_31 << 23);
                                float inv_scale_31 = __uint_as_float(inv_bits_31) * (float)(code_31 != 0);
                                float q_val_31 = vals[31] * inv_scale_31;
                                {
                                    unsigned short _fp8_pair;
                                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(q_val_31));
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_92 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_31 = (unsigned int)code_31;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_92 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_31);
                                }
                            }
                        }
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(acc_free_addr + (acc_stage) * 8);
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_acc_full ^= 1; }
                nst_epilogue = nst_epilogue + 1;
                mbarrier_wait(tile_full_addr + (tile_stage) * 8, _phase_tile_full);
                info[0] = sinfo[tile_stage * 7];
                info[1] = sinfo[tile_stage * 7 + 1];
                info[2] = sinfo[tile_stage * 7 + 2];
                info[3] = sinfo[tile_stage * 7 + 3];
                info[4] = sinfo[tile_stage * 7 + 4];
                info[5] = sinfo[tile_stage * 7 + 5];
                info[6] = sinfo[tile_stage * 7 + 6];
                meta_alpha = sscale[tile_stage * 33 + 32];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage) * 8);
                tile_stage += 1;
                if (tile_stage == 8) { tile_stage = 0; _phase_tile_full ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int nrec_mma = 0;
            int nst_mma = 0;
            unsigned int sa = 0;
            unsigned int sb = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int tile_stage_1 = 0;
            unsigned int pha = 0;
            unsigned int ab_tok = 1;
            int info_1[7];
            unsigned int _phase_tile_full_1 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_1) * 8, _phase_tile_full_1);
            info_1[0] = sinfo[tile_stage_1 * 7];
            info_1[1] = sinfo[tile_stage_1 * 7 + 1];
            info_1[2] = sinfo[tile_stage_1 * 7 + 2];
            info_1[3] = sinfo[tile_stage_1 * 7 + 3];
            info_1[4] = sinfo[tile_stage_1 * 7 + 4];
            info_1[5] = sinfo[tile_stage_1 * 7 + 5];
            info_1[6] = sinfo[tile_stage_1 * 7 + 6];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_1) * 8);
            tile_stage_1 += 1;
            if (tile_stage_1 == 8) { tile_stage_1 = 0; _phase_tile_full_1 ^= 1; }
            nrec_mma = nrec_mma + 1;
            unsigned int _phase_acc_free = 1;
            unsigned int _phase_b_full = 0;
            #pragma unroll 1
            for (int _tile_1 = 0; _tile_1 < num_m_tiles * group_capacity + 1; _tile_1++) {
                if (info_1[3] == 0) {
                    break;
                }
                uint32_t _mbar_token_0 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                ab_tok = _mbar_token_0;
                mbarrier_wait(acc_free_addr + (acc_stage_1) * 8, _phase_acc_free);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait_token(ab_full_addr + (sa) * 8, pha, ab_tok);
                mbarrier_wait(b_full_addr + (sb) * 8, _phase_b_full);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                {
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 32)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa) * 32)));
                    }
                    int _mma_a_lo_0 = make_warp_uniform((((a_addr) >> 4) & 0x3FFF) + (sa) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((b_addr) >> 4) & 0x3FFF) + (sb) * 256);
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 32)), a_desc + 0, b_desc + 0,
                            0x8880280U, tmem_sf_a, tmem_sf_b, ((((1) ? 1 : 0)) ? 0 : 1));
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 32)), a_desc + 2, b_desc + 2,
                            0x28880290U, tmem_sf_a, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 32)), a_desc + 4, b_desc + 4,
                            0x488802a0U, tmem_sf_a, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 32)), a_desc + 6, b_desc + 6,
                            0x688802b0U, tmem_sf_a, tmem_sf_b, 1);
                    }
                }
                elect_commit(ab_free_addr + (sa) * 8);
                elect_commit(b_free_addr + (sb) * 8);
                sa += 1;
                if (sa == 9) { sa = 0; pha ^= 1; }
                sb += 1;
                if (sb == 9) { sb = 0; _phase_b_full ^= 1; }
                nst_mma = nst_mma + 1;
                ab_tok = 1;
                if (k_tiles > 1) {
                    uint32_t _mbar_token_1 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                    ab_tok = _mbar_token_1;
                }
                #pragma unroll 1
                for (int k = 1; k < k_tiles; k++) {
                    mbarrier_wait_token(ab_full_addr + (sa) * 8, pha, ab_tok);
                    mbarrier_wait(b_full_addr + (sb) * 8, _phase_b_full);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 32)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa) * 32)));
                        }
                        int _mma_a_lo_1 = make_warp_uniform((((a_addr) >> 4) & 0x3FFF) + (sa) * 1024);
                        int _mma_b_lo_1 = make_warp_uniform((((b_addr) >> 4) & 0x3FFF) + (sb) * 256);
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 32)), a_desc + 0, b_desc + 0,
                                0x8880280U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 32)), a_desc + 2, b_desc + 2,
                                0x28880290U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 32)), a_desc + 4, b_desc + 4,
                                0x488802a0U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 32)), a_desc + 6, b_desc + 6,
                                0x688802b0U, tmem_sf_a, tmem_sf_b, 1);
                        }
                    }
                    elect_commit(ab_free_addr + (sa) * 8);
                    elect_commit(b_free_addr + (sb) * 8);
                    sa += 1;
                    if (sa == 9) { sa = 0; pha ^= 1; }
                    sb += 1;
                    if (sb == 9) { sb = 0; _phase_b_full ^= 1; }
                    nst_mma = nst_mma + 1;
                    ab_tok = 1;
                    if (k + 1 < k_tiles) {
                        uint32_t _mbar_token_2 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                        ab_tok = _mbar_token_2;
                    }
                }
                elect_commit(acc_full_addr + (acc_stage_1) * 8);
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_acc_free ^= 1; }
                nst_mma = nst_mma + 1;
                mbarrier_wait(tile_full_addr + (tile_stage_1) * 8, _phase_tile_full_1);
                info_1[0] = sinfo[tile_stage_1 * 7];
                info_1[1] = sinfo[tile_stage_1 * 7 + 1];
                info_1[2] = sinfo[tile_stage_1 * 7 + 2];
                info_1[3] = sinfo[tile_stage_1 * 7 + 3];
                info_1[4] = sinfo[tile_stage_1 * 7 + 4];
                info_1[5] = sinfo[tile_stage_1 * 7 + 5];
                info_1[6] = sinfo[tile_stage_1 * 7 + 6];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_1) * 8);
                tile_stage_1 += 1;
                if (tile_stage_1 == 8) { tile_stage_1 = 0; _phase_tile_full_1 ^= 1; }
                nrec_mma = nrec_mma + 1;
            }
        }
    }
    // ---- Role: tma ----
    if (warp == 5) {
        { // tma_main
            int nrec_tma = 0;
            int nst_tma = 0;
            unsigned int stage = 0;
            unsigned int tile_stage_2 = 0;
            int info_2[7];
            unsigned int _phase_tile_full_2 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_2) * 8, _phase_tile_full_2);
            info_2[0] = sinfo[tile_stage_2 * 7];
            info_2[1] = sinfo[tile_stage_2 * 7 + 1];
            info_2[2] = sinfo[tile_stage_2 * 7 + 2];
            info_2[3] = sinfo[tile_stage_2 * 7 + 3];
            info_2[4] = sinfo[tile_stage_2 * 7 + 4];
            info_2[5] = sinfo[tile_stage_2 * 7 + 5];
            info_2[6] = sinfo[tile_stage_2 * 7 + 6];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_2) * 8);
            tile_stage_2 += 1;
            if (tile_stage_2 == 8) { tile_stage_2 = 0; _phase_tile_full_2 ^= 1; }
            nrec_tma = nrec_tma + 1;
            int batch[1];
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (int _tile_2 = 0; _tile_2 < num_m_tiles * group_capacity + 1; _tile_2++) {
                if (info_2[3] == 0) {
                    break;
                }
                batch[0] = info_2[2] * num_m_tiles + info_2[0];
                int row_base_tma = info_2[1] * 32;
                #pragma unroll 1
                for (int k_1 = 0; k_1 < k_tiles; k_1++) {
                    mbarrier_wait(ab_free_addr + (stage) * 8, _phase_ab_free);
                    if (elect_sync()) {
                        {
                            mbarrier_arrive_expect_tx(ab_full_addr + (stage) * 8, 8704);
                            tma_4d_gmem2smem(a_addr + stage * 16384, A, 0, 0, info_2[5] + k_1, batch[0], ab_full_addr + (stage) * 8);
                            tma_4d_gmem2smem(sfa_addr + stage * 512, SFA, 0, 0, info_2[5] + k_1, batch[0], ab_full_addr + (stage) * 8);
                        }
                    }
                    stage += 1;
                    if (stage == 9) { stage = 0; _phase_ab_free ^= 1; }
                    nst_tma = nst_tma + 1;
                }
                mbarrier_wait(tile_full_addr + (tile_stage_2) * 8, _phase_tile_full_2);
                info_2[0] = sinfo[tile_stage_2 * 7];
                info_2[1] = sinfo[tile_stage_2 * 7 + 1];
                info_2[2] = sinfo[tile_stage_2 * 7 + 2];
                info_2[3] = sinfo[tile_stage_2 * 7 + 3];
                info_2[4] = sinfo[tile_stage_2 * 7 + 4];
                info_2[5] = sinfo[tile_stage_2 * 7 + 5];
                info_2[6] = sinfo[tile_stage_2 * 7 + 6];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_2) * 8);
                tile_stage_2 += 1;
                if (tile_stage_2 == 8) { tile_stage_2 = 0; _phase_tile_full_2 ^= 1; }
                nrec_tma = nrec_tma + 1;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 6) {
        { // scheduler_main
            int nrec_scheduler = 0;
            int nst_scheduler = 0;
            unsigned int tile_stage_3 = 0;
            int num_valid = num_non_exiting_tiles[0];
            int m_chunks = num_m_tiles;
            int total_items = m_chunks * group_capacity;
            int sched_first = bid;
            int sched_step = num_bids;
            unsigned int _phase_tile_free = 1;
            #pragma unroll 1
            for (int item = sched_first; item < total_items; item += sched_step) {
                int row_group = item / m_chunks;
                int m_tile = item - row_group * m_chunks;
                if (row_group >= num_valid) {
                    break;
                }
                mbarrier_wait(tile_free_addr + (tile_stage_3) * 8, _phase_tile_free);
                int sched_row_group = row_group;
                int lookup = row_group;
                int lookup_limit = row_group;
                int expert = tile_idx_to_expert_idx[lookup];
                int mn_limit_1 = tile_idx_to_mn_limit[lookup_limit];
                if (lane == 0) {
                    sscale[tile_stage_3 * 33 + 32] = alpha[expert];
                }
                if (elect_sync()) {
                    sinfo[tile_stage_3 * 7] = m_tile;
                    sinfo[tile_stage_3 * 7 + 1] = sched_row_group;
                    sinfo[tile_stage_3 * 7 + 2] = expert;
                    sinfo[tile_stage_3 * 7 + 3] = 1;
                    sinfo[tile_stage_3 * 7 + 4] = mn_limit_1;
                    sinfo[tile_stage_3 * 7 + 5] = 0;
                    sinfo[tile_stage_3 * 7 + 6] = k_tiles;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 4, 32;" ::: "memory");
                mbarrier_arrive(tile_full_addr + (tile_stage_3) * 8);
                tile_stage_3 += 1;
                if (tile_stage_3 == 8) { tile_stage_3 = 0; _phase_tile_free ^= 1; }
            }
            mbarrier_wait(tile_free_addr + (tile_stage_3) * 8, _phase_tile_free);
            if (elect_sync()) {
                sinfo[tile_stage_3 * 7] = 0;
                sinfo[tile_stage_3 * 7 + 1] = 0;
                sinfo[tile_stage_3 * 7 + 2] = -1;
                sinfo[tile_stage_3 * 7 + 3] = 0;
                sinfo[tile_stage_3 * 7 + 4] = 0;
                sinfo[tile_stage_3 * 7 + 5] = 0;
                sinfo[tile_stage_3 * 7 + 6] = 0;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 4, 32;" ::: "memory");
            mbarrier_arrive(tile_full_addr + (tile_stage_3) * 8);
            tile_stage_3 += 1;
            if (tile_stage_3 == 8) { tile_stage_3 = 0; _phase_tile_free ^= 1; }
        }
    }
    // ---- Role: gather ----
    if (warp == 7) {
        { // gather_main
            int nrec_gather = 0;
            int nst_gather = 0;
            unsigned int stage_1 = 0;
            unsigned int tile_stage_4 = 0;
            int info_3[7];
            unsigned int _phase_tile_full_3 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_4) * 8, _phase_tile_full_3);
            info_3[0] = sinfo[tile_stage_4 * 7];
            info_3[1] = sinfo[tile_stage_4 * 7 + 1];
            info_3[2] = sinfo[tile_stage_4 * 7 + 2];
            info_3[3] = sinfo[tile_stage_4 * 7 + 3];
            info_3[4] = sinfo[tile_stage_4 * 7 + 4];
            info_3[5] = sinfo[tile_stage_4 * 7 + 5];
            info_3[6] = sinfo[tile_stage_4 * 7 + 6];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_4) * 8);
            tile_stage_4 += 1;
            if (tile_stage_4 == 8) { tile_stage_4 = 0; _phase_tile_full_3 ^= 1; }
            nrec_gather = nrec_gather + 1;
            int gather_sub = warp - 7;
            int lane_0_1 = lane;
            int chunk = lane_0_1 % 8;
            int row_in_pass = lane_0_1 / 8;
            int row_src[8];
            int row_ok[8];
            int sf_src[1];
            int sf_ok[1];
            int cta_row0 = 0;
            unsigned int _phase_b_free = 1;
            #pragma unroll 1
            for (int _tile_3 = 0; _tile_3 < num_m_tiles * group_capacity + 1; _tile_3++) {
                if (info_3[3] == 0) {
                    break;
                }
                int row_base_1 = info_3[1] * 32;
                int mn_limit_2 = info_3[4];
                int row = gather_sub * 4 + row_in_pass;
                int prow = row_base_1 + cta_row0 + row;
                int _min_0 = ((prow) < (row_base_1 + 31) ? (prow) : (row_base_1 + 31));
                int safe_row = _min_0;
                int expanded = permuted_idx_to_expanded_idx[safe_row];
                int tok_row = expanded / top_k;
                int ok = (int)(prow < mn_limit_2 && expanded >= 0 && tok_row < num_rows_b);
                row_src[0] = tok_row * ok;
                row_ok[0] = ok;
                int row_0 = (gather_sub + 1) * 4 + row_in_pass;
                int prow_1 = row_base_1 + cta_row0 + row_0;
                int _min_1 = ((prow_1) < (row_base_1 + 31) ? (prow_1) : (row_base_1 + 31));
                int safe_row_2 = _min_1;
                int expanded_3 = permuted_idx_to_expanded_idx[safe_row_2];
                int tok_row_4 = expanded_3 / top_k;
                int ok_5 = (int)(prow_1 < mn_limit_2 && expanded_3 >= 0 && tok_row_4 < num_rows_b);
                row_src[1] = tok_row_4 * ok_5;
                row_ok[1] = ok_5;
                int row_6 = (gather_sub + 2) * 4 + row_in_pass;
                int prow_7 = row_base_1 + cta_row0 + row_6;
                int _min_2 = ((prow_7) < (row_base_1 + 31) ? (prow_7) : (row_base_1 + 31));
                int safe_row_8 = _min_2;
                int expanded_9 = permuted_idx_to_expanded_idx[safe_row_8];
                int tok_row_10 = expanded_9 / top_k;
                int ok_11 = (int)(prow_7 < mn_limit_2 && expanded_9 >= 0 && tok_row_10 < num_rows_b);
                row_src[2] = tok_row_10 * ok_11;
                row_ok[2] = ok_11;
                int row_12 = (gather_sub + 3) * 4 + row_in_pass;
                int prow_13 = row_base_1 + cta_row0 + row_12;
                int _min_3 = ((prow_13) < (row_base_1 + 31) ? (prow_13) : (row_base_1 + 31));
                int safe_row_14 = _min_3;
                int expanded_15 = permuted_idx_to_expanded_idx[safe_row_14];
                int tok_row_16 = expanded_15 / top_k;
                int ok_17 = (int)(prow_13 < mn_limit_2 && expanded_15 >= 0 && tok_row_16 < num_rows_b);
                row_src[3] = tok_row_16 * ok_17;
                row_ok[3] = ok_17;
                int row_18 = (gather_sub + 4) * 4 + row_in_pass;
                int prow_19 = row_base_1 + cta_row0 + row_18;
                int _min_4 = ((prow_19) < (row_base_1 + 31) ? (prow_19) : (row_base_1 + 31));
                int safe_row_20 = _min_4;
                int expanded_21 = permuted_idx_to_expanded_idx[safe_row_20];
                int tok_row_22 = expanded_21 / top_k;
                int ok_23 = (int)(prow_19 < mn_limit_2 && expanded_21 >= 0 && tok_row_22 < num_rows_b);
                row_src[4] = tok_row_22 * ok_23;
                row_ok[4] = ok_23;
                int row_24 = (gather_sub + 5) * 4 + row_in_pass;
                int prow_25 = row_base_1 + cta_row0 + row_24;
                int _min_5 = ((prow_25) < (row_base_1 + 31) ? (prow_25) : (row_base_1 + 31));
                int safe_row_26 = _min_5;
                int expanded_27 = permuted_idx_to_expanded_idx[safe_row_26];
                int tok_row_28 = expanded_27 / top_k;
                int ok_29 = (int)(prow_25 < mn_limit_2 && expanded_27 >= 0 && tok_row_28 < num_rows_b);
                row_src[5] = tok_row_28 * ok_29;
                row_ok[5] = ok_29;
                int row_30 = (gather_sub + 6) * 4 + row_in_pass;
                int prow_31 = row_base_1 + cta_row0 + row_30;
                int _min_6 = ((prow_31) < (row_base_1 + 31) ? (prow_31) : (row_base_1 + 31));
                int safe_row_32 = _min_6;
                int expanded_33 = permuted_idx_to_expanded_idx[safe_row_32];
                int tok_row_34 = expanded_33 / top_k;
                int ok_35 = (int)(prow_31 < mn_limit_2 && expanded_33 >= 0 && tok_row_34 < num_rows_b);
                row_src[6] = tok_row_34 * ok_35;
                row_ok[6] = ok_35;
                int row_36 = (gather_sub + 7) * 4 + row_in_pass;
                int prow_37 = row_base_1 + cta_row0 + row_36;
                int _min_7 = ((prow_37) < (row_base_1 + 31) ? (prow_37) : (row_base_1 + 31));
                int safe_row_38 = _min_7;
                int expanded_39 = permuted_idx_to_expanded_idx[safe_row_38];
                int tok_row_40 = expanded_39 / top_k;
                int ok_41 = (int)(prow_37 < mn_limit_2 && expanded_39 >= 0 && tok_row_40 < num_rows_b);
                row_src[7] = tok_row_40 * ok_41;
                row_ok[7] = ok_41;
                int srow = gather_sub * 32 + lane_0_1;
                int sprow = row_base_1 + srow;
                int _min_8 = ((sprow) < (row_base_1 + 31) ? (sprow) : (row_base_1 + 31));
                int safe_srow = _min_8;
                int sexpanded = permuted_idx_to_expanded_idx[safe_srow];
                int stok_row = sexpanded / top_k;
                int sok = (int)(sprow < mn_limit_2 && srow < 32 && sexpanded >= 0 && stok_row < num_rows_b);
                sf_src[0] = stok_row * sok;
                sf_ok[0] = sok;
                #pragma unroll 1
                for (int k_2 = 0; k_2 < k_tiles; k_2++) {
                    mbarrier_wait(b_free_addr + (stage_1) * 8, _phase_b_free);
                    int k0 = (info_3[5] + k_2) * 128;
                    int dst_off = (gather_sub * 4 + row_in_pass) * 128 + (chunk ^ (gather_sub * 4 + row_in_pass) % 8) * 16;
                    int src_off = row_src[0] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 4096 + (unsigned int)dst_off), "l"(B + src_off), "r"((row_ok[0] != 0) ? 16 : 0));
                    }
                    int dst_off_0 = ((gather_sub + 1) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 1) * 4 + row_in_pass) % 8) * 16;
                    int src_off_1 = row_src[1] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 4096 + (unsigned int)dst_off_0), "l"(B + src_off_1), "r"((row_ok[1] != 0) ? 16 : 0));
                    }
                    int dst_off_2 = ((gather_sub + 2) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 2) * 4 + row_in_pass) % 8) * 16;
                    int src_off_3 = row_src[2] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 4096 + (unsigned int)dst_off_2), "l"(B + src_off_3), "r"((row_ok[2] != 0) ? 16 : 0));
                    }
                    int dst_off_4 = ((gather_sub + 3) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 3) * 4 + row_in_pass) % 8) * 16;
                    int src_off_5 = row_src[3] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 4096 + (unsigned int)dst_off_4), "l"(B + src_off_5), "r"((row_ok[3] != 0) ? 16 : 0));
                    }
                    int dst_off_6 = ((gather_sub + 4) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 4) * 4 + row_in_pass) % 8) * 16;
                    int src_off_7 = row_src[4] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 4096 + (unsigned int)dst_off_6), "l"(B + src_off_7), "r"((row_ok[4] != 0) ? 16 : 0));
                    }
                    int dst_off_8 = ((gather_sub + 5) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 5) * 4 + row_in_pass) % 8) * 16;
                    int src_off_9 = row_src[5] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 4096 + (unsigned int)dst_off_8), "l"(B + src_off_9), "r"((row_ok[5] != 0) ? 16 : 0));
                    }
                    int dst_off_10 = ((gather_sub + 6) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 6) * 4 + row_in_pass) % 8) * 16;
                    int src_off_11 = row_src[6] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 4096 + (unsigned int)dst_off_10), "l"(B + src_off_11), "r"((row_ok[6] != 0) ? 16 : 0));
                    }
                    int dst_off_12 = ((gather_sub + 7) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 7) * 4 + row_in_pass) % 8) * 16;
                    int src_off_13 = row_src[7] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 4096 + (unsigned int)dst_off_12), "l"(B + src_off_13), "r"((row_ok[7] != 0) ? 16 : 0));
                    }
                    int sf_src_off = sf_src[0] * sf_cols + (info_3[5] + k_2) * 4;
                    {
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(sfb_addr + stage_1 * 512 + (unsigned int)(gather_sub / 4 * 512) + (unsigned int)(lane_0_1 * 16) + (unsigned int)(gather_sub % 4 * 4)), "l"(SFB + sf_src_off), "r"((sf_ok[0] != 0) ? 4 : 0));
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(b_full_addr + (stage_1) * 8) : "memory");
                    stage_1 += 1;
                    if (stage_1 == 9) { stage_1 = 0; _phase_b_free ^= 1; }
                    nst_gather = nst_gather + 1;
                }
                mbarrier_wait(tile_full_addr + (tile_stage_4) * 8, _phase_tile_full_3);
                info_3[0] = sinfo[tile_stage_4 * 7];
                info_3[1] = sinfo[tile_stage_4 * 7 + 1];
                info_3[2] = sinfo[tile_stage_4 * 7 + 2];
                info_3[3] = sinfo[tile_stage_4 * 7 + 3];
                info_3[4] = sinfo[tile_stage_4 * 7 + 4];
                info_3[5] = sinfo[tile_stage_4 * 7 + 5];
                info_3[6] = sinfo[tile_stage_4 * 7 + 6];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_4) * 8);
                tile_stage_4 += 1;
                if (tile_stage_4 == 8) { tile_stage_4 = 0; _phase_tile_full_3 ^= 1; }
                nrec_gather = nrec_gather + 1;
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }

    // Kernel epilogue ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
