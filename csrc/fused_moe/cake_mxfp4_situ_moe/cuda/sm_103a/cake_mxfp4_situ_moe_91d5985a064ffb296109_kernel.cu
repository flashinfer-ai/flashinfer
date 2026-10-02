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
#define TMEM_NCOLS 140
#define TMEM_ACC_OFFSET 0
#define TMEM_SF_A_OFFSET 128
#define TMEM_ACC1_OFFSET 64
#define TMEM_SF_A1_OFFSET 132
#define TMEM_SF_B_OFFSET 136
#define NUM_PAB_STAGES 5
#define NUM_PB_STAGES 5
#define NUM_PACC_STAGES 2
#define NUM_PTILE_STAGES 8
#define SMEM_A_OFF 1024
#define SMEM_A_STAGE_BYTES 16384
#define SMEM_A_STRIDE 16384
#define SMEM_B_OFF 164864
#define SMEM_B_STAGE_BYTES 4096
#define SMEM_B_STRIDE 4096
#define SMEM_SFA_OFF 185344
#define SMEM_SFA_STAGE_BYTES 512
#define SMEM_SFA_STRIDE 512
#define SMEM_SFB_OFF 190464
#define SMEM_SFB_STAGE_BYTES 512
#define SMEM_SFB_STRIDE 512
#define SMEM_SINFO_OFF 193024
#define SMEM_SINFO_STAGE_BYTES 224
#define SMEM_SINFO_STRIDE 224
#define SMEM_STOK_OFF 193248
#define SMEM_STOK_STAGE_BYTES 1024
#define SMEM_STOK_STRIDE 1024
#define SMEM_SSCALE_OFF 194272
#define SMEM_SSCALE_STAGE_BYTES 1056
#define SMEM_SSCALE_STRIDE 1056
#define SMEM_SEXCH_OFF 195328
#define SMEM_SEXCH_STAGE_BYTES 8448
#define SMEM_SEXCH_STRIDE 8448
#define SMEM_TOTAL 203776
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
kernel_cake_mxfp4_situ_moe_91d5985a064ffb296109(CakeTensorMap const* A, CakeTensorMap const* SFA, uint8_t* __restrict__ B, uint8_t* __restrict__ SFB, __nv_bfloat16* __restrict__ out, int* __restrict__ tile_idx_to_expert_idx, int* __restrict__ tile_idx_to_mn_limit, int* __restrict__ num_non_exiting_tiles, int* __restrict__ tile_idx_to_row_group, float* __restrict__ alpha, int* __restrict__ permuted_idx_to_expanded_idx, float* __restrict__ token_final_scales, int num_m_tiles, int group_capacity, int k_tiles, int k_cols, int sf_cols, int out_cols, int top_k, float* __restrict__ situ_beta, float* __restrict__ situ_linear_beta, uint8_t* __restrict__ act_sf, float* __restrict__ zero_buf, int zero_words, int num_rows_b, int act_cols, int act_sf_cols, int* __restrict__ dbg)
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
    #define ab_free_addr (mbar_base + 40)
    #define b_full_addr (mbar_base + 80)
    #define b_free_addr (mbar_base + 120)
    #define acc_full_addr (mbar_base + 160)
    #define acc_free_addr (mbar_base + 176)
    #define tile_full_addr (mbar_base + 192)
    #define tile_free_addr (mbar_base + 256)

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
    uint8_t* b = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int b_addr = smem + 164864;
    uint8_t* sfa = reinterpret_cast<uint8_t*>(smem_raw + 185344);
    const int sfa_addr = smem + 185344;
    uint8_t* sfb = reinterpret_cast<uint8_t*>(smem_raw + 190464);
    const int sfb_addr = smem + 190464;
    int* sinfo = reinterpret_cast<int*>(smem_raw + 193024);
    const int sinfo_addr = smem + 193024;
    int* stok = reinterpret_cast<int*>(smem_raw + 193248);
    const int stok_addr = smem + 193248;
    float* sscale = reinterpret_cast<float*>(smem_raw + 194272);
    const int sscale_addr = smem + 194272;
    float* sexch = reinterpret_cast<float*>(smem_raw + 195328);
    const int sexch_addr = smem + 195328;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 40 barriers)
    // Mbarriers at smem_raw[0..320)

    if (warp == 0) {
        // --- pipeline 'pab' ---
        // ab_full: 5 barriers, init_count=1
        // ab_free: 5 barriers, init_count=1
        // --- pipeline 'pb' ---
        // b_full: 5 barriers, init_count=32
        // b_free: 5 barriers, init_count=1
        // --- pipeline 'pacc' ---
        // acc_full: 2 barriers, init_count=1
        // acc_free: 2 barriers, init_count=128
        // --- pipeline 'ptile' ---
        // tile_full: 8 barriers, init_count=32
        // tile_free: 8 barriers, init_count=224
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(24), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(22), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(15), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(10), "r"((uint32_t)(1)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        if (lane < 8) {
            mbarrier_init(smem + 256 + lane * 8, 224);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (256 columns, 140 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 320);
    if (warp == 0) {
        int _tmem_hold = smem + 320;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_sf_a = taddr + 128;
    const int tmem_acc1 = taddr + 64;
    const int tmem_sf_a1 = taddr + 132;
    const int tmem_sf_b = taddr + 136;
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
            int cur_tok[1];
            float cur_scale[1];
            float meta_alpha = 0.0f;
            float vals[32];
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
            nrec_epilogue = nrec_epilogue + 1;
            int is_even_lane = (int)(lane_0 % 2 == 0);
            int is_gate_lane = (int)(epi_warp >= 2);
            int exch_row = epi_tidx & 63;
            float inv_fp8_max = 0.002232142857142857f;
            float zero_f32 = 0.0f;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int _tile = 0; _tile < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile++) {
                if (info[3] == 0) {
                    break;
                }
                int row_base = info[1] * 32;
                int mn_limit = info[4];
                mbarrier_wait(acc_full_addr + (acc_stage) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int m_tile_e = sinfo[tile_stage * 7] * 2;
                int h = m_tile_e * 128 + epi_tidx;
                if (m_tile_e < num_m_tiles) {
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
                        float v = vals[0] * sscale[tile_stage * 33];
                        int tok = stok[tile_stage * 32];
                        int col_ok = (int)(mn_limit > row_base);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok * out_cols + h])), "f"(v), "r"((unsigned int)(col_ok)) : "memory");
                        float v_0 = vals[1] * sscale[tile_stage * 33 + 1];
                        int tok_1 = stok[tile_stage * 32 + 1];
                        int col_ok_2 = (int)(mn_limit > row_base + 1);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_1 * out_cols + h])), "f"(v_0), "r"((unsigned int)(col_ok_2)) : "memory");
                        float v_3 = vals[2] * sscale[tile_stage * 33 + 2];
                        int tok_4 = stok[tile_stage * 32 + 2];
                        int col_ok_5 = (int)(mn_limit > row_base + 2);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_4 * out_cols + h])), "f"(v_3), "r"((unsigned int)(col_ok_5)) : "memory");
                        float v_6 = vals[3] * sscale[tile_stage * 33 + 3];
                        int tok_7 = stok[tile_stage * 32 + 3];
                        int col_ok_8 = (int)(mn_limit > row_base + 3);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_7 * out_cols + h])), "f"(v_6), "r"((unsigned int)(col_ok_8)) : "memory");
                        float v_9 = vals[4] * sscale[tile_stage * 33 + 4];
                        int tok_10 = stok[tile_stage * 32 + 4];
                        int col_ok_11 = (int)(mn_limit > row_base + 4);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_10 * out_cols + h])), "f"(v_9), "r"((unsigned int)(col_ok_11)) : "memory");
                        float v_12 = vals[5] * sscale[tile_stage * 33 + 5];
                        int tok_13 = stok[tile_stage * 32 + 5];
                        int col_ok_14 = (int)(mn_limit > row_base + 5);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_13 * out_cols + h])), "f"(v_12), "r"((unsigned int)(col_ok_14)) : "memory");
                        float v_15 = vals[6] * sscale[tile_stage * 33 + 6];
                        int tok_16 = stok[tile_stage * 32 + 6];
                        int col_ok_17 = (int)(mn_limit > row_base + 6);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_16 * out_cols + h])), "f"(v_15), "r"((unsigned int)(col_ok_17)) : "memory");
                        float v_18 = vals[7] * sscale[tile_stage * 33 + 7];
                        int tok_19 = stok[tile_stage * 32 + 7];
                        int col_ok_20 = (int)(mn_limit > row_base + 7);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_19 * out_cols + h])), "f"(v_18), "r"((unsigned int)(col_ok_20)) : "memory");
                        float v_21 = vals[8] * sscale[tile_stage * 33 + 8];
                        int tok_22 = stok[tile_stage * 32 + 8];
                        int col_ok_23 = (int)(mn_limit > row_base + 8);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_22 * out_cols + h])), "f"(v_21), "r"((unsigned int)(col_ok_23)) : "memory");
                        float v_24 = vals[9] * sscale[tile_stage * 33 + 9];
                        int tok_25 = stok[tile_stage * 32 + 9];
                        int col_ok_26 = (int)(mn_limit > row_base + 9);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_25 * out_cols + h])), "f"(v_24), "r"((unsigned int)(col_ok_26)) : "memory");
                        float v_27 = vals[10] * sscale[tile_stage * 33 + 10];
                        int tok_28 = stok[tile_stage * 32 + 10];
                        int col_ok_29 = (int)(mn_limit > row_base + 10);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_28 * out_cols + h])), "f"(v_27), "r"((unsigned int)(col_ok_29)) : "memory");
                        float v_30 = vals[11] * sscale[tile_stage * 33 + 11];
                        int tok_31 = stok[tile_stage * 32 + 11];
                        int col_ok_32 = (int)(mn_limit > row_base + 11);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_31 * out_cols + h])), "f"(v_30), "r"((unsigned int)(col_ok_32)) : "memory");
                        float v_33 = vals[12] * sscale[tile_stage * 33 + 12];
                        int tok_34 = stok[tile_stage * 32 + 12];
                        int col_ok_35 = (int)(mn_limit > row_base + 12);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_34 * out_cols + h])), "f"(v_33), "r"((unsigned int)(col_ok_35)) : "memory");
                        float v_36 = vals[13] * sscale[tile_stage * 33 + 13];
                        int tok_37 = stok[tile_stage * 32 + 13];
                        int col_ok_38 = (int)(mn_limit > row_base + 13);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_37 * out_cols + h])), "f"(v_36), "r"((unsigned int)(col_ok_38)) : "memory");
                        float v_39 = vals[14] * sscale[tile_stage * 33 + 14];
                        int tok_40 = stok[tile_stage * 32 + 14];
                        int col_ok_41 = (int)(mn_limit > row_base + 14);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_40 * out_cols + h])), "f"(v_39), "r"((unsigned int)(col_ok_41)) : "memory");
                        float v_42 = vals[15] * sscale[tile_stage * 33 + 15];
                        int tok_43 = stok[tile_stage * 32 + 15];
                        int col_ok_44 = (int)(mn_limit > row_base + 15);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_43 * out_cols + h])), "f"(v_42), "r"((unsigned int)(col_ok_44)) : "memory");
                        float v_45 = vals[16] * sscale[tile_stage * 33 + 16];
                        int tok_46 = stok[tile_stage * 32 + 16];
                        int col_ok_47 = (int)(mn_limit > row_base + 16);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_46 * out_cols + h])), "f"(v_45), "r"((unsigned int)(col_ok_47)) : "memory");
                        float v_48 = vals[17] * sscale[tile_stage * 33 + 17];
                        int tok_49 = stok[tile_stage * 32 + 17];
                        int col_ok_50 = (int)(mn_limit > row_base + 17);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_49 * out_cols + h])), "f"(v_48), "r"((unsigned int)(col_ok_50)) : "memory");
                        float v_51 = vals[18] * sscale[tile_stage * 33 + 18];
                        int tok_52 = stok[tile_stage * 32 + 18];
                        int col_ok_53 = (int)(mn_limit > row_base + 18);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_52 * out_cols + h])), "f"(v_51), "r"((unsigned int)(col_ok_53)) : "memory");
                        float v_54 = vals[19] * sscale[tile_stage * 33 + 19];
                        int tok_55 = stok[tile_stage * 32 + 19];
                        int col_ok_56 = (int)(mn_limit > row_base + 19);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_55 * out_cols + h])), "f"(v_54), "r"((unsigned int)(col_ok_56)) : "memory");
                        float v_57 = vals[20] * sscale[tile_stage * 33 + 20];
                        int tok_58 = stok[tile_stage * 32 + 20];
                        int col_ok_59 = (int)(mn_limit > row_base + 20);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_58 * out_cols + h])), "f"(v_57), "r"((unsigned int)(col_ok_59)) : "memory");
                        float v_60 = vals[21] * sscale[tile_stage * 33 + 21];
                        int tok_61 = stok[tile_stage * 32 + 21];
                        int col_ok_62 = (int)(mn_limit > row_base + 21);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_61 * out_cols + h])), "f"(v_60), "r"((unsigned int)(col_ok_62)) : "memory");
                        float v_63 = vals[22] * sscale[tile_stage * 33 + 22];
                        int tok_64 = stok[tile_stage * 32 + 22];
                        int col_ok_65 = (int)(mn_limit > row_base + 22);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_64 * out_cols + h])), "f"(v_63), "r"((unsigned int)(col_ok_65)) : "memory");
                        float v_66 = vals[23] * sscale[tile_stage * 33 + 23];
                        int tok_67 = stok[tile_stage * 32 + 23];
                        int col_ok_68 = (int)(mn_limit > row_base + 23);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_67 * out_cols + h])), "f"(v_66), "r"((unsigned int)(col_ok_68)) : "memory");
                        float v_69 = vals[24] * sscale[tile_stage * 33 + 24];
                        int tok_70 = stok[tile_stage * 32 + 24];
                        int col_ok_71 = (int)(mn_limit > row_base + 24);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_70 * out_cols + h])), "f"(v_69), "r"((unsigned int)(col_ok_71)) : "memory");
                        float v_72 = vals[25] * sscale[tile_stage * 33 + 25];
                        int tok_73 = stok[tile_stage * 32 + 25];
                        int col_ok_74 = (int)(mn_limit > row_base + 25);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_73 * out_cols + h])), "f"(v_72), "r"((unsigned int)(col_ok_74)) : "memory");
                        float v_75 = vals[26] * sscale[tile_stage * 33 + 26];
                        int tok_76 = stok[tile_stage * 32 + 26];
                        int col_ok_77 = (int)(mn_limit > row_base + 26);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_76 * out_cols + h])), "f"(v_75), "r"((unsigned int)(col_ok_77)) : "memory");
                        float v_78 = vals[27] * sscale[tile_stage * 33 + 27];
                        int tok_79 = stok[tile_stage * 32 + 27];
                        int col_ok_80 = (int)(mn_limit > row_base + 27);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_79 * out_cols + h])), "f"(v_78), "r"((unsigned int)(col_ok_80)) : "memory");
                        float v_81 = vals[28] * sscale[tile_stage * 33 + 28];
                        int tok_82 = stok[tile_stage * 32 + 28];
                        int col_ok_83 = (int)(mn_limit > row_base + 28);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_82 * out_cols + h])), "f"(v_81), "r"((unsigned int)(col_ok_83)) : "memory");
                        float v_84 = vals[29] * sscale[tile_stage * 33 + 29];
                        int tok_85 = stok[tile_stage * 32 + 29];
                        int col_ok_86 = (int)(mn_limit > row_base + 29);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_85 * out_cols + h])), "f"(v_84), "r"((unsigned int)(col_ok_86)) : "memory");
                        float v_87 = vals[30] * sscale[tile_stage * 33 + 30];
                        int tok_88 = stok[tile_stage * 32 + 30];
                        int col_ok_89 = (int)(mn_limit > row_base + 30);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_88 * out_cols + h])), "f"(v_87), "r"((unsigned int)(col_ok_89)) : "memory");
                        float v_90 = vals[31] * sscale[tile_stage * 33 + 31];
                        int tok_91 = stok[tile_stage * 32 + 31];
                        int col_ok_92 = (int)(mn_limit > row_base + 31);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_91 * out_cols + h])), "f"(v_90), "r"((unsigned int)(col_ok_92)) : "memory");
                    }
                }
                int m_tile_e_0 = sinfo[tile_stage * 7] * 2 + 1;
                int h_1 = m_tile_e_0 * 128 + epi_tidx;
                if (m_tile_e_0 < num_m_tiles) {
                    float _tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(taddr + 64 + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    vals[0] = _tmem_load_1[0] * meta_alpha;
                    vals[1] = _tmem_load_1[1] * meta_alpha;
                    vals[2] = _tmem_load_1[2] * meta_alpha;
                    vals[3] = _tmem_load_1[3] * meta_alpha;
                    vals[4] = _tmem_load_1[4] * meta_alpha;
                    vals[5] = _tmem_load_1[5] * meta_alpha;
                    vals[6] = _tmem_load_1[6] * meta_alpha;
                    vals[7] = _tmem_load_1[7] * meta_alpha;
                    vals[8] = _tmem_load_1[8] * meta_alpha;
                    vals[9] = _tmem_load_1[9] * meta_alpha;
                    vals[10] = _tmem_load_1[10] * meta_alpha;
                    vals[11] = _tmem_load_1[11] * meta_alpha;
                    vals[12] = _tmem_load_1[12] * meta_alpha;
                    vals[13] = _tmem_load_1[13] * meta_alpha;
                    vals[14] = _tmem_load_1[14] * meta_alpha;
                    vals[15] = _tmem_load_1[15] * meta_alpha;
                    vals[16] = _tmem_load_1[16] * meta_alpha;
                    vals[17] = _tmem_load_1[17] * meta_alpha;
                    vals[18] = _tmem_load_1[18] * meta_alpha;
                    vals[19] = _tmem_load_1[19] * meta_alpha;
                    vals[20] = _tmem_load_1[20] * meta_alpha;
                    vals[21] = _tmem_load_1[21] * meta_alpha;
                    vals[22] = _tmem_load_1[22] * meta_alpha;
                    vals[23] = _tmem_load_1[23] * meta_alpha;
                    vals[24] = _tmem_load_1[24] * meta_alpha;
                    vals[25] = _tmem_load_1[25] * meta_alpha;
                    vals[26] = _tmem_load_1[26] * meta_alpha;
                    vals[27] = _tmem_load_1[27] * meta_alpha;
                    vals[28] = _tmem_load_1[28] * meta_alpha;
                    vals[29] = _tmem_load_1[29] * meta_alpha;
                    vals[30] = _tmem_load_1[30] * meta_alpha;
                    vals[31] = _tmem_load_1[31] * meta_alpha;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    {
                        float v_1 = vals[0] * sscale[tile_stage * 33];
                        int tok_2 = stok[tile_stage * 32];
                        int col_ok_1 = (int)(mn_limit > row_base);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_2 * out_cols + h_1])), "f"(v_1), "r"((unsigned int)(col_ok_1)) : "memory");
                        float v_0_1 = vals[1] * sscale[tile_stage * 33 + 1];
                        int tok_1_1 = stok[tile_stage * 32 + 1];
                        int col_ok_2_1 = (int)(mn_limit > row_base + 1);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_1_1 * out_cols + h_1])), "f"(v_0_1), "r"((unsigned int)(col_ok_2_1)) : "memory");
                        float v_3_1 = vals[2] * sscale[tile_stage * 33 + 2];
                        int tok_4_1 = stok[tile_stage * 32 + 2];
                        int col_ok_5_1 = (int)(mn_limit > row_base + 2);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_4_1 * out_cols + h_1])), "f"(v_3_1), "r"((unsigned int)(col_ok_5_1)) : "memory");
                        float v_6_1 = vals[3] * sscale[tile_stage * 33 + 3];
                        int tok_7_1 = stok[tile_stage * 32 + 3];
                        int col_ok_8_1 = (int)(mn_limit > row_base + 3);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_7_1 * out_cols + h_1])), "f"(v_6_1), "r"((unsigned int)(col_ok_8_1)) : "memory");
                        float v_9_1 = vals[4] * sscale[tile_stage * 33 + 4];
                        int tok_10_1 = stok[tile_stage * 32 + 4];
                        int col_ok_11_1 = (int)(mn_limit > row_base + 4);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_10_1 * out_cols + h_1])), "f"(v_9_1), "r"((unsigned int)(col_ok_11_1)) : "memory");
                        float v_12_1 = vals[5] * sscale[tile_stage * 33 + 5];
                        int tok_13_1 = stok[tile_stage * 32 + 5];
                        int col_ok_14_1 = (int)(mn_limit > row_base + 5);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_13_1 * out_cols + h_1])), "f"(v_12_1), "r"((unsigned int)(col_ok_14_1)) : "memory");
                        float v_15_1 = vals[6] * sscale[tile_stage * 33 + 6];
                        int tok_16_1 = stok[tile_stage * 32 + 6];
                        int col_ok_17_1 = (int)(mn_limit > row_base + 6);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_16_1 * out_cols + h_1])), "f"(v_15_1), "r"((unsigned int)(col_ok_17_1)) : "memory");
                        float v_18_1 = vals[7] * sscale[tile_stage * 33 + 7];
                        int tok_19_1 = stok[tile_stage * 32 + 7];
                        int col_ok_20_1 = (int)(mn_limit > row_base + 7);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_19_1 * out_cols + h_1])), "f"(v_18_1), "r"((unsigned int)(col_ok_20_1)) : "memory");
                        float v_21_1 = vals[8] * sscale[tile_stage * 33 + 8];
                        int tok_22_1 = stok[tile_stage * 32 + 8];
                        int col_ok_23_1 = (int)(mn_limit > row_base + 8);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_22_1 * out_cols + h_1])), "f"(v_21_1), "r"((unsigned int)(col_ok_23_1)) : "memory");
                        float v_24_1 = vals[9] * sscale[tile_stage * 33 + 9];
                        int tok_25_1 = stok[tile_stage * 32 + 9];
                        int col_ok_26_1 = (int)(mn_limit > row_base + 9);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_25_1 * out_cols + h_1])), "f"(v_24_1), "r"((unsigned int)(col_ok_26_1)) : "memory");
                        float v_27_1 = vals[10] * sscale[tile_stage * 33 + 10];
                        int tok_28_1 = stok[tile_stage * 32 + 10];
                        int col_ok_29_1 = (int)(mn_limit > row_base + 10);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_28_1 * out_cols + h_1])), "f"(v_27_1), "r"((unsigned int)(col_ok_29_1)) : "memory");
                        float v_30_1 = vals[11] * sscale[tile_stage * 33 + 11];
                        int tok_31_1 = stok[tile_stage * 32 + 11];
                        int col_ok_32_1 = (int)(mn_limit > row_base + 11);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_31_1 * out_cols + h_1])), "f"(v_30_1), "r"((unsigned int)(col_ok_32_1)) : "memory");
                        float v_33_1 = vals[12] * sscale[tile_stage * 33 + 12];
                        int tok_34_1 = stok[tile_stage * 32 + 12];
                        int col_ok_35_1 = (int)(mn_limit > row_base + 12);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_34_1 * out_cols + h_1])), "f"(v_33_1), "r"((unsigned int)(col_ok_35_1)) : "memory");
                        float v_36_1 = vals[13] * sscale[tile_stage * 33 + 13];
                        int tok_37_1 = stok[tile_stage * 32 + 13];
                        int col_ok_38_1 = (int)(mn_limit > row_base + 13);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_37_1 * out_cols + h_1])), "f"(v_36_1), "r"((unsigned int)(col_ok_38_1)) : "memory");
                        float v_39_1 = vals[14] * sscale[tile_stage * 33 + 14];
                        int tok_40_1 = stok[tile_stage * 32 + 14];
                        int col_ok_41_1 = (int)(mn_limit > row_base + 14);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_40_1 * out_cols + h_1])), "f"(v_39_1), "r"((unsigned int)(col_ok_41_1)) : "memory");
                        float v_42_1 = vals[15] * sscale[tile_stage * 33 + 15];
                        int tok_43_1 = stok[tile_stage * 32 + 15];
                        int col_ok_44_1 = (int)(mn_limit > row_base + 15);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_43_1 * out_cols + h_1])), "f"(v_42_1), "r"((unsigned int)(col_ok_44_1)) : "memory");
                        float v_45_1 = vals[16] * sscale[tile_stage * 33 + 16];
                        int tok_46_1 = stok[tile_stage * 32 + 16];
                        int col_ok_47_1 = (int)(mn_limit > row_base + 16);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_46_1 * out_cols + h_1])), "f"(v_45_1), "r"((unsigned int)(col_ok_47_1)) : "memory");
                        float v_48_1 = vals[17] * sscale[tile_stage * 33 + 17];
                        int tok_49_1 = stok[tile_stage * 32 + 17];
                        int col_ok_50_1 = (int)(mn_limit > row_base + 17);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_49_1 * out_cols + h_1])), "f"(v_48_1), "r"((unsigned int)(col_ok_50_1)) : "memory");
                        float v_51_1 = vals[18] * sscale[tile_stage * 33 + 18];
                        int tok_52_1 = stok[tile_stage * 32 + 18];
                        int col_ok_53_1 = (int)(mn_limit > row_base + 18);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_52_1 * out_cols + h_1])), "f"(v_51_1), "r"((unsigned int)(col_ok_53_1)) : "memory");
                        float v_54_1 = vals[19] * sscale[tile_stage * 33 + 19];
                        int tok_55_1 = stok[tile_stage * 32 + 19];
                        int col_ok_56_1 = (int)(mn_limit > row_base + 19);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_55_1 * out_cols + h_1])), "f"(v_54_1), "r"((unsigned int)(col_ok_56_1)) : "memory");
                        float v_57_1 = vals[20] * sscale[tile_stage * 33 + 20];
                        int tok_58_1 = stok[tile_stage * 32 + 20];
                        int col_ok_59_1 = (int)(mn_limit > row_base + 20);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_58_1 * out_cols + h_1])), "f"(v_57_1), "r"((unsigned int)(col_ok_59_1)) : "memory");
                        float v_60_1 = vals[21] * sscale[tile_stage * 33 + 21];
                        int tok_61_1 = stok[tile_stage * 32 + 21];
                        int col_ok_62_1 = (int)(mn_limit > row_base + 21);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_61_1 * out_cols + h_1])), "f"(v_60_1), "r"((unsigned int)(col_ok_62_1)) : "memory");
                        float v_63_1 = vals[22] * sscale[tile_stage * 33 + 22];
                        int tok_64_1 = stok[tile_stage * 32 + 22];
                        int col_ok_65_1 = (int)(mn_limit > row_base + 22);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_64_1 * out_cols + h_1])), "f"(v_63_1), "r"((unsigned int)(col_ok_65_1)) : "memory");
                        float v_66_1 = vals[23] * sscale[tile_stage * 33 + 23];
                        int tok_67_1 = stok[tile_stage * 32 + 23];
                        int col_ok_68_1 = (int)(mn_limit > row_base + 23);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_67_1 * out_cols + h_1])), "f"(v_66_1), "r"((unsigned int)(col_ok_68_1)) : "memory");
                        float v_69_1 = vals[24] * sscale[tile_stage * 33 + 24];
                        int tok_70_1 = stok[tile_stage * 32 + 24];
                        int col_ok_71_1 = (int)(mn_limit > row_base + 24);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_70_1 * out_cols + h_1])), "f"(v_69_1), "r"((unsigned int)(col_ok_71_1)) : "memory");
                        float v_72_1 = vals[25] * sscale[tile_stage * 33 + 25];
                        int tok_73_1 = stok[tile_stage * 32 + 25];
                        int col_ok_74_1 = (int)(mn_limit > row_base + 25);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_73_1 * out_cols + h_1])), "f"(v_72_1), "r"((unsigned int)(col_ok_74_1)) : "memory");
                        float v_75_1 = vals[26] * sscale[tile_stage * 33 + 26];
                        int tok_76_1 = stok[tile_stage * 32 + 26];
                        int col_ok_77_1 = (int)(mn_limit > row_base + 26);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_76_1 * out_cols + h_1])), "f"(v_75_1), "r"((unsigned int)(col_ok_77_1)) : "memory");
                        float v_78_1 = vals[27] * sscale[tile_stage * 33 + 27];
                        int tok_79_1 = stok[tile_stage * 32 + 27];
                        int col_ok_80_1 = (int)(mn_limit > row_base + 27);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_79_1 * out_cols + h_1])), "f"(v_78_1), "r"((unsigned int)(col_ok_80_1)) : "memory");
                        float v_81_1 = vals[28] * sscale[tile_stage * 33 + 28];
                        int tok_82_1 = stok[tile_stage * 32 + 28];
                        int col_ok_83_1 = (int)(mn_limit > row_base + 28);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_82_1 * out_cols + h_1])), "f"(v_81_1), "r"((unsigned int)(col_ok_83_1)) : "memory");
                        float v_84_1 = vals[29] * sscale[tile_stage * 33 + 29];
                        int tok_85_1 = stok[tile_stage * 32 + 29];
                        int col_ok_86_1 = (int)(mn_limit > row_base + 29);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_85_1 * out_cols + h_1])), "f"(v_84_1), "r"((unsigned int)(col_ok_86_1)) : "memory");
                        float v_87_1 = vals[30] * sscale[tile_stage * 33 + 30];
                        int tok_88_1 = stok[tile_stage * 32 + 30];
                        int col_ok_89_1 = (int)(mn_limit > row_base + 30);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_88_1 * out_cols + h_1])), "f"(v_87_1), "r"((unsigned int)(col_ok_89_1)) : "memory");
                        float v_90_1 = vals[31] * sscale[tile_stage * 33 + 31];
                        int tok_91_1 = stok[tile_stage * 32 + 31];
                        int col_ok_92_1 = (int)(mn_limit > row_base + 31);
                        asm volatile("{ .reg .pred p_; .reg .b16 h_; setp.ne.b32 p_, %2, 0; cvt.rn.bf16.f32 h_, %1; @p_ st.global.b16 [%0], h_; }" :: "l"(reinterpret_cast<uint64_t>(&out[tok_91_1 * out_cols + h_1])), "f"(v_90_1), "r"((unsigned int)(col_ok_92_1)) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(acc_free_addr + (acc_stage) * 8);
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_acc_full ^= 1; }
                nst_epilogue = nst_epilogue + 1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage) * 8);
                tile_stage += 1;
                if (tile_stage == 8) { tile_stage = 0; _phase_tile_full ^= 1; }
                mbarrier_wait(tile_full_addr + (tile_stage) * 8, _phase_tile_full);
                info[0] = sinfo[tile_stage * 7];
                info[1] = sinfo[tile_stage * 7 + 1];
                info[2] = sinfo[tile_stage * 7 + 2];
                info[3] = sinfo[tile_stage * 7 + 3];
                info[4] = sinfo[tile_stage * 7 + 4];
                info[5] = sinfo[tile_stage * 7 + 5];
                info[6] = sinfo[tile_stage * 7 + 6];
                meta_alpha = sscale[tile_stage * 33 + 32];
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
            for (int _tile_1 = 0; _tile_1 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_1++) {
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
                        tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa * 2) * 32)));
                    }
                    int _mma_a_lo_0 = make_warp_uniform((((a_addr) >> 4) & 0x3FFF) + (sa * 2) * 1024);
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
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_a1, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa * 2 + 1) * 32)));
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((a_addr) >> 4) & 0x3FFF) + (sa * 2 + 1) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform((((b_addr) >> 4) & 0x3FFF) + (sb) * 256);
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 32)), a_desc + 0, b_desc + 0,
                            0x8880280U, tmem_sf_a1, tmem_sf_b, ((((1) ? 1 : 0)) ? 0 : 1));
                        tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 32)), a_desc + 2, b_desc + 2,
                            0x28880290U, tmem_sf_a1, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 32)), a_desc + 4, b_desc + 4,
                            0x488802a0U, tmem_sf_a1, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 32)), a_desc + 6, b_desc + 6,
                            0x688802b0U, tmem_sf_a1, tmem_sf_b, 1);
                    }
                }
                elect_commit(ab_free_addr + (sa) * 8);
                elect_commit(b_free_addr + (sb) * 8);
                sa += 1;
                if (sa == 5) { sa = 0; pha ^= 1; }
                sb += 1;
                if (sb == 5) { sb = 0; _phase_b_full ^= 1; }
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
                            tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa * 2) * 32)));
                        }
                        int _mma_a_lo_2 = make_warp_uniform((((a_addr) >> 4) & 0x3FFF) + (sa * 2) * 1024);
                        int _mma_b_lo_2 = make_warp_uniform((((b_addr) >> 4) & 0x3FFF) + (sb) * 256);
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 32)), a_desc + 0, b_desc + 0,
                                0x8880280U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 32)), a_desc + 2, b_desc + 2,
                                0x28880290U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 32)), a_desc + 4, b_desc + 4,
                                0x488802a0U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 32)), a_desc + 6, b_desc + 6,
                                0x688802b0U, tmem_sf_a, tmem_sf_b, 1);
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_sf_a1, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa * 2 + 1) * 32)));
                        }
                        int _mma_a_lo_3 = make_warp_uniform((((a_addr) >> 4) & 0x3FFF) + (sa * 2 + 1) * 1024);
                        int _mma_b_lo_3 = make_warp_uniform((((b_addr) >> 4) & 0x3FFF) + (sb) * 256);
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 32)), a_desc + 0, b_desc + 0,
                                0x8880280U, tmem_sf_a1, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 32)), a_desc + 2, b_desc + 2,
                                0x28880290U, tmem_sf_a1, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 32)), a_desc + 4, b_desc + 4,
                                0x488802a0U, tmem_sf_a1, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc1 + (acc_stage_1 * 32)), a_desc + 6, b_desc + 6,
                                0x688802b0U, tmem_sf_a1, tmem_sf_b, 1);
                        }
                    }
                    elect_commit(ab_free_addr + (sa) * 8);
                    elect_commit(b_free_addr + (sb) * 8);
                    sa += 1;
                    if (sa == 5) { sa = 0; pha ^= 1; }
                    sb += 1;
                    if (sb == 5) { sb = 0; _phase_b_full ^= 1; }
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
            int batch[2];
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (int _tile_2 = 0; _tile_2 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_2++) {
                if (info_2[3] == 0) {
                    break;
                }
                int _min_0 = ((info_2[0] * 2) < (num_m_tiles - 1) ? (info_2[0] * 2) : (num_m_tiles - 1));
                batch[0] = info_2[2] * num_m_tiles + _min_0;
                int _min_1 = ((info_2[0] * 2 + 1) < (num_m_tiles - 1) ? (info_2[0] * 2 + 1) : (num_m_tiles - 1));
                batch[1] = info_2[2] * num_m_tiles + _min_1;
                int row_base_tma = info_2[1] * 32;
                #pragma unroll 1
                for (int k_1 = 0; k_1 < k_tiles; k_1++) {
                    mbarrier_wait(ab_free_addr + (stage) * 8, _phase_ab_free);
                    if (elect_sync()) {
                        {
                            mbarrier_arrive_expect_tx(ab_full_addr + (stage) * 8, 17408);
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(a_addr + stage * 2 * 16384), "l"(A), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[0]),
                                   "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(sfa_addr + stage * 2 * 512), "l"(SFA), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[0]),
                                   "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(a_addr + (stage * 2 + 1) * 16384), "l"(A), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[1]),
                                   "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(sfa_addr + (stage * 2 + 1) * 512), "l"(SFA), "r"(0), "r"(0), "r"(info_2[5] + k_1), "r"(batch[1]),
                                   "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 5) { stage = 0; _phase_ab_free ^= 1; }
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
            int m_chunks = (num_m_tiles + 1) / 2;
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
                int meta_col = lane;
                if (meta_col < 32) {
                    int meta_prow = sched_row_group * 32 + meta_col;
                    sscale[tile_stage_3 * 33 + (unsigned int)meta_col] = 1.0f;
                    stok[tile_stage_3 * 32 + (unsigned int)meta_col] = meta_prow;
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
            for (int _tile_3 = 0; _tile_3 < (num_m_tiles + 1) / 2 * group_capacity + 1; _tile_3++) {
                if (info_3[3] == 0) {
                    break;
                }
                int row_base_1 = info_3[1] * 32;
                int mn_limit_2 = info_3[4];
                int row = gather_sub * 4 + row_in_pass;
                int prow = row_base_1 + cta_row0 + row;
                int ok = (int)(prow < mn_limit_2);
                row_src[0] = prow * ok;
                row_ok[0] = ok;
                int row_0 = (gather_sub + 1) * 4 + row_in_pass;
                int prow_1 = row_base_1 + cta_row0 + row_0;
                int ok_2 = (int)(prow_1 < mn_limit_2);
                row_src[1] = prow_1 * ok_2;
                row_ok[1] = ok_2;
                int row_3 = (gather_sub + 2) * 4 + row_in_pass;
                int prow_4 = row_base_1 + cta_row0 + row_3;
                int ok_5 = (int)(prow_4 < mn_limit_2);
                row_src[2] = prow_4 * ok_5;
                row_ok[2] = ok_5;
                int row_6 = (gather_sub + 3) * 4 + row_in_pass;
                int prow_7 = row_base_1 + cta_row0 + row_6;
                int ok_8 = (int)(prow_7 < mn_limit_2);
                row_src[3] = prow_7 * ok_8;
                row_ok[3] = ok_8;
                int row_9 = (gather_sub + 4) * 4 + row_in_pass;
                int prow_10 = row_base_1 + cta_row0 + row_9;
                int ok_11 = (int)(prow_10 < mn_limit_2);
                row_src[4] = prow_10 * ok_11;
                row_ok[4] = ok_11;
                int row_12 = (gather_sub + 5) * 4 + row_in_pass;
                int prow_13 = row_base_1 + cta_row0 + row_12;
                int ok_14 = (int)(prow_13 < mn_limit_2);
                row_src[5] = prow_13 * ok_14;
                row_ok[5] = ok_14;
                int row_15 = (gather_sub + 6) * 4 + row_in_pass;
                int prow_16 = row_base_1 + cta_row0 + row_15;
                int ok_17 = (int)(prow_16 < mn_limit_2);
                row_src[6] = prow_16 * ok_17;
                row_ok[6] = ok_17;
                int row_18 = (gather_sub + 7) * 4 + row_in_pass;
                int prow_19 = row_base_1 + cta_row0 + row_18;
                int ok_20 = (int)(prow_19 < mn_limit_2);
                row_src[7] = prow_19 * ok_20;
                row_ok[7] = ok_20;
                int srow = gather_sub * 32 + lane_0_1;
                int sprow = row_base_1 + srow;
                int sok = (int)(sprow < mn_limit_2 && srow < 32);
                sf_src[0] = sprow * sok;
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
                    if (stage_1 == 5) { stage_1 = 0; _phase_b_free ^= 1; }
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
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }

    // Kernel epilogue ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
