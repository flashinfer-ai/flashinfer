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
#define TMEM_NCOLS 32
#define TMEM_ACC_OFFSET 0
#define TMEM_SF_A_OFFSET 16
#define TMEM_SF_B_OFFSET 24
#define NUM_PAB_STAGES 5
#define NUM_PB_STAGES 5
#define NUM_PACC_STAGES 2
#define NUM_PTILE_STAGES 8
#define SMEM_A_OFF 1024
#define SMEM_A_STAGE_BYTES 32768
#define SMEM_A_STRIDE 32768
#define SMEM_B_OFF 164864
#define SMEM_B_STAGE_BYTES 2048
#define SMEM_B_STRIDE 2048
#define SMEM_SFA_OFF 175104
#define SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SFA_STRIDE 1024
#define SMEM_SFB_OFF 180224
#define SMEM_SFB_STAGE_BYTES 1024
#define SMEM_SFB_STRIDE 1024
#define SMEM_SINFO_OFF 185344
#define SMEM_SINFO_STAGE_BYTES 224
#define SMEM_SINFO_STRIDE 224
#define SMEM_STOK_OFF 185568
#define SMEM_STOK_STAGE_BYTES 256
#define SMEM_STOK_STRIDE 256
#define SMEM_SSCALE_OFF 185824
#define SMEM_SSCALE_STAGE_BYTES 288
#define SMEM_SSCALE_STRIDE 288
#define SMEM_SEXCH_OFF 186112
#define SMEM_SEXCH_STAGE_BYTES 2304
#define SMEM_SEXCH_STRIDE 2304
#define SMEM_TOTAL 188416
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


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mxfp4_situ_moe_5ffd3be1733f43c244bf(CakeTensorMap const* A, CakeTensorMap const* SFA, uint8_t* __restrict__ B, uint8_t* __restrict__ SFB, uint8_t* __restrict__ out, int* __restrict__ tile_idx_to_expert_idx, int* __restrict__ tile_idx_to_mn_limit, int* __restrict__ num_non_exiting_tiles, int* __restrict__ tile_idx_to_row_group, float* __restrict__ alpha, int* __restrict__ permuted_idx_to_expanded_idx, float* __restrict__ token_final_scales, int num_m_tiles, int group_capacity, int k_tiles, int k_cols, int sf_cols, int out_cols, int top_k, float* __restrict__ situ_beta, float* __restrict__ situ_linear_beta, uint8_t* __restrict__ act_sf, float* __restrict__ zero_buf, int zero_words, int num_rows_b, int act_cols, int act_sf_cols, int* __restrict__ dbg)
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
    uint8_t* sfa = reinterpret_cast<uint8_t*>(smem_raw + 175104);
    const int sfa_addr = smem + 175104;
    uint8_t* sfb = reinterpret_cast<uint8_t*>(smem_raw + 180224);
    const int sfb_addr = smem + 180224;
    int* sinfo = reinterpret_cast<int*>(smem_raw + 185344);
    const int sinfo_addr = smem + 185344;
    int* stok = reinterpret_cast<int*>(smem_raw + 185568);
    const int stok_addr = smem + 185568;
    float* sscale = reinterpret_cast<float*>(smem_raw + 185824);
    const int sscale_addr = smem + 185824;
    float* sexch = reinterpret_cast<float*>(smem_raw + 186112);
    const int sexch_addr = smem + 186112;

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

    // TMEM alloc (32 columns, 32 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 320);
    if (warp == 0) {
        int _tmem_hold = smem + 320;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(32) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_sf_a = taddr + 16;
    const int tmem_sf_b = taddr + 24;
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
            int cur_tok[8];
            float cur_scale[8];
            float meta_alpha = 0.0f;
            float vals[8];
            float amax[8];
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
            meta_alpha = sscale[tile_stage * 9 + 8];
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
                int row_base = info[1] * 8;
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
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], taddr + (unsigned int)(epi_warp * 32 << 16) + acc_stage * 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    vals[0] = _tmem_load_0[0] * meta_alpha;
                    vals[1] = _tmem_load_0[1] * meta_alpha;
                    vals[2] = _tmem_load_0[2] * meta_alpha;
                    vals[3] = _tmem_load_0[3] * meta_alpha;
                    vals[4] = _tmem_load_0[4] * meta_alpha;
                    vals[5] = _tmem_load_0[5] * meta_alpha;
                    vals[6] = _tmem_load_0[6] * meta_alpha;
                    vals[7] = _tmem_load_0[7] * meta_alpha;
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    {
                        if (is_gate_lane != 0) {
                            float x_g = vals[0];
                            float _exp2_0 = approx_exp2(x_g * -1.4426950408889634f);
                            float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                            float sig_g = _rcp_0;
                            float _tanh_approx_0;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(x_g * inv_beta));
                            sexch[exch_row * 9] = beta * _tanh_approx_0 * sig_g;
                            float x_g_0 = vals[1];
                            float _exp2_1 = approx_exp2(x_g_0 * -1.4426950408889634f);
                            float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                            float sig_g_1 = _rcp_1;
                            float _tanh_approx_1;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(x_g_0 * inv_beta));
                            sexch[exch_row * 9 + 1] = beta * _tanh_approx_1 * sig_g_1;
                            float x_g_2 = vals[2];
                            float _exp2_2 = approx_exp2(x_g_2 * -1.4426950408889634f);
                            float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                            float sig_g_3 = _rcp_2;
                            float _tanh_approx_2;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(x_g_2 * inv_beta));
                            sexch[exch_row * 9 + 2] = beta * _tanh_approx_2 * sig_g_3;
                            float x_g_4 = vals[3];
                            float _exp2_3 = approx_exp2(x_g_4 * -1.4426950408889634f);
                            float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                            float sig_g_5 = _rcp_3;
                            float _tanh_approx_3;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(x_g_4 * inv_beta));
                            sexch[exch_row * 9 + 3] = beta * _tanh_approx_3 * sig_g_5;
                            float x_g_6 = vals[4];
                            float _exp2_4 = approx_exp2(x_g_6 * -1.4426950408889634f);
                            float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                            float sig_g_7 = _rcp_4;
                            float _tanh_approx_4;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_4) : "f"(x_g_6 * inv_beta));
                            sexch[exch_row * 9 + 4] = beta * _tanh_approx_4 * sig_g_7;
                            float x_g_8 = vals[5];
                            float _exp2_5 = approx_exp2(x_g_8 * -1.4426950408889634f);
                            float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                            float sig_g_9 = _rcp_5;
                            float _tanh_approx_5;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_5) : "f"(x_g_8 * inv_beta));
                            sexch[exch_row * 9 + 5] = beta * _tanh_approx_5 * sig_g_9;
                            float x_g_10 = vals[6];
                            float _exp2_6 = approx_exp2(x_g_10 * -1.4426950408889634f);
                            float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                            float sig_g_11 = _rcp_6;
                            float _tanh_approx_6;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_6) : "f"(x_g_10 * inv_beta));
                            sexch[exch_row * 9 + 6] = beta * _tanh_approx_6 * sig_g_11;
                            float x_g_12 = vals[7];
                            float _exp2_7 = approx_exp2(x_g_12 * -1.4426950408889634f);
                            float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                            float sig_g_13 = _rcp_7;
                            float _tanh_approx_7;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_7) : "f"(x_g_12 * inv_beta));
                            sexch[exch_row * 9 + 7] = beta * _tanh_approx_7 * sig_g_13;
                        } else {
                            float _tanh_approx_8;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_8) : "f"(vals[0] * inv_linear_beta));
                            vals[0] = linear_beta * _tanh_approx_8;
                            float _tanh_approx_9;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_9) : "f"(vals[1] * inv_linear_beta));
                            vals[1] = linear_beta * _tanh_approx_9;
                            float _tanh_approx_10;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_10) : "f"(vals[2] * inv_linear_beta));
                            vals[2] = linear_beta * _tanh_approx_10;
                            float _tanh_approx_11;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_11) : "f"(vals[3] * inv_linear_beta));
                            vals[3] = linear_beta * _tanh_approx_11;
                            float _tanh_approx_12;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_12) : "f"(vals[4] * inv_linear_beta));
                            vals[4] = linear_beta * _tanh_approx_12;
                            float _tanh_approx_13;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_13) : "f"(vals[5] * inv_linear_beta));
                            vals[5] = linear_beta * _tanh_approx_13;
                            float _tanh_approx_14;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_14) : "f"(vals[6] * inv_linear_beta));
                            vals[6] = linear_beta * _tanh_approx_14;
                            float _tanh_approx_15;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_15) : "f"(vals[7] * inv_linear_beta));
                            vals[7] = linear_beta * _tanh_approx_15;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        if (is_gate_lane == 0) {
                            float v_u = vals[0] * sexch[exch_row * 9];
                            vals[0] = v_u;
                            float _fmax_0 = fmaxf(v_u, -v_u);
                            amax[0] = _fmax_0;
                            float v_u_0 = vals[1] * sexch[exch_row * 9 + 1];
                            vals[1] = v_u_0;
                            float _fmax_1 = fmaxf(v_u_0, -v_u_0);
                            amax[1] = _fmax_1;
                            float v_u_1 = vals[2] * sexch[exch_row * 9 + 2];
                            vals[2] = v_u_1;
                            float _fmax_2 = fmaxf(v_u_1, -v_u_1);
                            amax[2] = _fmax_2;
                            float v_u_2 = vals[3] * sexch[exch_row * 9 + 3];
                            vals[3] = v_u_2;
                            float _fmax_3 = fmaxf(v_u_2, -v_u_2);
                            amax[3] = _fmax_3;
                            float v_u_3 = vals[4] * sexch[exch_row * 9 + 4];
                            vals[4] = v_u_3;
                            float _fmax_4 = fmaxf(v_u_3, -v_u_3);
                            amax[4] = _fmax_4;
                            float v_u_4 = vals[5] * sexch[exch_row * 9 + 5];
                            vals[5] = v_u_4;
                            float _fmax_5 = fmaxf(v_u_4, -v_u_4);
                            amax[5] = _fmax_5;
                            float v_u_5 = vals[6] * sexch[exch_row * 9 + 6];
                            vals[6] = v_u_5;
                            float _fmax_6 = fmaxf(v_u_5, -v_u_5);
                            amax[6] = _fmax_6;
                            float v_u_6 = vals[7] * sexch[exch_row * 9 + 7];
                            vals[7] = v_u_6;
                            float _fmax_7 = fmaxf(v_u_6, -v_u_6);
                            amax[7] = _fmax_7;
                            float a_c = amax[0];
                            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, a_c, 1);
                            float _fmax_8 = fmaxf(a_c, _shfl_xor_0);
                            a_c = _fmax_8;
                            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, a_c, 2);
                            float _fmax_9 = fmaxf(a_c, _shfl_xor_1);
                            a_c = _fmax_9;
                            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, a_c, 4);
                            float _fmax_10 = fmaxf(a_c, _shfl_xor_2);
                            a_c = _fmax_10;
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, a_c, 8);
                            float _fmax_11 = fmaxf(a_c, _shfl_xor_3);
                            a_c = _fmax_11;
                            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, a_c, 16);
                            float _fmax_12 = fmaxf(a_c, _shfl_xor_4);
                            a_c = _fmax_12;
                            amax[0] = a_c;
                            float a_c_7 = amax[1];
                            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, a_c_7, 1);
                            float _fmax_13 = fmaxf(a_c_7, _shfl_xor_5);
                            a_c_7 = _fmax_13;
                            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, a_c_7, 2);
                            float _fmax_14 = fmaxf(a_c_7, _shfl_xor_6);
                            a_c_7 = _fmax_14;
                            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, a_c_7, 4);
                            float _fmax_15 = fmaxf(a_c_7, _shfl_xor_7);
                            a_c_7 = _fmax_15;
                            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, a_c_7, 8);
                            float _fmax_16 = fmaxf(a_c_7, _shfl_xor_8);
                            a_c_7 = _fmax_16;
                            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, a_c_7, 16);
                            float _fmax_17 = fmaxf(a_c_7, _shfl_xor_9);
                            a_c_7 = _fmax_17;
                            amax[1] = a_c_7;
                            float a_c_8 = amax[2];
                            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, a_c_8, 1);
                            float _fmax_18 = fmaxf(a_c_8, _shfl_xor_10);
                            a_c_8 = _fmax_18;
                            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, a_c_8, 2);
                            float _fmax_19 = fmaxf(a_c_8, _shfl_xor_11);
                            a_c_8 = _fmax_19;
                            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, a_c_8, 4);
                            float _fmax_20 = fmaxf(a_c_8, _shfl_xor_12);
                            a_c_8 = _fmax_20;
                            float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, a_c_8, 8);
                            float _fmax_21 = fmaxf(a_c_8, _shfl_xor_13);
                            a_c_8 = _fmax_21;
                            float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, a_c_8, 16);
                            float _fmax_22 = fmaxf(a_c_8, _shfl_xor_14);
                            a_c_8 = _fmax_22;
                            amax[2] = a_c_8;
                            float a_c_9 = amax[3];
                            float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, a_c_9, 1);
                            float _fmax_23 = fmaxf(a_c_9, _shfl_xor_15);
                            a_c_9 = _fmax_23;
                            float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, a_c_9, 2);
                            float _fmax_24 = fmaxf(a_c_9, _shfl_xor_16);
                            a_c_9 = _fmax_24;
                            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, a_c_9, 4);
                            float _fmax_25 = fmaxf(a_c_9, _shfl_xor_17);
                            a_c_9 = _fmax_25;
                            float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, a_c_9, 8);
                            float _fmax_26 = fmaxf(a_c_9, _shfl_xor_18);
                            a_c_9 = _fmax_26;
                            float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, a_c_9, 16);
                            float _fmax_27 = fmaxf(a_c_9, _shfl_xor_19);
                            a_c_9 = _fmax_27;
                            amax[3] = a_c_9;
                            float a_c_10 = amax[4];
                            float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, a_c_10, 1);
                            float _fmax_28 = fmaxf(a_c_10, _shfl_xor_20);
                            a_c_10 = _fmax_28;
                            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, a_c_10, 2);
                            float _fmax_29 = fmaxf(a_c_10, _shfl_xor_21);
                            a_c_10 = _fmax_29;
                            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, a_c_10, 4);
                            float _fmax_30 = fmaxf(a_c_10, _shfl_xor_22);
                            a_c_10 = _fmax_30;
                            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, a_c_10, 8);
                            float _fmax_31 = fmaxf(a_c_10, _shfl_xor_23);
                            a_c_10 = _fmax_31;
                            float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, a_c_10, 16);
                            float _fmax_32 = fmaxf(a_c_10, _shfl_xor_24);
                            a_c_10 = _fmax_32;
                            amax[4] = a_c_10;
                            float a_c_11 = amax[5];
                            float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, a_c_11, 1);
                            float _fmax_33 = fmaxf(a_c_11, _shfl_xor_25);
                            a_c_11 = _fmax_33;
                            float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, a_c_11, 2);
                            float _fmax_34 = fmaxf(a_c_11, _shfl_xor_26);
                            a_c_11 = _fmax_34;
                            float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, a_c_11, 4);
                            float _fmax_35 = fmaxf(a_c_11, _shfl_xor_27);
                            a_c_11 = _fmax_35;
                            float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, a_c_11, 8);
                            float _fmax_36 = fmaxf(a_c_11, _shfl_xor_28);
                            a_c_11 = _fmax_36;
                            float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, a_c_11, 16);
                            float _fmax_37 = fmaxf(a_c_11, _shfl_xor_29);
                            a_c_11 = _fmax_37;
                            amax[5] = a_c_11;
                            float a_c_12 = amax[6];
                            float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, a_c_12, 1);
                            float _fmax_38 = fmaxf(a_c_12, _shfl_xor_30);
                            a_c_12 = _fmax_38;
                            float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, a_c_12, 2);
                            float _fmax_39 = fmaxf(a_c_12, _shfl_xor_31);
                            a_c_12 = _fmax_39;
                            float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, a_c_12, 4);
                            float _fmax_40 = fmaxf(a_c_12, _shfl_xor_32);
                            a_c_12 = _fmax_40;
                            float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, a_c_12, 8);
                            float _fmax_41 = fmaxf(a_c_12, _shfl_xor_33);
                            a_c_12 = _fmax_41;
                            float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, a_c_12, 16);
                            float _fmax_42 = fmaxf(a_c_12, _shfl_xor_34);
                            a_c_12 = _fmax_42;
                            amax[6] = a_c_12;
                            float a_c_13 = amax[7];
                            float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, a_c_13, 1);
                            float _fmax_43 = fmaxf(a_c_13, _shfl_xor_35);
                            a_c_13 = _fmax_43;
                            float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, a_c_13, 2);
                            float _fmax_44 = fmaxf(a_c_13, _shfl_xor_36);
                            a_c_13 = _fmax_44;
                            float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, a_c_13, 4);
                            float _fmax_45 = fmaxf(a_c_13, _shfl_xor_37);
                            a_c_13 = _fmax_45;
                            float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, a_c_13, 8);
                            float _fmax_46 = fmaxf(a_c_13, _shfl_xor_38);
                            a_c_13 = _fmax_46;
                            float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, a_c_13, 16);
                            float _fmax_47 = fmaxf(a_c_13, _shfl_xor_39);
                            a_c_13 = _fmax_47;
                            amax[7] = a_c_13;
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
                            int prow_e_14 = row_base + 1;
                            if (prow_e_14 < mn_limit) {
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
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_14 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_1 = (unsigned int)code_1;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_14 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_1);
                                }
                            }
                            int prow_e_15 = row_base + 2;
                            if (prow_e_15 < mn_limit) {
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
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_15 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_2 = (unsigned int)code_2;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_15 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_2);
                                }
                            }
                            int prow_e_16 = row_base + 3;
                            if (prow_e_16 < mn_limit) {
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
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_16 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_3 = (unsigned int)code_3;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_16 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_3);
                                }
                            }
                            int prow_e_17 = row_base + 4;
                            if (prow_e_17 < mn_limit) {
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
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_17 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_4 = (unsigned int)code_4;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_17 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_4);
                                }
                            }
                            int prow_e_18 = row_base + 5;
                            if (prow_e_18 < mn_limit) {
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
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_18 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_5 = (unsigned int)code_5;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_18 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_5);
                                }
                            }
                            int prow_e_19 = row_base + 6;
                            if (prow_e_19 < mn_limit) {
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
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_19 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_6 = (unsigned int)code_6;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_19 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_6);
                                }
                            }
                            int prow_e_20 = row_base + 7;
                            if (prow_e_20 < mn_limit) {
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
                                    *(reinterpret_cast<unsigned char*>(out + (prow_e_20 * act_cols + j_col)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                                }
                                if (lane_0 == 0) {
                                    unsigned int code_u32_7 = (unsigned int)code_7;
                                    *(reinterpret_cast<unsigned char*>(act_sf + (prow_e_20 * act_sf_cols + info[0] * 2 + epi_tidx / 32)) + (0)) = (unsigned char)(code_u32_7);
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
                meta_alpha = sscale[tile_stage * 9 + 8];
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
                        tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 64)));
                        tcgen05_cp_32x128b_warpx4((tmem_sf_b + 4), make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 64 + 32)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa) * 64)));
                        tcgen05_cp_32x128b_warpx4((tmem_sf_a + 4), make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa) * 64 + 32)));
                    }
                    int _mma_a_lo_0 = make_warp_uniform((((a_addr) >> 4) & 0x3FFF) + (sa) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform((((b_addr) >> 4) & 0x3FFF) + (sb) * 128);
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                            0x8820280U, tmem_sf_a, tmem_sf_b, ((((1) ? 1 : 0)) ? 0 : 1));
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 2, b_desc + 2,
                            0x28820290U, tmem_sf_a, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 4, b_desc + 4,
                            0x488202a0U, tmem_sf_a, tmem_sf_b, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 6, b_desc + 6,
                            0x688202b0U, tmem_sf_a, tmem_sf_b, 1);
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((a_addr + 16384) >> 4) & 0x3FFF) + (sa) * 2048);
                    int _mma_b_lo_1 = make_warp_uniform((((b_addr + 1024) >> 4) & 0x3FFF) + (sb) * 128);
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                            0x8820280U, tmem_sf_a + 4, tmem_sf_b + 4, ((((0) ? 1 : 0)) ? 0 : 1));
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 2, b_desc + 2,
                            0x28820290U, tmem_sf_a + 4, tmem_sf_b + 4, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 4, b_desc + 4,
                            0x488202a0U, tmem_sf_a + 4, tmem_sf_b + 4, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 6, b_desc + 6,
                            0x688202b0U, tmem_sf_a + 4, tmem_sf_b + 4, 1);
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
                            tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 64)));
                            tcgen05_cp_32x128b_warpx4((tmem_sf_b + 4), make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sb) * 64 + 32)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa) * 64)));
                            tcgen05_cp_32x128b_warpx4((tmem_sf_a + 4), make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa) * 64 + 32)));
                        }
                        int _mma_a_lo_2 = make_warp_uniform((((a_addr) >> 4) & 0x3FFF) + (sa) * 2048);
                        int _mma_b_lo_2 = make_warp_uniform((((b_addr) >> 4) & 0x3FFF) + (sb) * 128);
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                0x8820280U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 2, b_desc + 2,
                                0x28820290U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 4, b_desc + 4,
                                0x488202a0U, tmem_sf_a, tmem_sf_b, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 6, b_desc + 6,
                                0x688202b0U, tmem_sf_a, tmem_sf_b, 1);
                        }
                        int _mma_a_lo_3 = make_warp_uniform((((a_addr + 16384) >> 4) & 0x3FFF) + (sa) * 2048);
                        int _mma_b_lo_3 = make_warp_uniform((((b_addr + 1024) >> 4) & 0x3FFF) + (sb) * 128);
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 0, b_desc + 0,
                                0x8820280U, tmem_sf_a + 4, tmem_sf_b + 4, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 2, b_desc + 2,
                                0x28820290U, tmem_sf_a + 4, tmem_sf_b + 4, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 4, b_desc + 4,
                                0x488202a0U, tmem_sf_a + 4, tmem_sf_b + 4, 1);
                            tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_stage_1 * 8)), a_desc + 6, b_desc + 6,
                                0x688202b0U, tmem_sf_a + 4, tmem_sf_b + 4, 1);
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
            int batch[1];
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (int _tile_2 = 0; _tile_2 < num_m_tiles * group_capacity + 1; _tile_2++) {
                if (info_2[3] == 0) {
                    break;
                }
                batch[0] = info_2[2] * num_m_tiles + info_2[0];
                int row_base_tma = info_2[1] * 8;
                #pragma unroll 1
                for (int k_1 = 0; k_1 < k_tiles; k_1++) {
                    mbarrier_wait(ab_free_addr + (stage) * 8, _phase_ab_free);
                    if (elect_sync()) {
                        {
                            mbarrier_arrive_expect_tx(ab_full_addr + (stage) * 8, 17408);
                            tma_4d_gmem2smem(a_addr + stage * 32768, A, 0, 0, (info_2[5] + k_1) * 2, batch[0], ab_full_addr + (stage) * 8);
                            tma_4d_gmem2smem(sfa_addr + stage * 1024, SFA, 0, 0, (info_2[5] + k_1) * 2, batch[0], ab_full_addr + (stage) * 8);
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
                    sscale[tile_stage_3 * 9 + 8] = alpha[expert];
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
            int row_src[2];
            int row_ok[2];
            int sf_src[1];
            int sf_ok[1];
            int cta_row0 = 0;
            unsigned int _phase_b_free = 1;
            #pragma unroll 1
            for (int _tile_3 = 0; _tile_3 < num_m_tiles * group_capacity + 1; _tile_3++) {
                if (info_3[3] == 0) {
                    break;
                }
                int row_base_1 = info_3[1] * 8;
                int mn_limit_2 = info_3[4];
                int row = gather_sub * 4 + row_in_pass;
                int prow = row_base_1 + cta_row0 + row;
                int _min_0 = ((prow) < (row_base_1 + 7) ? (prow) : (row_base_1 + 7));
                int safe_row = _min_0;
                int expanded = permuted_idx_to_expanded_idx[safe_row];
                int tok_row = expanded / top_k;
                int ok = (int)(prow < mn_limit_2 && expanded >= 0 && tok_row < num_rows_b);
                row_src[0] = tok_row * ok;
                row_ok[0] = ok;
                int row_0 = (gather_sub + 1) * 4 + row_in_pass;
                int prow_1 = row_base_1 + cta_row0 + row_0;
                int _min_1 = ((prow_1) < (row_base_1 + 7) ? (prow_1) : (row_base_1 + 7));
                int safe_row_2 = _min_1;
                int expanded_3 = permuted_idx_to_expanded_idx[safe_row_2];
                int tok_row_4 = expanded_3 / top_k;
                int ok_5 = (int)(prow_1 < mn_limit_2 && expanded_3 >= 0 && tok_row_4 < num_rows_b);
                row_src[1] = tok_row_4 * ok_5;
                row_ok[1] = ok_5;
                int srow = gather_sub * 32 + lane_0_1;
                int sprow = row_base_1 + srow;
                int _min_2 = ((sprow) < (row_base_1 + 7) ? (sprow) : (row_base_1 + 7));
                int safe_srow = _min_2;
                int sexpanded = permuted_idx_to_expanded_idx[safe_srow];
                int stok_row = sexpanded / top_k;
                int sok = (int)(sprow < mn_limit_2 && srow < 8 && sexpanded >= 0 && stok_row < num_rows_b);
                sf_src[0] = stok_row * sok;
                sf_ok[0] = sok;
                #pragma unroll 1
                for (int k_2 = 0; k_2 < k_tiles; k_2++) {
                    mbarrier_wait(b_free_addr + (stage_1) * 8, _phase_b_free);
                    int k0 = (info_3[5] + k_2) * 256;
                    int dst_off = (gather_sub * 4 + row_in_pass) * 128 + (chunk ^ (gather_sub * 4 + row_in_pass) % 8) * 16;
                    int src_off = row_src[0] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 2048 + (unsigned int)dst_off), "l"(B + src_off), "r"((row_ok[0] != 0) ? 16 : 0));
                    }
                    int dst_off_0 = ((gather_sub + 1) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 1) * 4 + row_in_pass) % 8) * 16;
                    int src_off_1 = row_src[1] * k_cols + k0 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 2048 + (unsigned int)dst_off_0), "l"(B + src_off_1), "r"((row_ok[1] != 0) ? 16 : 0));
                    }
                    int sf_src_off = sf_src[0] * sf_cols + (info_3[5] + k_2) * 8;
                    {
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(sfb_addr + stage_1 * 1024 + (unsigned int)(gather_sub / 4 * 512 * 2) + (unsigned int)(lane_0_1 * 16) + (unsigned int)(gather_sub % 4 * 4)), "l"(SFB + sf_src_off), "r"((sf_ok[0] != 0) ? 4 : 0));
                    }
                    int dst_off_2 = 1024 + (gather_sub * 4 + row_in_pass) * 128 + (chunk ^ (gather_sub * 4 + row_in_pass) % 8) * 16;
                    int src_off_3 = row_src[0] * k_cols + k0 + 128 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 2048 + (unsigned int)dst_off_2), "l"(B + src_off_3), "r"((row_ok[0] != 0) ? 16 : 0));
                    }
                    int dst_off_4 = 1024 + ((gather_sub + 1) * 4 + row_in_pass) * 128 + (chunk ^ ((gather_sub + 1) * 4 + row_in_pass) % 8) * 16;
                    int src_off_5 = row_src[1] * k_cols + k0 + 128 + chunk * 16;
                    {
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(b_addr + stage_1 * 2048 + (unsigned int)dst_off_4), "l"(B + src_off_5), "r"((row_ok[1] != 0) ? 16 : 0));
                    }
                    int sf_src_off_6 = sf_src[0] * sf_cols + (info_3[5] + k_2) * 8 + 4;
                    {
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(sfb_addr + stage_1 * 1024 + 512 + (unsigned int)(gather_sub / 4 * 512 * 2) + (unsigned int)(lane_0_1 * 16) + (unsigned int)(gather_sub % 4 * 4)), "l"(SFB + sf_src_off_6), "r"((sf_ok[0] != 0) ? 4 : 0));
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
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(32));
    }

    // Kernel epilogue ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
