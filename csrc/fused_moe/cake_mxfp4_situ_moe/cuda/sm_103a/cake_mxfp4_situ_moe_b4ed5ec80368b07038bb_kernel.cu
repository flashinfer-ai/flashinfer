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
#define TMEM_NCOLS 476
#define TMEM_ACC_OFFSET 0
#define TMEM_SF_A_OFFSET 464
#define TMEM_SF_B_OFFSET 468
#define NUM_PAB_STAGES 3
#define NUM_PACC_STAGES 1
#define NUM_PTILE_STAGES 2
#define NUM_PMETA_STAGES 2
#define SMEM_A_OFF 1024
#define SMEM_A_STAGE_BYTES 16384
#define SMEM_A_STRIDE 16384
#define SMEM_B_OFF 50176
#define SMEM_B_STAGE_BYTES 32768
#define SMEM_B_STRIDE 32768
#define SMEM_SFA_OFF 148480
#define SMEM_SFA_STAGE_BYTES 512
#define SMEM_SFA_STRIDE 512
#define SMEM_SFB_OFF 150016
#define SMEM_SFB_STAGE_BYTES 1024
#define SMEM_SFB_STRIDE 1024
#define SMEM_SC_OFF 153600
#define SMEM_SC_STAGE_BYTES 67584
#define SMEM_SC_STRIDE 67584
#define SMEM_SINFO_OFF 221184
#define SMEM_SINFO_STAGE_BYTES 40
#define SMEM_SINFO_STRIDE 40
#define SMEM_SMETA_TOK_OFF 221232
#define SMEM_SMETA_TOK_STAGE_BYTES 1024
#define SMEM_SMETA_TOK_STRIDE 1024
#define SMEM_SMETA_SCALE_OFF 222256
#define SMEM_SMETA_SCALE_STAGE_BYTES 1024
#define SMEM_SMETA_SCALE_STRIDE 1024
#define SMEM_TOTAL 223360
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mxfp4_situ_moe_b4ed5ec80368b07038bb(CakeTensorMap const* A, CakeTensorMap const* B, CakeTensorMap const* SFA, CakeTensorMap const* SFB, __nv_bfloat16* __restrict__ out, int* __restrict__ tile_idx_to_expert_idx, int* __restrict__ tile_idx_to_mn_limit, int* __restrict__ num_non_exiting_tiles, float* __restrict__ alpha, int* __restrict__ permuted_idx_to_expanded_idx, float* __restrict__ token_final_scales, int* __restrict__ tile_idx_to_row_group, int num_m_tiles, int num_n_tiles, int k_tiles, int out_cols, int top_k, int* __restrict__ dbg)
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
    #define ab_free_addr (mbar_base + 24)
    #define acc_full_addr (mbar_base + 48)
    #define acc_free_addr (mbar_base + 56)
    #define tile_full_addr (mbar_base + 64)
    #define tile_free_addr (mbar_base + 80)
    #define meta_full_addr (mbar_base + 96)
    #define meta_free_addr (mbar_base + 112)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFA)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFB)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int a_addr = smem + 1024;
    uint8_t* b = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int b_addr = smem + 50176;
    uint8_t* sfa = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int sfa_addr = smem + 148480;
    uint8_t* sfb = reinterpret_cast<uint8_t*>(smem_raw + 150016);
    const int sfb_addr = smem + 150016;
    uint8_t* sc = reinterpret_cast<uint8_t*>(smem_raw + 153600);
    const int sc_addr = smem + 153600;
    int* sinfo = reinterpret_cast<int*>(smem_raw + 221184);
    const int sinfo_addr = smem + 221184;
    int* smeta_tok = reinterpret_cast<int*>(smem_raw + 221232);
    const int smeta_tok_addr = smem + 221232;
    float* smeta_scale = reinterpret_cast<float*>(smem_raw + 222256);
    const int smeta_scale_addr = smem + 222256;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        // --- pipeline 'pab' ---
        // ab_full: 3 barriers, init_count=1
        // ab_free: 3 barriers, init_count=1
        // --- pipeline 'pacc' ---
        // acc_full: 1 barriers, init_count=1
        // acc_free: 1 barriers, init_count=128
        // --- pipeline 'ptile' ---
        // tile_full: 2 barriers, init_count=32
        // tile_free: 2 barriers, init_count=224
        // --- pipeline 'pmeta' ---
        // meta_full: 2 barriers, init_count=32
        // meta_free: 2 barriers, init_count=128
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 128;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(14), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(12), "r"((uint32_t)(224)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(10), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(8), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(7), "r"((uint32_t)(1)));
        if (lane < 16) {
            mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 476 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_sf_a = taddr + 464;
    const int tmem_sf_b = taddr + 468;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            int epi_tidx = tid;
            int epi_warp = warp;
            unsigned int acc_stage = 0;
            unsigned int tile_stage = 0;
            unsigned int meta_stage = 0;
            unsigned int acc_buf = 0;
            int info[5];
            float vals[32];
            unsigned int packed[16];
            unsigned int _phase_tile_full = 0;
            mbarrier_wait(tile_full_addr + (tile_stage) * 8, _phase_tile_full);
            info[0] = sinfo[tile_stage * 5];
            info[1] = sinfo[tile_stage * 5 + 1];
            info[2] = sinfo[tile_stage * 5 + 2];
            info[3] = sinfo[tile_stage * 5 + 3];
            info[4] = sinfo[tile_stage * 5 + 4];
            int e_coord_m = sinfo[tile_stage * 5];
            int e_coord_n = sinfo[tile_stage * 5 + 1];
            int e_limit = sinfo[tile_stage * 5 + 4];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage) * 8);
            tile_stage += 1;
            if (tile_stage == 2) { tile_stage = 0; _phase_tile_full ^= 1; }
            unsigned int _phase_meta_full = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int _tile = 0; _tile < num_m_tiles * num_n_tiles + 1; _tile++) {
                if (info[3] == 0) {
                    break;
                }
                int row_base = e_coord_m * 128;
                int col_base = e_coord_n * 256;
                int prow = row_base + epi_tidx;
                int is_valid_row = (int)(prow < e_limit);
                int is_partial = (int)(e_limit < row_base + 128);
                int _min_0 = ((out_cols - col_base) < (256) ? (out_cols - col_base) : (256));
                int valid_cols = _min_0;
                mbarrier_wait(meta_full_addr + (meta_stage) * 8, _phase_meta_full);
                float meta_scale = smeta_scale[meta_stage * 128 + (unsigned int)epi_tidx];
                mbarrier_wait(acc_full_addr + (acc_stage) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int rev = 1 - acc_buf;
                unsigned int real_sub = 7 * rev;
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_buf * 208 + real_sub * 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                vals[0] = _tmem_load_0[0] * meta_scale;
                vals[1] = _tmem_load_0[1] * meta_scale;
                vals[2] = _tmem_load_0[2] * meta_scale;
                vals[3] = _tmem_load_0[3] * meta_scale;
                vals[4] = _tmem_load_0[4] * meta_scale;
                vals[5] = _tmem_load_0[5] * meta_scale;
                vals[6] = _tmem_load_0[6] * meta_scale;
                vals[7] = _tmem_load_0[7] * meta_scale;
                vals[8] = _tmem_load_0[8] * meta_scale;
                vals[9] = _tmem_load_0[9] * meta_scale;
                vals[10] = _tmem_load_0[10] * meta_scale;
                vals[11] = _tmem_load_0[11] * meta_scale;
                vals[12] = _tmem_load_0[12] * meta_scale;
                vals[13] = _tmem_load_0[13] * meta_scale;
                vals[14] = _tmem_load_0[14] * meta_scale;
                vals[15] = _tmem_load_0[15] * meta_scale;
                vals[16] = _tmem_load_0[16] * meta_scale;
                vals[17] = _tmem_load_0[17] * meta_scale;
                vals[18] = _tmem_load_0[18] * meta_scale;
                vals[19] = _tmem_load_0[19] * meta_scale;
                vals[20] = _tmem_load_0[20] * meta_scale;
                vals[21] = _tmem_load_0[21] * meta_scale;
                vals[22] = _tmem_load_0[22] * meta_scale;
                vals[23] = _tmem_load_0[23] * meta_scale;
                vals[24] = _tmem_load_0[24] * meta_scale;
                vals[25] = _tmem_load_0[25] * meta_scale;
                vals[26] = _tmem_load_0[26] * meta_scale;
                vals[27] = _tmem_load_0[27] * meta_scale;
                vals[28] = _tmem_load_0[28] * meta_scale;
                vals[29] = _tmem_load_0[29] * meta_scale;
                vals[30] = _tmem_load_0[30] * meta_scale;
                vals[31] = _tmem_load_0[31] * meta_scale;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                if (is_valid_row != 0) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + real_sub * 64))), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub * 64 + 16)))), "r"(packed[4]), "r"(packed[5]), "r"(packed[6]), "r"(packed[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub * 64 + 32)))), "r"(packed[8]), "r"(packed[9]), "r"(packed[10]), "r"(packed[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub * 64 + 48)))), "r"(packed[12]), "r"(packed[13]), "r"(packed[14]), "r"(packed[15]) : "memory");
                }
                unsigned int real_sub_0 = acc_buf + 6 * rev;
                float _tmem_load_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_buf * 208 + real_sub_0 * 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(acc_free_addr + (acc_stage) * 8);
                    _phase_acc_full ^= 1;
                }
                vals[0] = _tmem_load_1[0] * meta_scale;
                vals[1] = _tmem_load_1[1] * meta_scale;
                vals[2] = _tmem_load_1[2] * meta_scale;
                vals[3] = _tmem_load_1[3] * meta_scale;
                vals[4] = _tmem_load_1[4] * meta_scale;
                vals[5] = _tmem_load_1[5] * meta_scale;
                vals[6] = _tmem_load_1[6] * meta_scale;
                vals[7] = _tmem_load_1[7] * meta_scale;
                vals[8] = _tmem_load_1[8] * meta_scale;
                vals[9] = _tmem_load_1[9] * meta_scale;
                vals[10] = _tmem_load_1[10] * meta_scale;
                vals[11] = _tmem_load_1[11] * meta_scale;
                vals[12] = _tmem_load_1[12] * meta_scale;
                vals[13] = _tmem_load_1[13] * meta_scale;
                vals[14] = _tmem_load_1[14] * meta_scale;
                vals[15] = _tmem_load_1[15] * meta_scale;
                vals[16] = _tmem_load_1[16] * meta_scale;
                vals[17] = _tmem_load_1[17] * meta_scale;
                vals[18] = _tmem_load_1[18] * meta_scale;
                vals[19] = _tmem_load_1[19] * meta_scale;
                vals[20] = _tmem_load_1[20] * meta_scale;
                vals[21] = _tmem_load_1[21] * meta_scale;
                vals[22] = _tmem_load_1[22] * meta_scale;
                vals[23] = _tmem_load_1[23] * meta_scale;
                vals[24] = _tmem_load_1[24] * meta_scale;
                vals[25] = _tmem_load_1[25] * meta_scale;
                vals[26] = _tmem_load_1[26] * meta_scale;
                vals[27] = _tmem_load_1[27] * meta_scale;
                vals[28] = _tmem_load_1[28] * meta_scale;
                vals[29] = _tmem_load_1[29] * meta_scale;
                vals[30] = _tmem_load_1[30] * meta_scale;
                vals[31] = _tmem_load_1[31] * meta_scale;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                if (is_valid_row != 0) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + real_sub_0 * 64))), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_0 * 64 + 16)))), "r"(packed[4]), "r"(packed[5]), "r"(packed[6]), "r"(packed[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_0 * 64 + 32)))), "r"(packed[8]), "r"(packed[9]), "r"(packed[10]), "r"(packed[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_0 * 64 + 48)))), "r"(packed[12]), "r"(packed[13]), "r"(packed[14]), "r"(packed[15]) : "memory");
                }
                unsigned int real_sub_1 = 2 * acc_buf + 5 * rev;
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_buf * 208 + real_sub_1 * 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                vals[0] = _tmem_load_2[0] * meta_scale;
                vals[1] = _tmem_load_2[1] * meta_scale;
                vals[2] = _tmem_load_2[2] * meta_scale;
                vals[3] = _tmem_load_2[3] * meta_scale;
                vals[4] = _tmem_load_2[4] * meta_scale;
                vals[5] = _tmem_load_2[5] * meta_scale;
                vals[6] = _tmem_load_2[6] * meta_scale;
                vals[7] = _tmem_load_2[7] * meta_scale;
                vals[8] = _tmem_load_2[8] * meta_scale;
                vals[9] = _tmem_load_2[9] * meta_scale;
                vals[10] = _tmem_load_2[10] * meta_scale;
                vals[11] = _tmem_load_2[11] * meta_scale;
                vals[12] = _tmem_load_2[12] * meta_scale;
                vals[13] = _tmem_load_2[13] * meta_scale;
                vals[14] = _tmem_load_2[14] * meta_scale;
                vals[15] = _tmem_load_2[15] * meta_scale;
                vals[16] = _tmem_load_2[16] * meta_scale;
                vals[17] = _tmem_load_2[17] * meta_scale;
                vals[18] = _tmem_load_2[18] * meta_scale;
                vals[19] = _tmem_load_2[19] * meta_scale;
                vals[20] = _tmem_load_2[20] * meta_scale;
                vals[21] = _tmem_load_2[21] * meta_scale;
                vals[22] = _tmem_load_2[22] * meta_scale;
                vals[23] = _tmem_load_2[23] * meta_scale;
                vals[24] = _tmem_load_2[24] * meta_scale;
                vals[25] = _tmem_load_2[25] * meta_scale;
                vals[26] = _tmem_load_2[26] * meta_scale;
                vals[27] = _tmem_load_2[27] * meta_scale;
                vals[28] = _tmem_load_2[28] * meta_scale;
                vals[29] = _tmem_load_2[29] * meta_scale;
                vals[30] = _tmem_load_2[30] * meta_scale;
                vals[31] = _tmem_load_2[31] * meta_scale;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                if (is_valid_row != 0) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + real_sub_1 * 64))), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_1 * 64 + 16)))), "r"(packed[4]), "r"(packed[5]), "r"(packed[6]), "r"(packed[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_1 * 64 + 32)))), "r"(packed[8]), "r"(packed[9]), "r"(packed[10]), "r"(packed[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_1 * 64 + 48)))), "r"(packed[12]), "r"(packed[13]), "r"(packed[14]), "r"(packed[15]) : "memory");
                }
                unsigned int real_sub_2 = 3 * acc_buf + 4 * rev;
                float _tmem_load_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_buf * 208 + real_sub_2 * 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                vals[0] = _tmem_load_3[0] * meta_scale;
                vals[1] = _tmem_load_3[1] * meta_scale;
                vals[2] = _tmem_load_3[2] * meta_scale;
                vals[3] = _tmem_load_3[3] * meta_scale;
                vals[4] = _tmem_load_3[4] * meta_scale;
                vals[5] = _tmem_load_3[5] * meta_scale;
                vals[6] = _tmem_load_3[6] * meta_scale;
                vals[7] = _tmem_load_3[7] * meta_scale;
                vals[8] = _tmem_load_3[8] * meta_scale;
                vals[9] = _tmem_load_3[9] * meta_scale;
                vals[10] = _tmem_load_3[10] * meta_scale;
                vals[11] = _tmem_load_3[11] * meta_scale;
                vals[12] = _tmem_load_3[12] * meta_scale;
                vals[13] = _tmem_load_3[13] * meta_scale;
                vals[14] = _tmem_load_3[14] * meta_scale;
                vals[15] = _tmem_load_3[15] * meta_scale;
                vals[16] = _tmem_load_3[16] * meta_scale;
                vals[17] = _tmem_load_3[17] * meta_scale;
                vals[18] = _tmem_load_3[18] * meta_scale;
                vals[19] = _tmem_load_3[19] * meta_scale;
                vals[20] = _tmem_load_3[20] * meta_scale;
                vals[21] = _tmem_load_3[21] * meta_scale;
                vals[22] = _tmem_load_3[22] * meta_scale;
                vals[23] = _tmem_load_3[23] * meta_scale;
                vals[24] = _tmem_load_3[24] * meta_scale;
                vals[25] = _tmem_load_3[25] * meta_scale;
                vals[26] = _tmem_load_3[26] * meta_scale;
                vals[27] = _tmem_load_3[27] * meta_scale;
                vals[28] = _tmem_load_3[28] * meta_scale;
                vals[29] = _tmem_load_3[29] * meta_scale;
                vals[30] = _tmem_load_3[30] * meta_scale;
                vals[31] = _tmem_load_3[31] * meta_scale;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                if (is_valid_row != 0) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + real_sub_2 * 64))), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_2 * 64 + 16)))), "r"(packed[4]), "r"(packed[5]), "r"(packed[6]), "r"(packed[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_2 * 64 + 32)))), "r"(packed[8]), "r"(packed[9]), "r"(packed[10]), "r"(packed[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_2 * 64 + 48)))), "r"(packed[12]), "r"(packed[13]), "r"(packed[14]), "r"(packed[15]) : "memory");
                }
                unsigned int real_sub_3 = 4 * acc_buf + 3 * rev;
                float _tmem_load_4[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_buf * 208 + real_sub_3 * 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                vals[0] = _tmem_load_4[0] * meta_scale;
                vals[1] = _tmem_load_4[1] * meta_scale;
                vals[2] = _tmem_load_4[2] * meta_scale;
                vals[3] = _tmem_load_4[3] * meta_scale;
                vals[4] = _tmem_load_4[4] * meta_scale;
                vals[5] = _tmem_load_4[5] * meta_scale;
                vals[6] = _tmem_load_4[6] * meta_scale;
                vals[7] = _tmem_load_4[7] * meta_scale;
                vals[8] = _tmem_load_4[8] * meta_scale;
                vals[9] = _tmem_load_4[9] * meta_scale;
                vals[10] = _tmem_load_4[10] * meta_scale;
                vals[11] = _tmem_load_4[11] * meta_scale;
                vals[12] = _tmem_load_4[12] * meta_scale;
                vals[13] = _tmem_load_4[13] * meta_scale;
                vals[14] = _tmem_load_4[14] * meta_scale;
                vals[15] = _tmem_load_4[15] * meta_scale;
                vals[16] = _tmem_load_4[16] * meta_scale;
                vals[17] = _tmem_load_4[17] * meta_scale;
                vals[18] = _tmem_load_4[18] * meta_scale;
                vals[19] = _tmem_load_4[19] * meta_scale;
                vals[20] = _tmem_load_4[20] * meta_scale;
                vals[21] = _tmem_load_4[21] * meta_scale;
                vals[22] = _tmem_load_4[22] * meta_scale;
                vals[23] = _tmem_load_4[23] * meta_scale;
                vals[24] = _tmem_load_4[24] * meta_scale;
                vals[25] = _tmem_load_4[25] * meta_scale;
                vals[26] = _tmem_load_4[26] * meta_scale;
                vals[27] = _tmem_load_4[27] * meta_scale;
                vals[28] = _tmem_load_4[28] * meta_scale;
                vals[29] = _tmem_load_4[29] * meta_scale;
                vals[30] = _tmem_load_4[30] * meta_scale;
                vals[31] = _tmem_load_4[31] * meta_scale;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                if (is_valid_row != 0) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + real_sub_3 * 64))), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_3 * 64 + 16)))), "r"(packed[4]), "r"(packed[5]), "r"(packed[6]), "r"(packed[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_3 * 64 + 32)))), "r"(packed[8]), "r"(packed[9]), "r"(packed[10]), "r"(packed[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_3 * 64 + 48)))), "r"(packed[12]), "r"(packed[13]), "r"(packed[14]), "r"(packed[15]) : "memory");
                }
                unsigned int real_sub_4 = 5 * acc_buf + 2 * rev;
                float _tmem_load_5[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_buf * 208 + real_sub_4 * 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                vals[0] = _tmem_load_5[0] * meta_scale;
                vals[1] = _tmem_load_5[1] * meta_scale;
                vals[2] = _tmem_load_5[2] * meta_scale;
                vals[3] = _tmem_load_5[3] * meta_scale;
                vals[4] = _tmem_load_5[4] * meta_scale;
                vals[5] = _tmem_load_5[5] * meta_scale;
                vals[6] = _tmem_load_5[6] * meta_scale;
                vals[7] = _tmem_load_5[7] * meta_scale;
                vals[8] = _tmem_load_5[8] * meta_scale;
                vals[9] = _tmem_load_5[9] * meta_scale;
                vals[10] = _tmem_load_5[10] * meta_scale;
                vals[11] = _tmem_load_5[11] * meta_scale;
                vals[12] = _tmem_load_5[12] * meta_scale;
                vals[13] = _tmem_load_5[13] * meta_scale;
                vals[14] = _tmem_load_5[14] * meta_scale;
                vals[15] = _tmem_load_5[15] * meta_scale;
                vals[16] = _tmem_load_5[16] * meta_scale;
                vals[17] = _tmem_load_5[17] * meta_scale;
                vals[18] = _tmem_load_5[18] * meta_scale;
                vals[19] = _tmem_load_5[19] * meta_scale;
                vals[20] = _tmem_load_5[20] * meta_scale;
                vals[21] = _tmem_load_5[21] * meta_scale;
                vals[22] = _tmem_load_5[22] * meta_scale;
                vals[23] = _tmem_load_5[23] * meta_scale;
                vals[24] = _tmem_load_5[24] * meta_scale;
                vals[25] = _tmem_load_5[25] * meta_scale;
                vals[26] = _tmem_load_5[26] * meta_scale;
                vals[27] = _tmem_load_5[27] * meta_scale;
                vals[28] = _tmem_load_5[28] * meta_scale;
                vals[29] = _tmem_load_5[29] * meta_scale;
                vals[30] = _tmem_load_5[30] * meta_scale;
                vals[31] = _tmem_load_5[31] * meta_scale;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                if (is_valid_row != 0) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + real_sub_4 * 64))), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_4 * 64 + 16)))), "r"(packed[4]), "r"(packed[5]), "r"(packed[6]), "r"(packed[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_4 * 64 + 32)))), "r"(packed[8]), "r"(packed[9]), "r"(packed[10]), "r"(packed[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_4 * 64 + 48)))), "r"(packed[12]), "r"(packed[13]), "r"(packed[14]), "r"(packed[15]) : "memory");
                }
                unsigned int real_sub_5 = 6 * acc_buf + rev;
                float _tmem_load_6[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_buf * 208 + real_sub_5 * 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                vals[0] = _tmem_load_6[0] * meta_scale;
                vals[1] = _tmem_load_6[1] * meta_scale;
                vals[2] = _tmem_load_6[2] * meta_scale;
                vals[3] = _tmem_load_6[3] * meta_scale;
                vals[4] = _tmem_load_6[4] * meta_scale;
                vals[5] = _tmem_load_6[5] * meta_scale;
                vals[6] = _tmem_load_6[6] * meta_scale;
                vals[7] = _tmem_load_6[7] * meta_scale;
                vals[8] = _tmem_load_6[8] * meta_scale;
                vals[9] = _tmem_load_6[9] * meta_scale;
                vals[10] = _tmem_load_6[10] * meta_scale;
                vals[11] = _tmem_load_6[11] * meta_scale;
                vals[12] = _tmem_load_6[12] * meta_scale;
                vals[13] = _tmem_load_6[13] * meta_scale;
                vals[14] = _tmem_load_6[14] * meta_scale;
                vals[15] = _tmem_load_6[15] * meta_scale;
                vals[16] = _tmem_load_6[16] * meta_scale;
                vals[17] = _tmem_load_6[17] * meta_scale;
                vals[18] = _tmem_load_6[18] * meta_scale;
                vals[19] = _tmem_load_6[19] * meta_scale;
                vals[20] = _tmem_load_6[20] * meta_scale;
                vals[21] = _tmem_load_6[21] * meta_scale;
                vals[22] = _tmem_load_6[22] * meta_scale;
                vals[23] = _tmem_load_6[23] * meta_scale;
                vals[24] = _tmem_load_6[24] * meta_scale;
                vals[25] = _tmem_load_6[25] * meta_scale;
                vals[26] = _tmem_load_6[26] * meta_scale;
                vals[27] = _tmem_load_6[27] * meta_scale;
                vals[28] = _tmem_load_6[28] * meta_scale;
                vals[29] = _tmem_load_6[29] * meta_scale;
                vals[30] = _tmem_load_6[30] * meta_scale;
                vals[31] = _tmem_load_6[31] * meta_scale;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                if (is_valid_row != 0) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + real_sub_5 * 64))), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_5 * 64 + 16)))), "r"(packed[4]), "r"(packed[5]), "r"(packed[6]), "r"(packed[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_5 * 64 + 32)))), "r"(packed[8]), "r"(packed[9]), "r"(packed[10]), "r"(packed[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_5 * 64 + 48)))), "r"(packed[12]), "r"(packed[13]), "r"(packed[14]), "r"(packed[15]) : "memory");
                }
                unsigned int real_sub_6 = 7 * acc_buf;
                float _tmem_load_7[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + acc_buf * 208 + real_sub_6 * 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                vals[0] = _tmem_load_7[0] * meta_scale;
                vals[1] = _tmem_load_7[1] * meta_scale;
                vals[2] = _tmem_load_7[2] * meta_scale;
                vals[3] = _tmem_load_7[3] * meta_scale;
                vals[4] = _tmem_load_7[4] * meta_scale;
                vals[5] = _tmem_load_7[5] * meta_scale;
                vals[6] = _tmem_load_7[6] * meta_scale;
                vals[7] = _tmem_load_7[7] * meta_scale;
                vals[8] = _tmem_load_7[8] * meta_scale;
                vals[9] = _tmem_load_7[9] * meta_scale;
                vals[10] = _tmem_load_7[10] * meta_scale;
                vals[11] = _tmem_load_7[11] * meta_scale;
                vals[12] = _tmem_load_7[12] * meta_scale;
                vals[13] = _tmem_load_7[13] * meta_scale;
                vals[14] = _tmem_load_7[14] * meta_scale;
                vals[15] = _tmem_load_7[15] * meta_scale;
                vals[16] = _tmem_load_7[16] * meta_scale;
                vals[17] = _tmem_load_7[17] * meta_scale;
                vals[18] = _tmem_load_7[18] * meta_scale;
                vals[19] = _tmem_load_7[19] * meta_scale;
                vals[20] = _tmem_load_7[20] * meta_scale;
                vals[21] = _tmem_load_7[21] * meta_scale;
                vals[22] = _tmem_load_7[22] * meta_scale;
                vals[23] = _tmem_load_7[23] * meta_scale;
                vals[24] = _tmem_load_7[24] * meta_scale;
                vals[25] = _tmem_load_7[25] * meta_scale;
                vals[26] = _tmem_load_7[26] * meta_scale;
                vals[27] = _tmem_load_7[27] * meta_scale;
                vals[28] = _tmem_load_7[28] * meta_scale;
                vals[29] = _tmem_load_7[29] * meta_scale;
                vals[30] = _tmem_load_7[30] * meta_scale;
                vals[31] = _tmem_load_7[31] * meta_scale;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                if (is_valid_row != 0) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + real_sub_6 * 64))), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_6 * 64 + 16)))), "r"(packed[4]), "r"(packed[5]), "r"(packed[6]), "r"(packed[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_6 * 64 + 32)))), "r"(packed[8]), "r"(packed[9]), "r"(packed[10]), "r"(packed[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sc_addr + ((unsigned int)(epi_tidx * 528) + (real_sub_6 * 64 + 48)))), "r"(packed[12]), "r"(packed[13]), "r"(packed[14]), "r"(packed[15]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                acc_buf = 1 - acc_buf;
                int reduce_row = epi_tidx;
                if (is_partial != 0) {
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    reduce_row = epi_tidx % 32 * 4 + epi_tidx / 32;
                }
                int reduce_ok = (int)(e_limit > row_base + reduce_row);
                int reduce_tok = smeta_tok[meta_stage * 128 + (unsigned int)reduce_row];
                if (reduce_ok != 0) {
                    if (valid_cols > 0) {
                        {
                            void* _cpred_dst_0 = reinterpret_cast<void*>(out + (reduce_tok * out_cols + col_base));
                            asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"
                                " [%0], [%1], %2;"
                                :: "l"(_cpred_dst_0), "r"(sc_addr + (unsigned int)(reduce_row * 528)), "r"((uint32_t)(valid_cols * 2))
                                : "memory");
                        }
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                mbarrier_arrive(meta_free_addr + (meta_stage) * 8);
                meta_stage += 1;
                if (meta_stage == 2) { meta_stage = 0; _phase_meta_full ^= 1; }
                mbarrier_wait(tile_full_addr + (tile_stage) * 8, _phase_tile_full);
                info[0] = sinfo[tile_stage * 5];
                info[1] = sinfo[tile_stage * 5 + 1];
                info[2] = sinfo[tile_stage * 5 + 2];
                info[3] = sinfo[tile_stage * 5 + 3];
                info[4] = sinfo[tile_stage * 5 + 4];
                e_coord_m = sinfo[tile_stage * 5];
                e_coord_n = sinfo[tile_stage * 5 + 1];
                e_limit = sinfo[tile_stage * 5 + 4];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage) * 8);
                tile_stage += 1;
                if (tile_stage == 2) { tile_stage = 0; _phase_tile_full ^= 1; }
            }
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int sa = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int tile_stage_1 = 0;
            unsigned int pha = 0;
            unsigned int ab_tok = 1;
            unsigned int acc_buf_1 = 0;
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
            unsigned int _phase_acc_free = 1;
            #pragma unroll 1
            for (int _tile_1 = 0; _tile_1 < num_m_tiles * num_n_tiles + 1; _tile_1++) {
                if (info_1[3] == 0) {
                    break;
                }
                int sfb_shift = 0;
                uint32_t _mbar_token_0 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                ab_tok = _mbar_token_0;
                mbarrier_wait(acc_free_addr + (acc_stage_1) * 8, _phase_acc_free);
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait_token(ab_full_addr + (sa) * 8, pha, ab_tok);
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa) * 32)));
                }
                if (elect_sync()) {
                    tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sa) * 64)));
                    tcgen05_cp_32x128b_warpx4((tmem_sf_b + 4), make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sa) * 64 + 32)));
                }
                int _mma_a_lo_0 = make_warp_uniform((((a_addr) >> 4) & 0x3FFF) + (sa) * 1024);
                int _mma_b_lo_0 = make_warp_uniform((((b_addr) >> 4) & 0x3FFF) + (sa) * 2048);
                {
                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                    tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_buf_1 * 208)), a_desc + 0, b_desc + 0,
                        0x8c01400U, tmem_sf_a, tmem_sf_b + sfb_shift, 0);
                    tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_buf_1 * 208)), a_desc + 2, b_desc + 2,
                        0x28c01410U, tmem_sf_a, tmem_sf_b + sfb_shift, 1);
                    tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_buf_1 * 208)), a_desc + 4, b_desc + 4,
                        0x48c01420U, tmem_sf_a, tmem_sf_b + sfb_shift, 1);
                    tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_buf_1 * 208)), a_desc + 6, b_desc + 6,
                        0x68c01430U, tmem_sf_a, tmem_sf_b + sfb_shift, 1);
                }
                elect_commit(ab_free_addr + (sa) * 8);
                sa += 1;
                if (sa == 3) { sa = 0; pha ^= 1; }
                ab_tok = 1;
                if (k_tiles > 1) {
                    uint32_t _mbar_token_1 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                    ab_tok = _mbar_token_1;
                }
                #pragma unroll 1
                for (int k = 1; k < k_tiles; k++) {
                    mbarrier_wait_token(ab_full_addr + (sa) * 8, pha, ab_tok);
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_a, make_sf_cp_desc_lo_sbo128((((sfa_addr) >> 4) + (sa) * 32)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_sf_b, make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sa) * 64)));
                        tcgen05_cp_32x128b_warpx4((tmem_sf_b + 4), make_sf_cp_desc_lo_sbo128((((sfb_addr) >> 4) + (sa) * 64 + 32)));
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((a_addr) >> 4) & 0x3FFF) + (sa) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform((((b_addr) >> 4) & 0x3FFF) + (sa) * 2048);
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_buf_1 * 208)), a_desc + 0, b_desc + 0,
                            0x8c01400U, tmem_sf_a, tmem_sf_b + sfb_shift, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_buf_1 * 208)), a_desc + 2, b_desc + 2,
                            0x28c01410U, tmem_sf_a, tmem_sf_b + sfb_shift, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_buf_1 * 208)), a_desc + 4, b_desc + 4,
                            0x48c01420U, tmem_sf_a, tmem_sf_b + sfb_shift, 1);
                        tcgen05_mma_mxf8_bs_elect((tmem_acc + (acc_buf_1 * 208)), a_desc + 6, b_desc + 6,
                            0x68c01430U, tmem_sf_a, tmem_sf_b + sfb_shift, 1);
                    }
                    elect_commit(ab_free_addr + (sa) * 8);
                    sa += 1;
                    if (sa == 3) { sa = 0; pha ^= 1; }
                    ab_tok = 1;
                    if (k + 1 < k_tiles) {
                        uint32_t _mbar_token_2 = mbarrier_try_wait(ab_full_addr + (sa) * 8, pha);
                        ab_tok = _mbar_token_2;
                    }
                }
                elect_commit(acc_full_addr + (acc_stage_1) * 8);
                _phase_acc_free ^= 1;
                acc_buf_1 = 1 - acc_buf_1;
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
    // ---- Role: tma ----
    if (warp == 5) {
        { // tma_main
            unsigned int stage = 0;
            unsigned int tile_stage_2 = 0;
            int rank_tma = 0;
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
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (int _tile_2 = 0; _tile_2 < num_m_tiles * num_n_tiles + 1; _tile_2++) {
                if (info_2[3] == 0) {
                    break;
                }
                int row_base_tma = info_2[0] * 128;
                int col_base_tma = info_2[1] * 256 + rank_tma * 256;
                int expert_tma = info_2[2];
                int sfb_atom = info_2[1] * 4 / 2;
                #pragma unroll 1
                for (int k_1 = 0; k_1 < k_tiles; k_1++) {
                    mbarrier_wait(ab_free_addr + (stage) * 8, _phase_ab_free);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(ab_full_addr + (stage) * 8, 34304);
                        tma_2d_gmem2smem(a_addr + stage * 16384, A, k_1 * 128, row_base_tma, ab_full_addr + (stage) * 8);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(b_addr + stage * 32768), "l"(B), "r"(k_1 * 128), "r"(col_base_tma), "r"(expert_tma),
                               "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        tma_4d_gmem2smem(sfa_addr + stage * 512, SFA, 0, 0, k_1, info_2[0], ab_full_addr + (stage) * 8);
                        asm volatile(
                            "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                            :: "r"(sfb_addr + stage * 1024), "l"(SFB), "r"(0), "r"(0), "r"(k_1), "r"(sfb_atom), "r"(expert_tma),
                               "r"(ab_full_addr + (stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    stage += 1;
                    if (stage == 3) { stage = 0; _phase_ab_free ^= 1; }
                }
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
    // ---- Role: scheduler ----
    if (warp == 6) {
        { // scheduler_main
            unsigned int tile_stage_3 = 0;
            int num_valid = num_non_exiting_tiles[0];
            int total_tiles = num_m_tiles * num_n_tiles;
            int first_item = bid;
            int item_stride = num_bids;
            int rank_sched = 0;
            unsigned int _phase_tile_free = 1;
            #pragma unroll 1
            for (int item = first_item; item < total_tiles; item += item_stride) {
                int m_tile = item / num_n_tiles;
                int n_tile = item - m_tile * num_n_tiles;
                if (m_tile >= num_valid) {
                    break;
                }
                mbarrier_wait(tile_free_addr + (tile_stage_3) * 8, _phase_tile_free);
                int group = m_tile;
                int expert = tile_idx_to_expert_idx[group];
                int mn_limit = tile_idx_to_mn_limit[group];
                if (elect_sync()) {
                    sinfo[tile_stage_3 * 5] = group + rank_sched;
                    sinfo[tile_stage_3 * 5 + 1] = n_tile;
                    sinfo[tile_stage_3 * 5 + 2] = expert;
                    sinfo[tile_stage_3 * 5 + 3] = 1;
                    sinfo[tile_stage_3 * 5 + 4] = mn_limit;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 4, 32;" ::: "memory");
                mbarrier_arrive(tile_full_addr + (tile_stage_3) * 8);
                tile_stage_3 += 1;
                if (tile_stage_3 == 2) { tile_stage_3 = 0; _phase_tile_free ^= 1; }
            }
            mbarrier_wait(tile_free_addr + (tile_stage_3) * 8, _phase_tile_free);
            if (elect_sync()) {
                sinfo[tile_stage_3 * 5] = 0;
                sinfo[tile_stage_3 * 5 + 1] = 0;
                sinfo[tile_stage_3 * 5 + 2] = -1;
                sinfo[tile_stage_3 * 5 + 3] = 0;
                sinfo[tile_stage_3 * 5 + 4] = 0;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 4, 32;" ::: "memory");
            mbarrier_arrive(tile_full_addr + (tile_stage_3) * 8);
            tile_stage_3 += 1;
            if (tile_stage_3 == 2) { tile_stage_3 = 0; _phase_tile_free ^= 1; }
        }
    }
    // ---- Role: meta ----
    if (warp == 7) {
        { // meta_main
            unsigned int tile_stage_4 = 0;
            unsigned int meta_stage_1 = 0;
            int info_3[5];
            unsigned int _phase_tile_full_3 = 0;
            mbarrier_wait(tile_full_addr + (tile_stage_4) * 8, _phase_tile_full_3);
            info_3[0] = sinfo[tile_stage_4 * 5];
            info_3[1] = sinfo[tile_stage_4 * 5 + 1];
            info_3[2] = sinfo[tile_stage_4 * 5 + 2];
            info_3[3] = sinfo[tile_stage_4 * 5 + 3];
            info_3[4] = sinfo[tile_stage_4 * 5 + 4];
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(tile_free_addr + (tile_stage_4) * 8);
            tile_stage_4 += 1;
            if (tile_stage_4 == 2) { tile_stage_4 = 0; _phase_tile_full_3 ^= 1; }
            int lane_meta = lane;
            unsigned int _phase_meta_free = 1;
            #pragma unroll 1
            for (int _tile_3 = 0; _tile_3 < num_m_tiles * num_n_tiles + 1; _tile_3++) {
                if (info_3[3] == 0) {
                    break;
                }
                mbarrier_wait(meta_free_addr + (meta_stage_1) * 8, _phase_meta_free);
                int expert_meta = info_3[2];
                float alpha_e = alpha[expert_meta];
                int row_base_meta = info_3[0] * 128;
                int mn_limit_meta = info_3[4];
                int meta_row = lane_meta;
                int meta_prow = row_base_meta + meta_row;
                int meta_valid = (int)(meta_prow < mn_limit_meta);
                int meta_expanded = permuted_idx_to_expanded_idx[meta_prow];
                int _max_0 = ((meta_expanded) > (0) ? (meta_expanded) : (0));
                int meta_safe = _max_0;
                int meta_token = meta_safe / top_k;
                int meta_topk = meta_safe - meta_token * top_k;
                int meta_gather = meta_token * meta_valid;
                float meta_weight = token_final_scales[meta_gather * top_k + meta_topk];
                smeta_scale[meta_stage_1 * 128 + (unsigned int)meta_row] = alpha_e * meta_weight;
                smeta_tok[meta_stage_1 * 128 + (unsigned int)meta_row] = meta_token;
                int meta_row_0 = lane_meta + 32;
                int meta_prow_1 = row_base_meta + meta_row_0;
                int meta_valid_2 = (int)(meta_prow_1 < mn_limit_meta);
                int meta_expanded_3 = permuted_idx_to_expanded_idx[meta_prow_1];
                int _max_1 = ((meta_expanded_3) > (0) ? (meta_expanded_3) : (0));
                int meta_safe_4 = _max_1;
                int meta_token_5 = meta_safe_4 / top_k;
                int meta_topk_6 = meta_safe_4 - meta_token_5 * top_k;
                int meta_gather_7 = meta_token_5 * meta_valid_2;
                float meta_weight_8 = token_final_scales[meta_gather_7 * top_k + meta_topk_6];
                smeta_scale[meta_stage_1 * 128 + (unsigned int)meta_row_0] = alpha_e * meta_weight_8;
                smeta_tok[meta_stage_1 * 128 + (unsigned int)meta_row_0] = meta_token_5;
                int meta_row_9 = lane_meta + 64;
                int meta_prow_10 = row_base_meta + meta_row_9;
                int meta_valid_11 = (int)(meta_prow_10 < mn_limit_meta);
                int meta_expanded_12 = permuted_idx_to_expanded_idx[meta_prow_10];
                int _max_2 = ((meta_expanded_12) > (0) ? (meta_expanded_12) : (0));
                int meta_safe_13 = _max_2;
                int meta_token_14 = meta_safe_13 / top_k;
                int meta_topk_15 = meta_safe_13 - meta_token_14 * top_k;
                int meta_gather_16 = meta_token_14 * meta_valid_11;
                float meta_weight_17 = token_final_scales[meta_gather_16 * top_k + meta_topk_15];
                smeta_scale[meta_stage_1 * 128 + (unsigned int)meta_row_9] = alpha_e * meta_weight_17;
                smeta_tok[meta_stage_1 * 128 + (unsigned int)meta_row_9] = meta_token_14;
                int meta_row_18 = lane_meta + 96;
                int meta_prow_19 = row_base_meta + meta_row_18;
                int meta_valid_20 = (int)(meta_prow_19 < mn_limit_meta);
                int meta_expanded_21 = permuted_idx_to_expanded_idx[meta_prow_19];
                int _max_3 = ((meta_expanded_21) > (0) ? (meta_expanded_21) : (0));
                int meta_safe_22 = _max_3;
                int meta_token_23 = meta_safe_22 / top_k;
                int meta_topk_24 = meta_safe_22 - meta_token_23 * top_k;
                int meta_gather_25 = meta_token_23 * meta_valid_20;
                float meta_weight_26 = token_final_scales[meta_gather_25 * top_k + meta_topk_24];
                smeta_scale[meta_stage_1 * 128 + (unsigned int)meta_row_18] = alpha_e * meta_weight_26;
                smeta_tok[meta_stage_1 * 128 + (unsigned int)meta_row_18] = meta_token_23;
                mbarrier_arrive(meta_full_addr + (meta_stage_1) * 8);
                meta_stage_1 += 1;
                if (meta_stage_1 == 2) { meta_stage_1 = 0; _phase_meta_free ^= 1; }
                mbarrier_wait(tile_full_addr + (tile_stage_4) * 8, _phase_tile_full_3);
                info_3[0] = sinfo[tile_stage_4 * 5];
                info_3[1] = sinfo[tile_stage_4 * 5 + 1];
                info_3[2] = sinfo[tile_stage_4 * 5 + 2];
                info_3[3] = sinfo[tile_stage_4 * 5 + 3];
                info_3[4] = sinfo[tile_stage_4 * 5 + 4];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(tile_free_addr + (tile_stage_4) * 8);
                tile_stage_4 += 1;
                if (tile_stage_4 == 2) { tile_stage_4 = 0; _phase_tile_full_3 ^= 1; }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }

    // Kernel epilogue ops
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
