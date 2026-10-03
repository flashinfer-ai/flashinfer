/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#define TMEM_NCOLS 48
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFW_OFFSET 32
#define TMEM_TMEM_SFX_OFFSET 40
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_RED_PIPE_STAGES 1
#define SMEM_SMEM_W_OFF 1024
#define SMEM_SMEM_W_STAGE_BYTES 32768
#define SMEM_SMEM_W_STRIDE 43008
#define SMEM_SMEM_SFW0_OFF 41984
#define SMEM_SMEM_SFW0_STAGE_BYTES 512
#define SMEM_SMEM_SFW0_STRIDE 43008
#define SMEM_SMEM_SFW1_OFF 42496
#define SMEM_SMEM_SFW1_STAGE_BYTES 512
#define SMEM_SMEM_SFW1_STRIDE 43008
#define SMEM_SMEM_X_OFF 33792
#define SMEM_SMEM_X_STAGE_BYTES 8192
#define SMEM_SMEM_X_STRIDE 43008
#define SMEM_SMEM_SFX_ALL_OFF 43008
#define SMEM_SMEM_SFX_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_SFX_ALL_STRIDE 43008
#define SMEM_SMEM_SFX0_OFF 43008
#define SMEM_SMEM_SFX0_STAGE_BYTES 512
#define SMEM_SMEM_SFX0_STRIDE 43008
#define SMEM_SMEM_SFX1_OFF 43520
#define SMEM_SMEM_SFX1_STAGE_BYTES 512
#define SMEM_SMEM_SFX1_STRIDE 43008
#define SMEM_SMEM_EPI_OFF 173056
#define SMEM_SMEM_EPI_STAGE_BYTES 16384
#define SMEM_SMEM_EPI_STRIDE 16384
#define SMEM_SMEM_RED_OFF 173056
#define SMEM_SMEM_RED_STAGE_BYTES 18432
#define SMEM_SMEM_RED_STRIDE 18432
#define SMEM_SMEM_STG_OFF 191488
#define SMEM_SMEM_STG_STAGE_BYTES 13824
#define SMEM_SMEM_STG_STRIDE 13824
#define SMEM_TOTAL 205312
#define THREADS 192

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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(4,1,1) void
kernel_cake_kimi_k3_fp8_projection_4ab68c9b9048a5d1dff9(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap SFW, const __grid_constant__ CUtensorMap SFX, __nv_bfloat16* __restrict__ out, float* __restrict__ partials, unsigned int* __restrict__ counters, int M, int n_tiles, int n_valid, int ldo, int num_k_iters, int sf_k_tiles, int split, int tok_per_cta, int total_work, int store_vec, __nv_bfloat16* __restrict__ x, int K, const __grid_constant__ CUtensorMap XB)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 32)
    #define mainloop_done_addr (mbar_base + 64)
    #define epilogue_done_addr (mbar_base + 72)
    #define red_full_addr (mbar_base + 80)
    #define peers_free_addr (mbar_base + 88)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_w = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w_addr = smem + 1024;
    uint8_t* smem_sfw0 = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int smem_sfw0_addr = smem + 41984;
    uint8_t* smem_sfw1 = reinterpret_cast<uint8_t*>(smem_raw + 42496);
    const int smem_sfw1_addr = smem + 42496;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_x_addr = smem + 33792;
    uint8_t* smem_sfx_all = reinterpret_cast<uint8_t*>(smem_raw + 43008);
    const int smem_sfx_all_addr = smem + 43008;
    uint8_t* smem_sfx0 = reinterpret_cast<uint8_t*>(smem_raw + 43008);
    const int smem_sfx0_addr = smem + 43008;
    uint8_t* smem_sfx1 = reinterpret_cast<uint8_t*>(smem_raw + 43520);
    const int smem_sfx1_addr = smem + 43520;
    uint8_t* smem_epi = reinterpret_cast<uint8_t*>(smem_raw + 173056);
    const int smem_epi_addr = smem + 173056;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 173056);
    const int smem_red_addr = smem + 173056;
    float* smem_stg = reinterpret_cast<float*>(smem_raw + 191488);
    const int smem_stg_addr = smem + 191488;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // mma_done: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // epilogue_done: 1 barriers, init_count=4
            mbarrier_init(smem + 72, 4);
            // --- pipeline 'red_pipe' ---
            // red_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // peers_free: 1 barriers, init_count=3
            mbarrier_init(smem + 88, 3);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 48 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 96);
    if (warp == 0) {
        int _tmem_hold = smem + 96;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfw = taddr + 32;
    const int tmem_tmem_sfx = taddr + 40;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                int w0 = bid;
                int ws = num_bids;
                int n_items = (total_work - bid + num_bids - 1) / num_bids;
                unsigned int load_stage = 0;
                #pragma unroll 1
                for (int it = 0; it < n_items; it++) {
                    int work = w0 + it * ws;
                    int tile = work / split;
                    int rank = work - tile * split;
                    int k_begin = num_k_iters * rank / split;
                    int k_end = num_k_iters * (rank + 1) / split;
                    int k_count = k_end - k_begin;
                    int n_tile = tile % n_tiles;
                    int m_tile = tile / n_tiles;
                    int w_tile0 = n_tile * (2 * num_k_iters);
                    int x_row = m_tile * 32;
                    int sfw_unit0 = n_tile / 2 * sf_k_tiles * 2 + n_tile % 2;
                    int sfx_unit0 = m_tile * sf_k_tiles;
                    #pragma unroll 1
                    for (int i = 0; i < k_count; i++) {
                        int iter_k = k_begin + i;
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k_group = iter_k * 2;
                        tma_3d_gmem2smem(smem_w_addr + load_stage * 43008, (&W), 0, 0, w_tile0 + k_group, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw0_addr + load_stage * 43008, (&SFW), 0, 0, sfw_unit0 + k_group * 2, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw1_addr + load_stage * 43008, (&SFW), 0, 0, sfw_unit0 + k_group * 2 + 2, tma_full_addr + (load_stage) * 8);
                        if (it == 0 && i == 0) {
                            asm volatile("griddepcontrol.wait;" ::: "memory");
                        }
                        tma_3d_gmem2smem(smem_x_addr + load_stage * 43008, (&X), 0, x_row, k_group, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfx_all_addr + load_stage * 43008, (&SFX), 0, 0, sfx_unit0 + k_group, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 43008);
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            int w0_m = bid;
            int ws_m = num_bids;
            int n_items_m = (total_work - bid + num_bids - 1) / num_bids;
            unsigned int mma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int xq_m = 0;
            unsigned int res_stage_m = 0;
            int m_prev_m = -1;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (int it_m = 0; it_m < n_items_m; it_m++) {
                int work_m = w0_m + it_m * ws_m;
                int tile_1 = work_m / split;
                int rank_1 = work_m - tile_1 * split;
                int k_begin_1 = num_k_iters * rank_1 / split;
                int k_end_1 = num_k_iters * (rank_1 + 1) / split;
                int k_count_1 = k_end_1 - k_begin_1;
                mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
                #pragma unroll 1
                for (int i_1 = 0; i_1 < k_count_1; i_1++) {
                    mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((i_1 == 0) ? 1 : 0);
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfw, make_sf_cp_desc_lo_sbo128((((smem_sfw0_addr) >> 4) + (mma_stage) * 2688)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfx, make_sf_cp_desc_lo_sbo128((((smem_sfx0_addr) >> 4) + (mma_stage) * 2688)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfw + 4, make_sf_cp_desc_lo_sbo128((((smem_sfw1_addr) >> 4) + (mma_stage) * 2688)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfx + 4, make_sf_cp_desc_lo_sbo128((((smem_sfx1_addr) >> 4) + (mma_stage) * 2688)));
                        int _mma_a_lo_0 = (((smem_w_addr) >> 4) & 0x3FFF) + (mma_stage) * 2688;
                        int _mma_b_lo_0 = (((smem_x_addr) >> 4) & 0x3FFF) + (mma_stage) * 2688;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 0, b_desc + 0,
                                0x8880000U, tmem_tmem_sfw, tmem_tmem_sfx, ((init_flag) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 2, b_desc + 2,
                                0x28880010U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 4, b_desc + 4,
                                0x48880020U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 6, b_desc + 6,
                                0x68880030U, tmem_tmem_sfw, tmem_tmem_sfx, 1);
                        }
                        int _mma_a_lo_1 = (((smem_w_addr + 16384) >> 4) & 0x3FFF) + (mma_stage) * 2688;
                        int _mma_b_lo_1 = (((smem_x_addr + 4096) >> 4) & 0x3FFF) + (mma_stage) * 2688;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 0, b_desc + 0,
                                0x8880000U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 2, b_desc + 2,
                                0x28880010U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 4, b_desc + 4,
                                0x48880020U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (acc_stage * 32)), a_desc + 6, b_desc + 6,
                                0x68880030U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1);
                        }
                    }
                    elect_commit(mma_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 4) { mma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(mainloop_done_addr + (acc_stage) * 8);
                _phase_epilogue_done ^= 1;
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int epi_group = (warp - 2) / 4;
            int tok_g0 = epi_group * 32;
            const int lane_row = epi_warp * 32 + lane;
            const int epi_tid = epi_warp * 32 + lane;
            int w0_e = bid;
            int ws_e = num_bids;
            int n_items_e = (total_work - bid + num_bids - 1) / num_bids;
            unsigned int acc_stage_e = 0;
            unsigned int red_stage = 0;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_peers_free = 1;
            unsigned int _phase_red_full = 0;
            #pragma unroll 1
            for (int it_e = 0; it_e < n_items_e; it_e++) {
                int work_e = w0_e + it_e * ws_e;
                int tile_2 = work_e / split;
                int rank_2 = work_e - tile_2 * split;
                int k_begin_2 = num_k_iters * rank_2 / split;
                int k_end_2 = num_k_iters * (rank_2 + 1) / split;
                int k_count_2 = k_end_2 - k_begin_2;
                int n_tile_e = tile_2 % n_tiles;
                int m_tile_e = tile_2 / n_tiles;
                int feature = n_tile_e * 128 + lane_row;
                int tok0 = m_tile_e * 32;
                unsigned long long col = (unsigned long long)feature;
                mbarrier_wait(mainloop_done_addr + (acc_stage_e) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_col_e = 0;
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    mbarrier_arrive(epilogue_done_addr + (acc_stage_e) * 8);
                }
                _phase_mainloop_done ^= 1;
                unsigned int epi_base = smem_epi_addr + (unsigned int)(epi_group * 16384);
                int my_rank = cta_rank;
                int _min_0 = ((32) < ((my_rank + 1) * 8) ? (32) : ((my_rank + 1) * 8));
                int rows_mine = _min_0 - my_rank * 8;
                unsigned int line_b = (unsigned int)(lane_row * 36);
                unsigned int slot_b = (unsigned int)(my_rank * 4608);
                mbarrier_wait_cluster_hint(peers_free_addr + (red_stage) * 8, _phase_peers_free, 10000000);
                if (tid == 64) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (epi_group == 0) {
                    if (0 == my_rank) {
                        {
                            uint32_t _addr_0 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_0), "f"(_tmem_load_0[0]) : "memory");
                        }
                        {
                            uint32_t _addr_1 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_1), "f"(_tmem_load_0[1]) : "memory");
                        }
                        {
                            uint32_t _addr_2 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_2), "f"(_tmem_load_0[2]) : "memory");
                        }
                        {
                            uint32_t _addr_3 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(_tmem_load_0[3]) : "memory");
                        }
                        {
                            uint32_t _addr_4 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_4), "f"(_tmem_load_0[4]) : "memory");
                        }
                        {
                            uint32_t _addr_5 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_5), "f"(_tmem_load_0[5]) : "memory");
                        }
                        {
                            uint32_t _addr_6 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(_tmem_load_0[6]) : "memory");
                        }
                        {
                            uint32_t _addr_7 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(_tmem_load_0[7]) : "memory");
                        }
                    } else {
                        int _max_0 = ((0) > (-my_rank) ? (0) : (-my_rank));
                        int _min_1 = ((1) < (_max_0) ? (1) : (_max_0));
                        int o_idx = -_min_1;
                        unsigned int stg_b = (unsigned int)(o_idx * 4608) + line_b;
                        {
                            uint32_t _addr_8 = static_cast<uint32_t>(smem_stg_addr + stg_b);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(_tmem_load_0[0]) : "memory");
                        }
                        {
                            uint32_t _addr_9 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(_tmem_load_0[1]) : "memory");
                        }
                        {
                            uint32_t _addr_10 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(_tmem_load_0[2]) : "memory");
                        }
                        {
                            uint32_t _addr_11 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(_tmem_load_0[3]) : "memory");
                        }
                        {
                            uint32_t _addr_12 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(_tmem_load_0[4]) : "memory");
                        }
                        {
                            uint32_t _addr_13 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(_tmem_load_0[5]) : "memory");
                        }
                        {
                            uint32_t _addr_14 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(_tmem_load_0[6]) : "memory");
                        }
                        {
                            uint32_t _addr_15 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(_tmem_load_0[7]) : "memory");
                        }
                    }
                    if (1 == my_rank) {
                        {
                            uint32_t _addr_16 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(_tmem_load_0[8]) : "memory");
                        }
                        {
                            uint32_t _addr_17 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(_tmem_load_0[9]) : "memory");
                        }
                        {
                            uint32_t _addr_18 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_18), "f"(_tmem_load_0[10]) : "memory");
                        }
                        {
                            uint32_t _addr_19 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(_tmem_load_0[11]) : "memory");
                        }
                        {
                            uint32_t _addr_20 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_20), "f"(_tmem_load_0[12]) : "memory");
                        }
                        {
                            uint32_t _addr_21 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_21), "f"(_tmem_load_0[13]) : "memory");
                        }
                        {
                            uint32_t _addr_22 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_22), "f"(_tmem_load_0[14]) : "memory");
                        }
                        {
                            uint32_t _addr_23 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(_tmem_load_0[15]) : "memory");
                        }
                    } else {
                        int _max_1 = ((0) > (1 - my_rank) ? (0) : (1 - my_rank));
                        int _min_2 = ((1) < (_max_1) ? (1) : (_max_1));
                        int o_idx_1 = 1 - _min_2;
                        unsigned int stg_b_1 = (unsigned int)(o_idx_1 * 4608) + line_b;
                        {
                            uint32_t _addr_24 = static_cast<uint32_t>(smem_stg_addr + stg_b_1);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_24), "f"(_tmem_load_0[8]) : "memory");
                        }
                        {
                            uint32_t _addr_25 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_25), "f"(_tmem_load_0[9]) : "memory");
                        }
                        {
                            uint32_t _addr_26 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_26), "f"(_tmem_load_0[10]) : "memory");
                        }
                        {
                            uint32_t _addr_27 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_27), "f"(_tmem_load_0[11]) : "memory");
                        }
                        {
                            uint32_t _addr_28 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_28), "f"(_tmem_load_0[12]) : "memory");
                        }
                        {
                            uint32_t _addr_29 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_29), "f"(_tmem_load_0[13]) : "memory");
                        }
                        {
                            uint32_t _addr_30 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_30), "f"(_tmem_load_0[14]) : "memory");
                        }
                        {
                            uint32_t _addr_31 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_31), "f"(_tmem_load_0[15]) : "memory");
                        }
                    }
                    if (2 == my_rank) {
                        {
                            uint32_t _addr_32 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_32), "f"(_tmem_load_0[16]) : "memory");
                        }
                        {
                            uint32_t _addr_33 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_33), "f"(_tmem_load_0[17]) : "memory");
                        }
                        {
                            uint32_t _addr_34 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_34), "f"(_tmem_load_0[18]) : "memory");
                        }
                        {
                            uint32_t _addr_35 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_35), "f"(_tmem_load_0[19]) : "memory");
                        }
                        {
                            uint32_t _addr_36 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_36), "f"(_tmem_load_0[20]) : "memory");
                        }
                        {
                            uint32_t _addr_37 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_37), "f"(_tmem_load_0[21]) : "memory");
                        }
                        {
                            uint32_t _addr_38 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_38), "f"(_tmem_load_0[22]) : "memory");
                        }
                        {
                            uint32_t _addr_39 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_39), "f"(_tmem_load_0[23]) : "memory");
                        }
                    } else {
                        int _max_2 = ((0) > (2 - my_rank) ? (0) : (2 - my_rank));
                        int _min_3 = ((1) < (_max_2) ? (1) : (_max_2));
                        int o_idx_2 = 2 - _min_3;
                        unsigned int stg_b_2 = (unsigned int)(o_idx_2 * 4608) + line_b;
                        {
                            uint32_t _addr_40 = static_cast<uint32_t>(smem_stg_addr + stg_b_2);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_40), "f"(_tmem_load_0[16]) : "memory");
                        }
                        {
                            uint32_t _addr_41 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_41), "f"(_tmem_load_0[17]) : "memory");
                        }
                        {
                            uint32_t _addr_42 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_42), "f"(_tmem_load_0[18]) : "memory");
                        }
                        {
                            uint32_t _addr_43 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_43), "f"(_tmem_load_0[19]) : "memory");
                        }
                        {
                            uint32_t _addr_44 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_44), "f"(_tmem_load_0[20]) : "memory");
                        }
                        {
                            uint32_t _addr_45 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_45), "f"(_tmem_load_0[21]) : "memory");
                        }
                        {
                            uint32_t _addr_46 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_46), "f"(_tmem_load_0[22]) : "memory");
                        }
                        {
                            uint32_t _addr_47 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_47), "f"(_tmem_load_0[23]) : "memory");
                        }
                    }
                    if (3 == my_rank) {
                        {
                            uint32_t _addr_48 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_48), "f"(_tmem_load_0[24]) : "memory");
                        }
                        {
                            uint32_t _addr_49 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_49), "f"(_tmem_load_0[25]) : "memory");
                        }
                        {
                            uint32_t _addr_50 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_50), "f"(_tmem_load_0[26]) : "memory");
                        }
                        {
                            uint32_t _addr_51 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_51), "f"(_tmem_load_0[27]) : "memory");
                        }
                        {
                            uint32_t _addr_52 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_52), "f"(_tmem_load_0[28]) : "memory");
                        }
                        {
                            uint32_t _addr_53 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_53), "f"(_tmem_load_0[29]) : "memory");
                        }
                        {
                            uint32_t _addr_54 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_54), "f"(_tmem_load_0[30]) : "memory");
                        }
                        {
                            uint32_t _addr_55 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_55), "f"(_tmem_load_0[31]) : "memory");
                        }
                    } else {
                        int _max_3 = ((0) > (3 - my_rank) ? (0) : (3 - my_rank));
                        int _min_4 = ((1) < (_max_3) ? (1) : (_max_3));
                        int o_idx_3 = 3 - _min_4;
                        unsigned int stg_b_3 = (unsigned int)(o_idx_3 * 4608) + line_b;
                        {
                            uint32_t _addr_56 = static_cast<uint32_t>(smem_stg_addr + stg_b_3);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_56), "f"(_tmem_load_0[24]) : "memory");
                        }
                        {
                            uint32_t _addr_57 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_57), "f"(_tmem_load_0[25]) : "memory");
                        }
                        {
                            uint32_t _addr_58 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_58), "f"(_tmem_load_0[26]) : "memory");
                        }
                        {
                            uint32_t _addr_59 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_59), "f"(_tmem_load_0[27]) : "memory");
                        }
                        {
                            uint32_t _addr_60 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_60), "f"(_tmem_load_0[28]) : "memory");
                        }
                        {
                            uint32_t _addr_61 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_61), "f"(_tmem_load_0[29]) : "memory");
                        }
                        {
                            uint32_t _addr_62 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_62), "f"(_tmem_load_0[30]) : "memory");
                        }
                        {
                            uint32_t _addr_63 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_63), "f"(_tmem_load_0[31]) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                int _min_5 = ((8) < (rows_mine) ? (8) : (rows_mine));
                int _max_4 = ((0) > (_min_5) ? (0) : (_min_5));
                int rows_round = _max_4;
                if (tid == 64) {
                    if (0 != my_rank) {
                        int _max_5 = ((0) > (-my_rank) ? (0) : (-my_rank));
                        int _min_6 = ((1) < (_max_5) ? (1) : (_max_5));
                        int o_idx_c = -_min_6;
                        uint32_t _mapa_0;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_0) : "r"(smem_red_addr + slot_b), "r"(0));
                        uint32_t _mapa_1;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_1) : "r"(red_full_addr), "r"(0));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_mapa_0), "r"(smem_stg_addr + (unsigned int)(o_idx_c * 4608)), "r"((uint32_t)(4608)), "r"(_mapa_1)
                            : "memory");
                    }
                    if (1 != my_rank) {
                        int _max_6 = ((0) > (1 - my_rank) ? (0) : (1 - my_rank));
                        int _min_7 = ((1) < (_max_6) ? (1) : (_max_6));
                        int o_idx_c_1 = 1 - _min_7;
                        uint32_t _mapa_2;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_2) : "r"(smem_red_addr + slot_b), "r"(1));
                        uint32_t _mapa_3;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_3) : "r"(red_full_addr), "r"(1));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_mapa_2), "r"(smem_stg_addr + (unsigned int)(o_idx_c_1 * 4608)), "r"((uint32_t)(4608)), "r"(_mapa_3)
                            : "memory");
                    }
                    if (2 != my_rank) {
                        int _max_7 = ((0) > (2 - my_rank) ? (0) : (2 - my_rank));
                        int _min_8 = ((1) < (_max_7) ? (1) : (_max_7));
                        int o_idx_c_2 = 2 - _min_8;
                        uint32_t _mapa_4;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_4) : "r"(smem_red_addr + slot_b), "r"(2));
                        uint32_t _mapa_5;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_5) : "r"(red_full_addr), "r"(2));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_mapa_4), "r"(smem_stg_addr + (unsigned int)(o_idx_c_2 * 4608)), "r"((uint32_t)(4608)), "r"(_mapa_5)
                            : "memory");
                    }
                    if (3 != my_rank) {
                        int _max_8 = ((0) > (3 - my_rank) ? (0) : (3 - my_rank));
                        int _min_9 = ((1) < (_max_8) ? (1) : (_max_8));
                        int o_idx_c_3 = 3 - _min_9;
                        uint32_t _mapa_6;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_6) : "r"(smem_red_addr + slot_b), "r"(3));
                        uint32_t _mapa_7;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_7) : "r"(red_full_addr), "r"(3));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_mapa_6), "r"(smem_stg_addr + (unsigned int)(o_idx_c_3 * 4608)), "r"((uint32_t)(4608)), "r"(_mapa_7)
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    int _min_10 = ((1) < (rows_round) ? (1) : (rows_round));
                    mbarrier_arrive_expect_tx(red_full_addr + (red_stage) * 8, _min_10 * 13824);
                }
                mbarrier_wait_cluster_hint(red_full_addr + (red_stage) * 8, _phase_red_full, 10000000);
                if (epi_group == 0) {
                    if (0 == my_rank) {
                        float total = 0.0f;
                        #pragma unroll
                        for (int src = 0; src < 4; src++) {
                            total = total + smem_red[src * 1152 + lane_row * 9];
                        }
                        int tok_c = tok0;
                        if (feature < n_valid && tok_c < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total);
                        }
                        float total_0 = 0.0f;
                        #pragma unroll
                        for (int src_1 = 0; src_1 < 4; src_1++) {
                            total_0 = total_0 + smem_red[src_1 * 1152 + lane_row * 9 + 1];
                        }
                        int tok_c_1 = tok0 + 1;
                        if (feature < n_valid && tok_c_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_1 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_0);
                        }
                        float total_2 = 0.0f;
                        #pragma unroll
                        for (int src_2 = 0; src_2 < 4; src_2++) {
                            total_2 = total_2 + smem_red[src_2 * 1152 + lane_row * 9 + 2];
                        }
                        int tok_c_3 = tok0 + 2;
                        if (feature < n_valid && tok_c_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_3 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_2);
                        }
                        float total_4 = 0.0f;
                        #pragma unroll
                        for (int src_3 = 0; src_3 < 4; src_3++) {
                            total_4 = total_4 + smem_red[src_3 * 1152 + lane_row * 9 + 3];
                        }
                        int tok_c_5 = tok0 + 3;
                        if (feature < n_valid && tok_c_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_5 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_4);
                        }
                        float total_6 = 0.0f;
                        #pragma unroll
                        for (int src_4 = 0; src_4 < 4; src_4++) {
                            total_6 = total_6 + smem_red[src_4 * 1152 + lane_row * 9 + 4];
                        }
                        int tok_c_7 = tok0 + 4;
                        if (feature < n_valid && tok_c_7 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_7 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_6);
                        }
                        float total_8 = 0.0f;
                        #pragma unroll
                        for (int src_5 = 0; src_5 < 4; src_5++) {
                            total_8 = total_8 + smem_red[src_5 * 1152 + lane_row * 9 + 5];
                        }
                        int tok_c_9 = tok0 + 5;
                        if (feature < n_valid && tok_c_9 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_9 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_8);
                        }
                        float total_10 = 0.0f;
                        #pragma unroll
                        for (int src_6 = 0; src_6 < 4; src_6++) {
                            total_10 = total_10 + smem_red[src_6 * 1152 + lane_row * 9 + 6];
                        }
                        int tok_c_11 = tok0 + 6;
                        if (feature < n_valid && tok_c_11 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_11 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_10);
                        }
                        float total_12 = 0.0f;
                        #pragma unroll
                        for (int src_7 = 0; src_7 < 4; src_7++) {
                            total_12 = total_12 + smem_red[src_7 * 1152 + lane_row * 9 + 7];
                        }
                        int tok_c_13 = tok0 + 7;
                        if (feature < n_valid && tok_c_13 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_13 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_12);
                        }
                    }
                    if (1 == my_rank) {
                        float total_1 = 0.0f;
                        #pragma unroll
                        for (int src_8 = 0; src_8 < 4; src_8++) {
                            total_1 = total_1 + smem_red[src_8 * 1152 + lane_row * 9];
                        }
                        int tok_c_2 = tok0 + 8;
                        if (feature < n_valid && tok_c_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_1);
                        }
                        float total_0_1 = 0.0f;
                        #pragma unroll
                        for (int src_9 = 0; src_9 < 4; src_9++) {
                            total_0_1 = total_0_1 + smem_red[src_9 * 1152 + lane_row * 9 + 1];
                        }
                        int tok_c_1_1 = tok0 + 9;
                        if (feature < n_valid && tok_c_1_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_1_1 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_0_1);
                        }
                        float total_2_1 = 0.0f;
                        #pragma unroll
                        for (int src_10 = 0; src_10 < 4; src_10++) {
                            total_2_1 = total_2_1 + smem_red[src_10 * 1152 + lane_row * 9 + 2];
                        }
                        int tok_c_3_1 = tok0 + 10;
                        if (feature < n_valid && tok_c_3_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_3_1 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_2_1);
                        }
                        float total_4_1 = 0.0f;
                        #pragma unroll
                        for (int src_11 = 0; src_11 < 4; src_11++) {
                            total_4_1 = total_4_1 + smem_red[src_11 * 1152 + lane_row * 9 + 3];
                        }
                        int tok_c_5_1 = tok0 + 11;
                        if (feature < n_valid && tok_c_5_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_5_1 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_4_1);
                        }
                        float total_6_1 = 0.0f;
                        #pragma unroll
                        for (int src_12 = 0; src_12 < 4; src_12++) {
                            total_6_1 = total_6_1 + smem_red[src_12 * 1152 + lane_row * 9 + 4];
                        }
                        int tok_c_7_1 = tok0 + 12;
                        if (feature < n_valid && tok_c_7_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_7_1 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_6_1);
                        }
                        float total_8_1 = 0.0f;
                        #pragma unroll
                        for (int src_13 = 0; src_13 < 4; src_13++) {
                            total_8_1 = total_8_1 + smem_red[src_13 * 1152 + lane_row * 9 + 5];
                        }
                        int tok_c_9_1 = tok0 + 13;
                        if (feature < n_valid && tok_c_9_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_9_1 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_8_1);
                        }
                        float total_10_1 = 0.0f;
                        #pragma unroll
                        for (int src_14 = 0; src_14 < 4; src_14++) {
                            total_10_1 = total_10_1 + smem_red[src_14 * 1152 + lane_row * 9 + 6];
                        }
                        int tok_c_11_1 = tok0 + 14;
                        if (feature < n_valid && tok_c_11_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_11_1 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_10_1);
                        }
                        float total_12_1 = 0.0f;
                        #pragma unroll
                        for (int src_15 = 0; src_15 < 4; src_15++) {
                            total_12_1 = total_12_1 + smem_red[src_15 * 1152 + lane_row * 9 + 7];
                        }
                        int tok_c_13_1 = tok0 + 15;
                        if (feature < n_valid && tok_c_13_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_13_1 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_12_1);
                        }
                    }
                    if (2 == my_rank) {
                        float total_3 = 0.0f;
                        #pragma unroll
                        for (int src_16 = 0; src_16 < 4; src_16++) {
                            total_3 = total_3 + smem_red[src_16 * 1152 + lane_row * 9];
                        }
                        int tok_c_4 = tok0 + 16;
                        if (feature < n_valid && tok_c_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_3);
                        }
                        float total_0_2 = 0.0f;
                        #pragma unroll
                        for (int src_17 = 0; src_17 < 4; src_17++) {
                            total_0_2 = total_0_2 + smem_red[src_17 * 1152 + lane_row * 9 + 1];
                        }
                        int tok_c_1_2 = tok0 + 17;
                        if (feature < n_valid && tok_c_1_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_1_2 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_0_2);
                        }
                        float total_2_2 = 0.0f;
                        #pragma unroll
                        for (int src_18 = 0; src_18 < 4; src_18++) {
                            total_2_2 = total_2_2 + smem_red[src_18 * 1152 + lane_row * 9 + 2];
                        }
                        int tok_c_3_2 = tok0 + 18;
                        if (feature < n_valid && tok_c_3_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_3_2 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_2_2);
                        }
                        float total_4_2 = 0.0f;
                        #pragma unroll
                        for (int src_19 = 0; src_19 < 4; src_19++) {
                            total_4_2 = total_4_2 + smem_red[src_19 * 1152 + lane_row * 9 + 3];
                        }
                        int tok_c_5_2 = tok0 + 19;
                        if (feature < n_valid && tok_c_5_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_5_2 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_4_2);
                        }
                        float total_6_2 = 0.0f;
                        #pragma unroll
                        for (int src_20 = 0; src_20 < 4; src_20++) {
                            total_6_2 = total_6_2 + smem_red[src_20 * 1152 + lane_row * 9 + 4];
                        }
                        int tok_c_7_2 = tok0 + 20;
                        if (feature < n_valid && tok_c_7_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_7_2 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_6_2);
                        }
                        float total_8_2 = 0.0f;
                        #pragma unroll
                        for (int src_21 = 0; src_21 < 4; src_21++) {
                            total_8_2 = total_8_2 + smem_red[src_21 * 1152 + lane_row * 9 + 5];
                        }
                        int tok_c_9_2 = tok0 + 21;
                        if (feature < n_valid && tok_c_9_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_9_2 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_8_2);
                        }
                        float total_10_2 = 0.0f;
                        #pragma unroll
                        for (int src_22 = 0; src_22 < 4; src_22++) {
                            total_10_2 = total_10_2 + smem_red[src_22 * 1152 + lane_row * 9 + 6];
                        }
                        int tok_c_11_2 = tok0 + 22;
                        if (feature < n_valid && tok_c_11_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_11_2 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_10_2);
                        }
                        float total_12_2 = 0.0f;
                        #pragma unroll
                        for (int src_23 = 0; src_23 < 4; src_23++) {
                            total_12_2 = total_12_2 + smem_red[src_23 * 1152 + lane_row * 9 + 7];
                        }
                        int tok_c_13_2 = tok0 + 23;
                        if (feature < n_valid && tok_c_13_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_13_2 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_12_2);
                        }
                    }
                    if (3 == my_rank) {
                        float total_5 = 0.0f;
                        #pragma unroll
                        for (int src_24 = 0; src_24 < 4; src_24++) {
                            total_5 = total_5 + smem_red[src_24 * 1152 + lane_row * 9];
                        }
                        int tok_c_6 = tok0 + 24;
                        if (feature < n_valid && tok_c_6 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_5);
                        }
                        float total_0_3 = 0.0f;
                        #pragma unroll
                        for (int src_25 = 0; src_25 < 4; src_25++) {
                            total_0_3 = total_0_3 + smem_red[src_25 * 1152 + lane_row * 9 + 1];
                        }
                        int tok_c_1_3 = tok0 + 25;
                        if (feature < n_valid && tok_c_1_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_1_3 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_0_3);
                        }
                        float total_2_3 = 0.0f;
                        #pragma unroll
                        for (int src_26 = 0; src_26 < 4; src_26++) {
                            total_2_3 = total_2_3 + smem_red[src_26 * 1152 + lane_row * 9 + 2];
                        }
                        int tok_c_3_3 = tok0 + 26;
                        if (feature < n_valid && tok_c_3_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_3_3 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_2_3);
                        }
                        float total_4_3 = 0.0f;
                        #pragma unroll
                        for (int src_27 = 0; src_27 < 4; src_27++) {
                            total_4_3 = total_4_3 + smem_red[src_27 * 1152 + lane_row * 9 + 3];
                        }
                        int tok_c_5_3 = tok0 + 27;
                        if (feature < n_valid && tok_c_5_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_5_3 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_4_3);
                        }
                        float total_6_3 = 0.0f;
                        #pragma unroll
                        for (int src_28 = 0; src_28 < 4; src_28++) {
                            total_6_3 = total_6_3 + smem_red[src_28 * 1152 + lane_row * 9 + 4];
                        }
                        int tok_c_7_3 = tok0 + 28;
                        if (feature < n_valid && tok_c_7_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_7_3 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_6_3);
                        }
                        float total_8_3 = 0.0f;
                        #pragma unroll
                        for (int src_29 = 0; src_29 < 4; src_29++) {
                            total_8_3 = total_8_3 + smem_red[src_29 * 1152 + lane_row * 9 + 5];
                        }
                        int tok_c_9_3 = tok0 + 29;
                        if (feature < n_valid && tok_c_9_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_9_3 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_8_3);
                        }
                        float total_10_3 = 0.0f;
                        #pragma unroll
                        for (int src_30 = 0; src_30 < 4; src_30++) {
                            total_10_3 = total_10_3 + smem_red[src_30 * 1152 + lane_row * 9 + 6];
                        }
                        int tok_c_11_3 = tok0 + 30;
                        if (feature < n_valid && tok_c_11_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_11_3 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_10_3);
                        }
                        float total_12_3 = 0.0f;
                        #pragma unroll
                        for (int src_31 = 0; src_31 < 4; src_31++) {
                            total_12_3 = total_12_3 + smem_red[src_31 * 1152 + lane_row * 9 + 7];
                        }
                        int tok_c_13_3 = tok0 + 31;
                        if (feature < n_valid && tok_c_13_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_13_3 * (unsigned long long)ldo + col)) + (0)) = __float2bfloat16_rn(total_12_3);
                        }
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (n_items_e > it_e + 1) {
                    if (tid == 64) {
                        #pragma unroll
                        for (int p = 0; p < 4; p++) {
                            if (p != my_rank) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .b32 remAddr32;\n\t"
                                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                    "}"
                                    :: "r"(peers_free_addr), "r"(p) : "memory");
                            }
                        }
                    }
                }
                _phase_peers_free ^= 1;
                _phase_red_full ^= 1;
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
    }
}

} // extern "C"
