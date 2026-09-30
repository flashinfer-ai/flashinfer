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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 4
#define NUM_CHUNK_PIPE_STAGES 2
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_A32_OFF 1024
#define SMEM_A32_STAGE_BYTES 16384
#define SMEM_A32_STRIDE 53248
#define SMEM_A1_OFF 17408
#define SMEM_A1_STAGE_BYTES 8192
#define SMEM_A1_STRIDE 53248
#define SMEM_A2_OFF 25600
#define SMEM_A2_STAGE_BYTES 8192
#define SMEM_A2_STRIDE 53248
#define SMEM_A3_OFF 33792
#define SMEM_A3_STAGE_BYTES 8192
#define SMEM_A3_STRIDE 53248
#define SMEM_B1_OFF 41984
#define SMEM_B1_STAGE_BYTES 4096
#define SMEM_B1_STRIDE 53248
#define SMEM_B2_OFF 46080
#define SMEM_B2_STAGE_BYTES 4096
#define SMEM_B2_STRIDE 53248
#define SMEM_B3_OFF 50176
#define SMEM_B3_STAGE_BYTES 4096
#define SMEM_B3_STRIDE 53248
#define SMEM_WORK_RESPONSE_OFF 214016
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 214144
#define THREADS 448

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


__device__ __forceinline__ void tcgen05_mma_f16_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
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

extern "C" {

__global__ __launch_bounds__(448) __cluster_dims__(2,1,1) void
kernel_cake_dense_projection_gemm_b2a0d33bf7580b7ddeaf(CakeTensorMap const* A, CakeTensorMap const* B, float* __restrict__ out32, int M, int N, int m_tiles, int n_tiles, int k_iters_split, int ldo, int out_l, int num_cluster_tiles)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define raw_full_addr (mbar_base + 0)
    #define parts_full_addr (mbar_base + 32)
    #define mma_done_addr (mbar_base + 64)
    #define chunk_full_addr (mbar_base + 96)
    #define chunk_empty_addr (mbar_base + 112)
    #define mainloop_done_addr (mbar_base + 128)
    #define epilogue_done_addr (mbar_base + 144)
    #define work_full_addr (mbar_base + 160)
    #define work_empty_addr (mbar_base + 192)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* a32 = reinterpret_cast<float*>(smem_raw + 1024);
    const int a32_addr = smem + 1024;
    __nv_bfloat16* a1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int a1_addr = smem + 17408;
    __nv_bfloat16* a2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int a2_addr = smem + 25600;
    __nv_bfloat16* a3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int a3_addr = smem + 33792;
    __nv_bfloat16* b1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int b1_addr = smem + 41984;
    __nv_bfloat16* b2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 46080);
    const int b2_addr = smem + 46080;
    __nv_bfloat16* b3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int b3_addr = smem + 50176;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 214016);
    const int work_response_addr = smem + 214016;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 28 barriers)
    // Mbarriers at smem_raw[0..224)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // raw_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // parts_full: 4 barriers, init_count=8
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            mbarrier_init(smem + 48, 8);
            mbarrier_init(smem + 56, 8);
            // mma_done: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'chunk_pipe' ---
            // chunk_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // chunk_empty: 2 barriers, init_count=16
            mbarrier_init(smem + 112, 16);
            mbarrier_init(smem + 120, 16);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // epilogue_done: 2 barriers, init_count=16
            mbarrier_init(smem + 144, 16);
            mbarrier_init(smem + 152, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // work_empty: 4 barriers, init_count=802
            mbarrier_init(smem + 192, 802);
            mbarrier_init(smem + 200, 802);
            mbarrier_init(smem + 208, 802);
            mbarrier_init(smem + 216, 802);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 0) {
        int _tmem_hold = smem + 224;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < num_cluster_tiles; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int pair = this_bid / 2;
                    int rank = this_bid % 2;
                    int pairs_per_l = m_tiles / 2 * n_tiles;
                    int l = pair / pairs_per_l;
                    int prem = pair - l * pairs_per_l;
                    int pair_m = prem / n_tiles;
                    int bid_n = prem % n_tiles;
                    int off_m = (pair_m * 2 + rank) * 128;
                    int off_n = bid_n * 128;
                    int b_col = off_n + cta_rank * 64;
                    int k_first = l * k_iters_split;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < k_iters_split; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k0 = (k_first + iter_k) * 32;
                        #pragma unroll
                        for (int r = 0; r < 4; r++) {
                            tma_3d_gmem2smem(a32_addr + load_stage * 53248 + (unsigned int)(r * 4096), A, k0, off_m + 32 * r, 0, raw_full_addr + (load_stage) * 8);
                        }
                        #pragma unroll
                        for (int p = 0; p < 2; p++) {
                            tma_3d_gmem2smem(b1_addr + load_stage * 53248 + (unsigned int)(p * 2048), B, b_col + 32 * p, k0, 0, raw_full_addr + (load_stage) * 8);
                            tma_3d_gmem2smem(b2_addr + load_stage * 53248 + (unsigned int)(p * 2048), B, b_col + 32 * p, k0, 1, raw_full_addr + (load_stage) * 8);
                            tma_3d_gmem2smem(b3_addr + load_stage * 53248 + (unsigned int)(p * 2048), B, b_col + 32 * p, k0, 2, raw_full_addr + (load_stage) * 8);
                        }
                        mbarrier_arrive_expect_tx(raw_full_addr + (load_stage) * 8, 28672);
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    uint32_t _clc_valid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_0)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                    this_bid = _clc_ctaid_0 + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int chunk_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_chunk_empty = 1;
            unsigned int _phase_parts_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    unsigned int small_buf = mma_epi_stage + 2;
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < k_iters_split; iter_k_1++) {
                        int kc = iter_k_1 % 2;
                        if (kc == 0) {
                            mbarrier_wait(chunk_empty_addr + (chunk_stage) * 8, _phase_chunk_empty);
                        }
                        mbarrier_wait(parts_full_addr + (mma_tma_stage) * 8, _phase_parts_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int big_init = ((kc == 0) ? 1 : 0);
                        int small_init = ((iter_k_1 == 0) ? 1 : 0);
                        int _mma_a_lo_0 = (((a1_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_0 = ((((b1_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_tma_stage) * 3328;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (chunk_stage * 128))), "r"(((big_init) ? 0 : 1)));
                        int _mma_a_lo_1 = (((a1_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_1 = ((((b2_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_tma_stage) * 3328;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accum + (small_buf * 128))), "r"(((small_init) ? 0 : 1)));
                        int _mma_a_lo_2 = (((a2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_2 = ((((b1_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_tma_stage) * 3328;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_accum + (small_buf * 128))), "r"(1));
                        int _mma_a_lo_3 = (((a1_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_3 = ((((b3_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_tma_stage) * 3328;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_accum + (small_buf * 128))), "r"(1));
                        int _mma_a_lo_4 = (((a2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_4 = ((((b2_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_tma_stage) * 3328;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_accum + (small_buf * 128))), "r"(1));
                        int _mma_a_lo_5 = (((a3_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_5 = ((((b1_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_tma_stage) * 3328;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_accum + (small_buf * 128))), "r"(1));
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_parts_full ^= 1; }
                        int chunk_end = ((kc == 1) ? 1 : 0);
                        if (iter_k_1 == k_iters_split - 1) {
                            chunk_end = 1;
                        }
                        if (chunk_end == 1) {
                            elect_commit_cg2_multicast(chunk_full_addr + (chunk_stage) * 8, (uint16_t)(3));
                            chunk_stage += 1;
                            if (chunk_stage == 2) { chunk_stage = 0; _phase_chunk_empty ^= 1; }
                        }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t _clc_valid_2 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_2)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_2 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_2)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_2 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int chunk_stage_1 = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_part = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            int lane_base = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(col_part * 64);
            unsigned int this_bid_1 = bid;
            unsigned int _phase_chunk_full = 0;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles; _tile_iter_2++) {
                int pair_1 = this_bid_1 / 2;
                int rank_1 = this_bid_1 % 2;
                int pairs_per_l_1 = m_tiles / 2 * n_tiles;
                int l_1 = pair_1 / pairs_per_l_1;
                int prem_1 = pair_1 - l_1 * pairs_per_l_1;
                int pair_m_1 = prem_1 / n_tiles;
                int bid_n_1 = prem_1 % n_tiles;
                int off_m_1 = (pair_m_1 * 2 + rank_1) * 128;
                int off_n_1 = bid_n_1 * 128;
                int num_chunks = (k_iters_split + 1) / 2;
                float run[64];
                #pragma unroll
                for (int j = 0; j < 64; j++) {
                    run[j] = 0.0f;
                }
                #pragma unroll 1
                for (int _c = 0; _c < num_chunks; _c++) {
                    mbarrier_wait(chunk_full_addr + (chunk_stage_1) * 8, _phase_chunk_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int big_addr = (unsigned int)lane_base + chunk_stage_1 * 128;
                    #pragma unroll
                    for (int piece = 0; piece < 2; piece++) {
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(big_addr + piece * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 32; j_1++) {
                            run[piece * 32 + j_1] = run[piece * 32 + j_1] + _tmem_load_0[j_1];
                        }
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((chunk_empty_addr + (chunk_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    chunk_stage_1 += 1;
                    if (chunk_stage_1 == 2) { chunk_stage_1 = 0; _phase_chunk_full ^= 1; }
                }
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int small_addr = (unsigned int)lane_base + (epi_stage + 2) * 128;
                #pragma unroll
                for (int piece_1 = 0; piece_1 < 2; piece_1++) {
                    float _tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(small_addr + piece_1 * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 32; j_2++) {
                        run[piece_1 * 32 + j_2] = run[piece_1 * 32 + j_2] + _tmem_load_1[j_2];
                    }
                }
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                int global_row = off_m_1 + local_row;
                int col_base = off_n_1 + col_part * 64;
                unsigned long long batch_off = (unsigned long long)l_1 * (unsigned long long)out_l;
                unsigned long long row_base = batch_off + (unsigned long long)global_row * (unsigned long long)ldo + (unsigned long long)col_base;
                if (global_row < M) {
                    #pragma unroll
                    for (int q = 0; q < 16; q++) {
                        if (col_base + 4 * q + 4 <= N) {
                            {
                                float4 _v4 = make_float4(run[4 * q + 0], run[4 * q + 1], run[4 * q + 2], run[4 * q + 3]);
                                *reinterpret_cast<float4*>(out32 + (row_base + (unsigned long long)(4 * q)) + 0) = _v4;
                            }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t _clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_3)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                this_bid_1 = _clc_ctaid_3 + (unsigned int)cta_rank;
            }
        }
    }
    // ---- Role: producer ----
    if (warp >= 10 && warp <= 13) {
        { // producer_main
            unsigned int p_stage = 0;
            unsigned int work_stage_3 = 0;
            const int tid_1 = (warp - 10) * 32 + lane;
            const int row = tid_1;
            const int src_off = tid_1 * 128;
            const int dst_off = 0;
            const int swz8 = row & 7;
            unsigned int this_bid_2 = bid;
            unsigned int _phase_raw_full = 0;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < num_cluster_tiles; _tile_iter_3++) {
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < k_iters_split; iter_k_2++) {
                    mbarrier_wait(raw_full_addr + (p_stage) * 8, _phase_raw_full);
                    unsigned int src_row = a32_addr + p_stage * 53248 + (unsigned int)src_off;
                    unsigned int base1 = a1_addr + p_stage * 53248 + (unsigned int)dst_off;
                    unsigned int base2 = a2_addr + p_stage * 53248 + (unsigned int)dst_off;
                    unsigned int base3 = a3_addr + p_stage * 53248 + (unsigned int)dst_off;
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        unsigned int w0[4];
                        unsigned int w1[4];
                        unsigned int w2[4];
                        unsigned int w3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w0[(0) + 3]))
                            : "r"(src_row + (unsigned int)((4 * h ^ swz8) * 16)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w1[(0) + 3]))
                            : "r"(src_row + (unsigned int)((4 * h + 1 ^ swz8) * 16)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w2[(0) + 3]))
                            : "r"(src_row + (unsigned int)((4 * h + 2 ^ swz8) * 16)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w3[(0) + 3]))
                            : "r"(src_row + (unsigned int)((4 * h + 3 ^ swz8) * 16)));
                        float hi[16];
                        float mid[16];
                        float lo[16];
                        #pragma unroll
                        for (int jj = 0; jj < 4; jj++) {
                            float v = 0.0f;
                            v = reinterpret_cast<float*>(&w0[jj])[0];
                            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(v);
                            float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                            float p1 = _cvt_f32_0;
                            float r1 = v - p1;
                            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(r1);
                            float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                            float p2 = _cvt_f32_1;
                            hi[jj] = p1;
                            mid[jj] = p2;
                            __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(r1 - p2);
                            float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                            lo[jj] = _cvt_f32_2;
                        }
                        #pragma unroll
                        for (int jj_1 = 0; jj_1 < 4; jj_1++) {
                            float v_1 = 0.0f;
                            v_1 = reinterpret_cast<float*>(&w1[jj_1])[0];
                            __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(v_1);
                            float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                            float p1_1 = _cvt_f32_3;
                            float r1_1 = v_1 - p1_1;
                            __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(r1_1);
                            float _cvt_f32_4 = __bfloat162float(_cvt_bf16_4);
                            float p2_1 = _cvt_f32_4;
                            hi[4 + jj_1] = p1_1;
                            mid[4 + jj_1] = p2_1;
                            __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(r1_1 - p2_1);
                            float _cvt_f32_5 = __bfloat162float(_cvt_bf16_5);
                            lo[4 + jj_1] = _cvt_f32_5;
                        }
                        #pragma unroll
                        for (int jj_2 = 0; jj_2 < 4; jj_2++) {
                            float v_2 = 0.0f;
                            v_2 = reinterpret_cast<float*>(&w2[jj_2])[0];
                            __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(v_2);
                            float _cvt_f32_6 = __bfloat162float(_cvt_bf16_6);
                            float p1_2 = _cvt_f32_6;
                            float r1_2 = v_2 - p1_2;
                            __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(r1_2);
                            float _cvt_f32_7 = __bfloat162float(_cvt_bf16_7);
                            float p2_2 = _cvt_f32_7;
                            hi[8 + jj_2] = p1_2;
                            mid[8 + jj_2] = p2_2;
                            __nv_bfloat16 _cvt_bf16_8 = __float2bfloat16(r1_2 - p2_2);
                            float _cvt_f32_8 = __bfloat162float(_cvt_bf16_8);
                            lo[8 + jj_2] = _cvt_f32_8;
                        }
                        #pragma unroll
                        for (int jj_3 = 0; jj_3 < 4; jj_3++) {
                            float v_3 = 0.0f;
                            v_3 = reinterpret_cast<float*>(&w3[jj_3])[0];
                            __nv_bfloat16 _cvt_bf16_9 = __float2bfloat16(v_3);
                            float _cvt_f32_9 = __bfloat162float(_cvt_bf16_9);
                            float p1_3 = _cvt_f32_9;
                            float r1_3 = v_3 - p1_3;
                            __nv_bfloat16 _cvt_bf16_10 = __float2bfloat16(r1_3);
                            float _cvt_f32_10 = __bfloat162float(_cvt_bf16_10);
                            float p2_3 = _cvt_f32_10;
                            hi[12 + jj_3] = p1_3;
                            mid[12 + jj_3] = p2_3;
                            __nv_bfloat16 _cvt_bf16_11 = __float2bfloat16(r1_3 - p2_3);
                            float _cvt_f32_11 = __bfloat162float(_cvt_bf16_11);
                            lo[12 + jj_3] = _cvt_f32_11;
                        }
                        uint32_t hi_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hi[_lp*2 + 0], hi[_lp*2+1 + 0]));
                            hi_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t mid_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mid[_lp*2 + 0], mid[_lp*2+1 + 0]));
                            mid_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        uint32_t lo_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(lo[_lp*2 + 0], lo[_lp*2+1 + 0]));
                            lo_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int q_1 = 0; q_1 < 2; q_1++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((base1 + (unsigned int)(row * 64 + (h * 32 + q_1 * 16) ^ (row * 64 + (h * 32 + q_1 * 16) >> 7 & 3) << 4))), "r"(hi_bf16[4 * q_1]), "r"(hi_bf16[4 * q_1 + 1]), "r"(hi_bf16[4 * q_1 + 2]), "r"(hi_bf16[4 * q_1 + 3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((base2 + (unsigned int)(row * 64 + (h * 32 + q_1 * 16) ^ (row * 64 + (h * 32 + q_1 * 16) >> 7 & 3) << 4))), "r"(mid_bf16[4 * q_1]), "r"(mid_bf16[4 * q_1 + 1]), "r"(mid_bf16[4 * q_1 + 2]), "r"(mid_bf16[4 * q_1 + 3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((base3 + (unsigned int)(row * 64 + (h * 32 + q_1 * 16) ^ (row * 64 + (h * 32 + q_1 * 16) >> 7 & 3) << 4))), "r"(lo_bf16[4 * q_1]), "r"(lo_bf16[4 * q_1 + 1]), "r"(lo_bf16[4 * q_1 + 2]), "r"(lo_bf16[4 * q_1 + 3]) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((parts_full_addr + (p_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    p_stage += 1;
                    if (p_stage == 4) { p_stage = 0; _phase_raw_full ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                uint32_t _clc_valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_1)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_1)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_3 * 8), "r"(0) : "memory");
                work_stage_3 += 1;
                if (work_stage_3 == 4) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                this_bid_2 = _clc_ctaid_1 + (unsigned int)cta_rank;
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
