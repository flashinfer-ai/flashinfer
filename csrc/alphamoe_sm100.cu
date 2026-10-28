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

// Generated device source with the retained TVM-FFI semantic wrapper.
// Generated helpers and kernel body are embedded verbatim; one prelude
// diagnostic uses public-neutral wording.
// Host adaptation captures launch metadata, isolates generated macros,
// and selects the output-column split without changing the public API.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>

#include "tvm_ffi_utils.h"

// clang-format off

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
struct __align__(64) LoomTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(LoomTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(LoomTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(LoomTensorMap) >= alignof(CUtensorMap), "LoomTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace w8a8_i2 {
#define kernel_alpha_moe_w8a8_up_down kernel_alpha_moe_w8a8_up_down_w8a8_i2
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
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else

#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 80
#define TMEM_UP_ACC_OFFSET 0
#define TMEM_DOWN_ACC_OFFSET 64
#define NUM_UP_PIPE_STAGES 4
#define NUM_DOWN_PIPE_STAGES 2
#define SMEM_SMEM_W1_OFF 1024
#define SMEM_SMEM_W1_STAGE_BYTES 8192
#define SMEM_SMEM_W1_STRIDE 18432
#define SMEM_SMEM_X_OFF 17408
#define SMEM_SMEM_X_STAGE_BYTES 1024
#define SMEM_SMEM_X_STRIDE 18432
#define SMEM_SMEM_W1B_OFF 9216
#define SMEM_SMEM_W1B_STAGE_BYTES 8192
#define SMEM_SMEM_W1B_STRIDE 18432
#define SMEM_SMEM_XB_OFF 18432
#define SMEM_SMEM_XB_STAGE_BYTES 1024
#define SMEM_SMEM_XB_STRIDE 18432
#define SMEM_SMEM_W2_OFF 74752
#define SMEM_SMEM_W2_STAGE_BYTES 16384
#define SMEM_SMEM_W2_STRIDE 16384
#define SMEM_SMEM_ACT_OFF 107520
#define SMEM_SMEM_ACT_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_STRIDE 1024
#define SMEM_SMEM_AMAX_OFF 108544
#define SMEM_SMEM_AMAX_STAGE_BYTES 32
#define SMEM_SMEM_AMAX_STRIDE 32
#define SMEM_SMEM_ACT_SCALE_OFF 108576
#define SMEM_SMEM_ACT_SCALE_STAGE_BYTES 32
#define SMEM_SMEM_ACT_SCALE_STRIDE 32
#define SMEM_SMEM_PEER_AMAX_OFF 108608
#define SMEM_SMEM_PEER_AMAX_STAGE_BYTES 128
#define SMEM_SMEM_PEER_AMAX_STRIDE 128
#define SMEM_SMEM_OUT_OFF 108736
#define SMEM_SMEM_OUT_STAGE_BYTES 4096
#define SMEM_SMEM_OUT_STRIDE 4096
#define SMEM_TOTAL 112896
#define THREADS 192
#define LAUNCH_MIN_BLOCKS 2

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

__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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

__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
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

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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
    // Non-multicast, single-CTA gather4 in the .shared::cta window (the
    // cuda::ptx form): the destination and the barrier are the executing
    // CTA's. A CTA pair's gather takes the .shared::cluster window
    // (tma_gather4_gmem2smem_cta2), its barrier being the pair leader's.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) __cluster_dims__(1,1,4) void
kernel_alpha_moe_w8a8_up_down(const __grid_constant__ CUtensorMap x, const __grid_constant__ CUtensorMap W1, float* __restrict__ x_scale, float* __restrict__ w1_scale, const __grid_constant__ CUtensorMap W2, float* __restrict__ w2_scale, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, float* __restrict__ topk_weights, __nv_bfloat16* __restrict__ out, int M, int K, int top_k, int route_block_m, float scaling_factor)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define up_full_addr (mbar_base + 0)
    #define up_ready_addr (mbar_base + 32)
    #define up_free_addr (mbar_base + 64)
    #define down_full_addr (mbar_base + 96)
    #define down_ready_addr (mbar_base + 112)
    #define down_free_addr (mbar_base + 128)
    #define amax_ready_addr (mbar_base + 144)
    #define act_ready_addr (mbar_base + 152)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = (((blockIdx.z / 4) * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * gridDim.y * (gridDim.z / 4);

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_w1 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w1_addr = smem + 1024;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_x_addr = smem + 17408;
    uint8_t* smem_w1b = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_w1b_addr = smem + 9216;
    uint8_t* smem_xb = reinterpret_cast<uint8_t*>(smem_raw + 18432);
    const int smem_xb_addr = smem + 18432;
    uint8_t* smem_w2 = reinterpret_cast<uint8_t*>(smem_raw + 74752);
    const int smem_w2_addr = smem + 74752;
    uint8_t* smem_act = reinterpret_cast<uint8_t*>(smem_raw + 107520);
    const int smem_act_addr = smem + 107520;
    unsigned int* smem_amax = reinterpret_cast<unsigned int*>(smem_raw + 108544);
    const int smem_amax_addr = smem + 108544;
    float* smem_act_scale = reinterpret_cast<float*>(smem_raw + 108576);
    const int smem_act_scale_addr = smem + 108576;
    unsigned int* smem_peer_amax = reinterpret_cast<unsigned int*>(smem_raw + 108608);
    const int smem_peer_amax_addr = smem + 108608;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 108736);
    const int smem_out_addr = smem + 108736;
    if (blockIdx.x * 8 >= num_tokens_post_padded[0]) {
        return;
    }

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'up_pipe' ---
            // up_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // up_ready: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // up_free: 4 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            // --- pipeline 'down_pipe' ---
            // down_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // down_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // down_free: 2 barriers, init_count=4
            mbarrier_init(smem + 128, 4);
            mbarrier_init(smem + 136, 4);
            // amax_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // act_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 144, 96);
            mbarrier_expect_tx(smem + 152, 768);
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 80 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_up_acc = taddr;
    const int tmem_down_acc = taddr + 64;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int _phase_up_free = 1;
            unsigned int _phase_down_free = 1;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int route_work = blockIdx.x;
                int intermediate_block = blockIdx.y;
                int subtiles_per_route_block = route_block_m / 8;
                int route_block = route_work / subtiles_per_route_block;
                int route_subtile = route_work % subtiles_per_route_block;
                int route_base = route_block * route_block_m + route_subtile * 8;
                bool route_active = route_base < num_tokens_post_padded[0];
                int expert = 0;
                if (route_active) {
                    expert = expert_ids[route_block];
                }
                int k_blocks = K / 128;
                const int panel = cta_rank;
                int pair0 = M * top_k;
                int pair1 = M * top_k;
                int pair2 = M * top_k;
                int pair3 = M * top_k;
                int pair4 = M * top_k;
                int pair5 = M * top_k;
                int pair6 = M * top_k;
                int pair7 = M * top_k;
                if (route_active) {
                    pair0 = sorted_token_ids[route_base];
                    pair1 = sorted_token_ids[route_base + 1];
                    pair2 = sorted_token_ids[route_base + 2];
                    pair3 = sorted_token_ids[route_base + 3];
                    pair4 = sorted_token_ids[route_base + 4];
                    pair5 = sorted_token_ids[route_base + 5];
                    pair6 = sorted_token_ids[route_base + 6];
                    pair7 = sorted_token_ids[route_base + 7];
                }
                int row0 = ((pair0 / top_k) < (M - 1) ? (pair0 / top_k) : (M - 1));
                int row1 = ((pair1 / top_k) < (M - 1) ? (pair1 / top_k) : (M - 1));
                int row2 = ((pair2 / top_k) < (M - 1) ? (pair2 / top_k) : (M - 1));
                int row3 = ((pair3 / top_k) < (M - 1) ? (pair3 / top_k) : (M - 1));
                int row4 = ((pair4 / top_k) < (M - 1) ? (pair4 / top_k) : (M - 1));
                int row5 = ((pair5 / top_k) < (M - 1) ? (pair5 / top_k) : (M - 1));
                int row6 = ((pair6 / top_k) < (M - 1) ? (pair6 / top_k) : (M - 1));
                int row7 = ((pair7 / top_k) < (M - 1) ? (pair7 / top_k) : (M - 1));
                unsigned int stage = 0;
                const int pf_rank = blockIdx.z;
                const int pf_ranks = gridDim.z;
                bool l2_on = 1;
                l2_on = num_tokens_post_padded[0] <= 192;
                if (l2_on) {
                    if (elect_sync()) {
                        #pragma unroll 1
                        for (int pf0 = 8; pf0 < 16; pf0++) {
                            if (k_blocks > pf0) {
                                asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&W1))), "r"((int)(0)), "r"((int)(intermediate_block * 256 + panel * 64)), "r"((int)(pf0)), "r"((int)(expert)) : "memory");
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int ks = 0; ks < (k_blocks + 1) / 2; ks++) {
                        mbarrier_wait(up_free_addr + (stage) * 8, _phase_up_free);
                        if (elect_sync()) {
                            int kb_second = ((ks * 2 + 1) < (k_blocks - 1) ? (ks * 2 + 1) : (k_blocks - 1));
                            mbarrier_arrive_expect_tx(up_full_addr + (stage) * 8, 18432);
                            tma_gather4_gmem2smem(smem_x_addr + stage * 18432, (&x), ks * 2 * 128, row0, row1, row2, row3, up_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(smem_x_addr + stage * 18432 + 512, (&x), ks * 2 * 128, row4, row5, row6, row7, up_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(smem_xb_addr + stage * 18432, (&x), kb_second * 128, row0, row1, row2, row3, up_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(smem_xb_addr + stage * 18432 + 512, (&x), kb_second * 128, row4, row5, row6, row7, up_full_addr + (stage) * 8);
                            tma_4d_gmem2smem(smem_w1_addr + stage * 18432, (&W1), 0, intermediate_block * 256 + panel * 64, ks * 2, expert, up_full_addr + (stage) * 8);
                            tma_4d_gmem2smem(smem_w1b_addr + stage * 18432, (&W1), 0, intermediate_block * 256 + panel * 64, kb_second, expert, up_full_addr + (stage) * 8);
                            int pf_kb = ks * 2 + 16;
                            if (pf_kb < k_blocks) {
                                asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&W1))), "r"((int)(0)), "r"((int)(intermediate_block * 256 + panel * 64)), "r"((int)(pf_kb)), "r"((int)(expert)) : "memory");
                            }
                            if (k_blocks > pf_kb + 1) {
                                asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&W1))), "r"((int)(0)), "r"((int)(intermediate_block * 256 + panel * 64)), "r"((int)(pf_kb + 1)), "r"((int)(expert)) : "memory");
                            }
                            int pf_ob = pf_rank + ks * pf_ranks;
                            if (pf_ob < K / 128) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W2))), "r"((int)(intermediate_block * 128)), "r"((int)(pf_ob * 128)), "r"((int)(expert)) : "memory");
                            }
                        }
                        stage += 1;
                        if (stage == 4) { stage = 0; _phase_up_free ^= 1; }
                    }
                } else {
                    #pragma unroll 1
                    for (int ks_1 = 0; ks_1 < (k_blocks + 1) / 2; ks_1++) {
                        mbarrier_wait(up_free_addr + (stage) * 8, _phase_up_free);
                        if (elect_sync()) {
                            int kb_second_1 = ((ks_1 * 2 + 1) < (k_blocks - 1) ? (ks_1 * 2 + 1) : (k_blocks - 1));
                            mbarrier_arrive_expect_tx(up_full_addr + (stage) * 8, 18432);
                            tma_gather4_gmem2smem(smem_x_addr + stage * 18432, (&x), ks_1 * 2 * 128, row0, row1, row2, row3, up_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(smem_x_addr + stage * 18432 + 512, (&x), ks_1 * 2 * 128, row4, row5, row6, row7, up_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(smem_xb_addr + stage * 18432, (&x), kb_second_1 * 128, row0, row1, row2, row3, up_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(smem_xb_addr + stage * 18432 + 512, (&x), kb_second_1 * 128, row4, row5, row6, row7, up_full_addr + (stage) * 8);
                            tma_4d_gmem2smem(smem_w1_addr + stage * 18432, (&W1), 0, intermediate_block * 256 + panel * 64, ks_1 * 2, expert, up_full_addr + (stage) * 8);
                            tma_4d_gmem2smem(smem_w1b_addr + stage * 18432, (&W1), 0, intermediate_block * 256 + panel * 64, kb_second_1, expert, up_full_addr + (stage) * 8);
                        }
                        stage += 1;
                        if (stage == 4) { stage = 0; _phase_up_free ^= 1; }
                    }
                }
                unsigned int down_load_stage = 0;
                int out_blocks = K / 128;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int ob = blockIdx.z; ob < out_blocks; ob += gridDim.z) {
                        mbarrier_wait(down_free_addr + (down_load_stage) * 8, _phase_down_free);
                        mbarrier_arrive_expect_tx(down_full_addr + (down_load_stage) * 8, 16384);
                        tma_3d_gmem2smem(smem_w2_addr + down_load_stage * 16384, (&W2), intermediate_block * 128, ob * 128, expert, down_full_addr + (down_load_stage) * 8);
                        down_load_stage += 1;
                        if (down_load_stage == 2) { down_load_stage = 0; _phase_down_free ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int _phase_up_free_1 = 1;
            unsigned int _phase_up_full = 0;
            unsigned int _phase_down_free_1 = 1;
            unsigned int _phase_down_full = 0;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int k_blocks_mma = K / 128;
                unsigned int up_stage = 0;
                #pragma unroll 1
                for (int ks_mma = 0; ks_mma < (k_blocks_mma + 1) / 2; ks_mma++) {
                    mbarrier_wait(up_free_addr + (up_stage) * 8, _phase_up_free_1);
                    mbarrier_wait(up_full_addr + (up_stage) * 8, _phase_up_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_w1_addr) >> 4) & 0x3FFF) + (up_stage) * 1152);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (up_stage) * 1152);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67239952, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67239952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67239952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67239952, 1);
                        }
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((smem_w1b_addr) >> 4) & 0x3FFF) + (up_stage) * 1152);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_xb_addr) >> 4) & 0x3FFF) + (up_stage) * 1152);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67239952, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67239952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67239952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67239952, 1);
                        }
                    }
                    elect_commit(up_ready_addr + (up_stage) * 8);
                    up_stage += 1;
                    if (up_stage == 4) { up_stage = 0; _phase_up_free_1 ^= 1; _phase_up_full ^= 1; }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 160;" ::: "memory");
                mbarrier_wait_cluster_hint(act_ready_addr, 0, 10000000);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int out_blocks_mma = K / 128;
                unsigned int down_stage = 0;
                #pragma unroll 1
                for (int ob_mma = blockIdx.z; ob_mma < out_blocks_mma; ob_mma += gridDim.z) {
                    mbarrier_wait(down_free_addr + (down_stage) * 8, _phase_down_free_1);
                    mbarrier_wait(down_full_addr + (down_stage) * 8, _phase_down_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_2 = make_warp_uniform((((smem_w2_addr) >> 4) & 0x3FFF) + (down_stage) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform(((smem_act_addr) >> 4) & 0x3FFF);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                        }
                    }
                    elect_commit(down_ready_addr + (down_stage) * 8);
                    down_stage += 1;
                    if (down_stage == 2) { down_stage = 0; _phase_down_free_1 ^= 1; _phase_down_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: consumer ----
    if (warp >= 2 && warp <= 5) {
        { // consumer_main
            unsigned int _phase_up_ready = 0;
            unsigned int _phase_down_ready = 0;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int route_work_c = blockIdx.x;
                int intermediate_blocks_c = gridDim.y;
                int intermediate_block_c = blockIdx.y;
                int subtiles_per_route_block_c = route_block_m / 8;
                int route_block_c = route_work_c / subtiles_per_route_block_c;
                int route_subtile_c = route_work_c % subtiles_per_route_block_c;
                int route_base_c = route_block_c * route_block_m + route_subtile_c * 8;
                bool route_active_c = route_base_c < num_tokens_post_padded[0];
                int expert_c = 0;
                if (route_active_c) {
                    expert_c = expert_ids[route_block_c];
                }
                int k_blocks_c = K / 128;
                const int consumer_lane = tid % 32;
                const int consumer_warp = warp % 4;
                const int consumer_tid = consumer_warp * 32 + consumer_lane;
                const int physical_row = consumer_warp * 32 + consumer_lane;
                const int scale_row = consumer_lane / 8 % 2;
                int lane_pair = M * top_k;
                int lane_token = M - 1;
                float lane_route_weight = 0.0f;
                if (consumer_lane < 8 && route_active_c) {
                    lane_pair = sorted_token_ids[route_base_c + consumer_lane];
                    int _min_0 = ((lane_pair / top_k) < (M - 1) ? (lane_pair / top_k) : (M - 1));
                    lane_token = _min_0;
                    if (lane_pair < M * top_k) {
                        lane_route_weight = topk_weights[lane_pair];
                    }
                }
                int route_pairs[8];
                float route_weights[8];
                #pragma unroll
                for (int token_slot_route = 0; token_slot_route < 8; token_slot_route++) {
                    int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, lane_pair, token_slot_route);
                    route_pairs[token_slot_route] = _shfl_sync_1;
                    float _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, lane_route_weight, token_slot_route);
                    route_weights[token_slot_route] = _shfl_sync_2;
                }
                const int panel_c = cta_rank;
                float acc[8];
                #pragma unroll
                for (int token_slot_init = 0; token_slot_init < 8; token_slot_init++) {
                    acc[token_slot_init] = 0.0f;
                }
                int w1_scale_row = (expert_c * intermediate_blocks_c * 2 + intermediate_block_c * 2 + consumer_lane) * k_blocks_c;
                const int scale_slot = consumer_lane % 16;
                const int w1_scale_half = (expert_c * intermediate_blocks_c * 2 + intermediate_block_c * 2 + consumer_lane / 16) * k_blocks_c;
                int x_scale_rows[8];
                float x_scales[8];
                float x_scales_next[8];
                int _min_3 = ((scale_slot) < (k_blocks_c - 1) ? (scale_slot) : (k_blocks_c - 1));
                const int first_scale_kb = _min_3;
                #pragma unroll
                for (int token_slot_xs = 0; token_slot_xs < 8; token_slot_xs++) {
                    int _min_4 = ((route_pairs[token_slot_xs] / top_k) < (M - 1) ? (route_pairs[token_slot_xs] / top_k) : (M - 1));
                    x_scale_rows[token_slot_xs] = _min_4 * k_blocks_c;
                    x_scales[token_slot_xs] = 0.0f;
                    x_scales_next[token_slot_xs] = x_scale[x_scale_rows[token_slot_xs] + first_scale_kb];
                }
                float w1_scales = 0.0f;
                float w1_scales_next = w1_scale[w1_scale_half + first_scale_kb];
                if (consumer_tid < 8) {
                    smem_amax[consumer_tid] = 0;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                unsigned int up_stage_c = 0;
                #pragma unroll 1
                for (int ks_c = 0; ks_c < (k_blocks_c + 1) / 2; ks_c++) {
                    mbarrier_wait(up_ready_addr + (up_stage_c) * 8, _phase_up_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int pack_addr = taddr + (unsigned int)(consumer_warp * 32 << 16) + up_stage_c * 16;
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], pack_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(up_free_addr + (up_stage_c) * 8);
                    }
                    up_stage_c += 1;
                    if (up_stage_c == 4) { up_stage_c = 0; _phase_up_ready ^= 1; }
                    #pragma unroll
                    for (int pack_half = 0; pack_half < 2; pack_half++) {
                        int kb_k = ks_c * 2 + pack_half;
                        if (kb_k < k_blocks_c) {
                            int slot_k = kb_k % 16;
                            if (slot_k == 0) {
                                #pragma unroll
                                for (int token_slot_nk_cp = 0; token_slot_nk_cp < 8; token_slot_nk_cp++) {
                                    x_scales[token_slot_nk_cp] = x_scales_next[token_slot_nk_cp];
                                }
                                w1_scales = w1_scales_next;
                                int next_kb_k = ((kb_k + 16 + scale_slot) < (k_blocks_c - 1) ? (kb_k + 16 + scale_slot) : (k_blocks_c - 1));
                                #pragma unroll
                                for (int token_slot_nk = 0; token_slot_nk < 8; token_slot_nk++) {
                                    x_scales_next[token_slot_nk] = x_scale[x_scale_rows[token_slot_nk] + next_kb_k];
                                }
                                w1_scales_next = w1_scale[w1_scale_half + next_kb_k];
                            }
                            float wscale_k = __shfl_sync(0xFFFFFFFF, w1_scales, slot_k + 16 * scale_row);
                            #pragma unroll
                            for (int token_slot_kr = 0; token_slot_kr < 8; token_slot_kr++) {
                                float xscale_kr = __shfl_sync(0xFFFFFFFF, x_scales[token_slot_kr], slot_k);
                                float scale_kr = xscale_kr * wscale_k;
                                acc[token_slot_kr] = __fmaf_rn(_tmem_load_0[pack_half * 8 + token_slot_kr], scale_kr, acc[token_slot_kr]);
                            }
                        }
                    }
                }
                const int w2s_blocks = K / 128;
                const int w2s_rank = blockIdx.z;
                const int w2s_ranks = gridDim.z;
                const int w2s_base = expert_c * w2s_blocks;
                int _min_5 = ((w2s_rank + consumer_lane * w2s_ranks) < (w2s_blocks - 1) ? (w2s_rank + consumer_lane * w2s_ranks) : (w2s_blocks - 1));
                const int w2s_first = _min_5;
                float w2_scales = w2_scale[(w2s_base + w2s_first) * intermediate_blocks_c + intermediate_block_c];
                int w2s_count = 0;
                float act[8];
                unsigned int amax_words[8];
                bool gate_lane = consumer_lane < 8;
                #pragma unroll
                for (int token_slot_act = 0; token_slot_act < 8; token_slot_act++) {
                    float peer = __shfl_xor_sync(0xFFFFFFFF, acc[token_slot_act], 8);
                    float gate = acc[token_slot_act];
                    float _exp_6 = __expf(-gate);
                    float sig = approx_rcp(1.0f + _exp_6);
                    act[token_slot_act] = gate * sig * peer;
                    unsigned int gate_word = 0;
                    if (gate_lane) {
                        uint32_t _amf_u_8 = __float_as_uint(fabsf(act[token_slot_act]));
                        uint32_t _amf_mask_8 = -int32_t(_amf_u_8 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_7 = _amf_u_8 ^ _amf_mask_8;
                        gate_word = _atomic_max_float_encode_7;
                    }
                    unsigned int _warp_redux_9;
                    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_9) : "r"(gate_word));
                    amax_words[token_slot_act] = _warp_redux_9;
                }
                if (consumer_lane < 8) {
                    unsigned int lane_word = amax_words[0];
                    #pragma unroll
                    for (int token_slot_sel = 1; token_slot_sel < 8; token_slot_sel++) {
                        if (consumer_lane == token_slot_sel) {
                            lane_word = amax_words[token_slot_sel];
                        }
                    }
                    atomicMax(&smem_amax[consumer_lane], lane_word);
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (consumer_tid < 8) {
                    unsigned int local_amax = smem_amax[consumer_tid];
                    if (panel_c != 0) {
                        uint32_t _cluster_mapa_10;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_10) : "r"(smem_peer_amax_addr + (unsigned int)((panel_c * 8 + consumer_tid) * 4)), "r"(0));
                        uint32_t _mbar_peer_11;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_11) : "r"(amax_ready_addr), "r"(0));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_10), "r"(local_amax), "r"(_mbar_peer_11) : "memory");
                    }
                    if (panel_c != 1) {
                        uint32_t _cluster_mapa_12;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_12) : "r"(smem_peer_amax_addr + (unsigned int)((panel_c * 8 + consumer_tid) * 4)), "r"(1));
                        uint32_t _mbar_peer_13;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_13) : "r"(amax_ready_addr), "r"(1));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_12), "r"(local_amax), "r"(_mbar_peer_13) : "memory");
                    }
                    if (panel_c != 2) {
                        uint32_t _cluster_mapa_14;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_14) : "r"(smem_peer_amax_addr + (unsigned int)((panel_c * 8 + consumer_tid) * 4)), "r"(2));
                        uint32_t _mbar_peer_15;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_15) : "r"(amax_ready_addr), "r"(2));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_14), "r"(local_amax), "r"(_mbar_peer_15) : "memory");
                    }
                    if (panel_c != 3) {
                        uint32_t _cluster_mapa_16;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_16) : "r"(smem_peer_amax_addr + (unsigned int)((panel_c * 8 + consumer_tid) * 4)), "r"(3));
                        uint32_t _mbar_peer_17;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_17) : "r"(amax_ready_addr), "r"(3));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_16), "r"(local_amax), "r"(_mbar_peer_17) : "memory");
                    }
                }
                if (warp == 2) {
                    if (elect_sync()) {
                        mbarrier_arrive(amax_ready_addr);
                        mbarrier_arrive(act_ready_addr);
                    }
                }
                mbarrier_wait_cluster_hint(amax_ready_addr, 0, 10000000);
                if (consumer_tid < 8) {
                    unsigned int merged = smem_amax[consumer_tid];
                    if (panel_c != 0) {
                        unsigned int _max_18 = ((merged) > (smem_peer_amax[consumer_tid]) ? (merged) : (smem_peer_amax[consumer_tid]));
                        merged = _max_18;
                    }
                    if (panel_c != 1) {
                        unsigned int _max_19 = ((merged) > (smem_peer_amax[8 + consumer_tid]) ? (merged) : (smem_peer_amax[8 + consumer_tid]));
                        merged = _max_19;
                    }
                    if (panel_c != 2) {
                        unsigned int _max_20 = ((merged) > (smem_peer_amax[16 + consumer_tid]) ? (merged) : (smem_peer_amax[16 + consumer_tid]));
                        merged = _max_20;
                    }
                    if (panel_c != 3) {
                        unsigned int _max_21 = ((merged) > (smem_peer_amax[24 + consumer_tid]) ? (merged) : (smem_peer_amax[24 + consumer_tid]));
                        merged = _max_21;
                    }
                    uint32_t _amf_u_22 = merged;
                    uint32_t _amf_mask_22 = ((_amf_u_22 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_22 ^ _amf_mask_22);
                    smem_act_scale[consumer_tid] = ((_amf_dec_0 == 0.0f) ? 1.0f : _amf_dec_0 / 448.0f);
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                const int feature = panel_c * 32 + consumer_warp * 8 + consumer_lane % 8;
                bool even_feature_lane = gate_lane && consumer_lane % 2 == 0;
                #pragma unroll
                for (int token_slot_quant = 0; token_slot_quant < 8; token_slot_quant++) {
                    float inv_scale = approx_rcp(smem_act_scale[token_slot_quant]);
                    float peer_feature = __shfl_xor_sync(0xFFFFFFFF, act[token_slot_quant], 1);
                    if (even_feature_lane) {
                        {
                            uint16_t _fp8_pair_23;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_23) : "f"(0.0f), "f"(act[token_slot_quant] * inv_scale));
                            uint32_t _byte_23 = (uint32_t)(_fp8_pair_23 & 0xFF);
                            uint32_t _addr_23 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + feature ^ (token_slot_quant * 128 + feature >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_23), "r"(_byte_23) : "memory");
                        }
                        {
                            uint16_t _fp8_pair_24;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_24) : "f"(0.0f), "f"(peer_feature * inv_scale));
                            uint32_t _byte_24 = (uint32_t)(_fp8_pair_24 & 0xFF);
                            uint32_t _addr_24 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + (feature + 1) ^ (token_slot_quant * 128 + (feature + 1) >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_24), "r"(_byte_24) : "memory");
                        }
                    }
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (consumer_warp == 0 && consumer_lane < 16) {
                    unsigned int chunk_addr = (unsigned int)(smem_act_addr + (unsigned int)(consumer_lane / 2 * 128 + (panel_c * 2 + consumer_lane % 2) * 16 ^ (consumer_lane / 2 * 128 + (panel_c * 2 + consumer_lane % 2) * 16 >> 7 & 7) << 4));
                    unsigned int chunk[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&chunk[0])), "=r"(*reinterpret_cast<uint32_t*>(&chunk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&chunk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&chunk[(0) + 3]))
                        : "r"(chunk_addr));
                    if (panel_c != 0) {
                        uint32_t _cluster_mapa_25;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_25) : "r"(chunk_addr), "r"(0));
                        uint32_t _mbar_peer_26;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_26) : "r"(act_ready_addr), "r"(0));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_25), "r"(chunk[0]), "r"(chunk[1]), "r"(chunk[2]), "r"(chunk[3]), "r"(_mbar_peer_26) : "memory");
                    }
                    if (panel_c != 1) {
                        uint32_t _cluster_mapa_27;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_27) : "r"(chunk_addr), "r"(1));
                        uint32_t _mbar_peer_28;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_28) : "r"(act_ready_addr), "r"(1));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_27), "r"(chunk[0]), "r"(chunk[1]), "r"(chunk[2]), "r"(chunk[3]), "r"(_mbar_peer_28) : "memory");
                    }
                    if (panel_c != 2) {
                        uint32_t _cluster_mapa_29;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_29) : "r"(chunk_addr), "r"(2));
                        uint32_t _mbar_peer_30;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_30) : "r"(act_ready_addr), "r"(2));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_29), "r"(chunk[0]), "r"(chunk[1]), "r"(chunk[2]), "r"(chunk[3]), "r"(_mbar_peer_30) : "memory");
                    }
                    if (panel_c != 3) {
                        uint32_t _cluster_mapa_31;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_31) : "r"(chunk_addr), "r"(3));
                        uint32_t _mbar_peer_32;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_32) : "r"(act_ready_addr), "r"(3));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_31), "r"(chunk[0]), "r"(chunk[1]), "r"(chunk[2]), "r"(chunk[3]), "r"(_mbar_peer_32) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                asm volatile("barrier.sync 1, 160;" ::: "memory");
                float down_act_scale[8];
                float down_token_scale[8];
                #pragma unroll
                for (int token_slot_scale = 0; token_slot_scale < 8; token_slot_scale++) {
                    down_act_scale[token_slot_scale] = smem_act_scale[token_slot_scale];
                    int pair_scale = route_pairs[token_slot_scale];
                    bool valid_scale = pair_scale < M * top_k;
                    float route_weight_scale = 0.0f;
                    if (valid_scale) {
                        route_weight_scale = route_weights[token_slot_scale];
                    }
                    down_token_scale[token_slot_scale] = down_act_scale[token_slot_scale] * route_weight_scale * scaling_factor;
                }
                int out_blocks_c = K / 128;
                unsigned int down_stage_c = 0;
                #pragma unroll 1
                for (int ob_c = blockIdx.z; ob_c < out_blocks_c; ob_c += gridDim.z) {
                    mbarrier_wait(down_ready_addr + (down_stage_c) * 8, _phase_down_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int down_addr = taddr + 64 + (unsigned int)(consumer_warp * 32 << 16) + down_stage_c * 8;
                    float _tmem_load_1[8];
                    tmem_ld_x8(&_tmem_load_1[0], down_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(down_free_addr + (down_stage_c) * 8);
                    }
                    int w2s_slot = w2s_count % 32;
                    float w2s = __shfl_sync(0xFFFFFFFF, w2_scales, w2s_slot);
                    if (w2s_slot == 31) {
                        int w2s_next = ((ob_c + (consumer_lane + 1) * w2s_ranks) < (w2s_blocks - 1) ? (ob_c + (consumer_lane + 1) * w2s_ranks) : (w2s_blocks - 1));
                        w2_scales = w2_scale[(w2s_base + w2s_next) * intermediate_blocks_c + intermediate_block_c];
                    }
                    w2s_count = w2s_count + 1;
                    #pragma unroll
                    for (int token_slot_out = 0; token_slot_out < 8; token_slot_out++) {
                        float out_scale = down_token_scale[token_slot_out] * w2s;
                        smem_out[((int)down_stage_c * 8 + token_slot_out) * 128 + physical_row] = _tmem_load_1[token_slot_out] * out_scale;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            #pragma unroll
                            for (int token_slot_reduce = 0; token_slot_reduce < 8; token_slot_reduce++) {
                                int pair_reduce = route_pairs[token_slot_reduce];
                                if (pair_reduce < M * top_k) {
                                    int token_reduce = pair_reduce / top_k;
                                    {
                                        void* _cpred_dst_33 = reinterpret_cast<void*>(out + (token_reduce * K + ob_c * 128));
                                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"
                                            " [%0], [%1], %2;"
                                            :: "l"(_cpred_dst_33), "r"(smem_out_addr + (unsigned int)(((int)down_stage_c * 8 + token_slot_reduce) * 128 * 2)), "r"((uint32_t)(256))
                                            : "memory");
                                    }
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    down_stage_c += 1;
                    if (down_stage_c == 2) { down_stage_c = 0; _phase_down_ready ^= 1; }
                }
                if (warp == 2) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"

#undef CAKE_INF
#undef TMEM_NCOLS
#undef TMEM_UP_ACC_OFFSET
#undef TMEM_DOWN_ACC_OFFSET
#undef NUM_UP_PIPE_STAGES
#undef NUM_DOWN_PIPE_STAGES
#undef SMEM_SMEM_W1_OFF
#undef SMEM_SMEM_W1_STAGE_BYTES
#undef SMEM_SMEM_W1_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_SMEM_W1B_OFF
#undef SMEM_SMEM_W1B_STAGE_BYTES
#undef SMEM_SMEM_W1B_STRIDE
#undef SMEM_SMEM_XB_OFF
#undef SMEM_SMEM_XB_STAGE_BYTES
#undef SMEM_SMEM_XB_STRIDE
#undef SMEM_SMEM_W2_OFF
#undef SMEM_SMEM_W2_STAGE_BYTES
#undef SMEM_SMEM_W2_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_AMAX_OFF
#undef SMEM_SMEM_AMAX_STAGE_BYTES
#undef SMEM_SMEM_AMAX_STRIDE
#undef SMEM_SMEM_ACT_SCALE_OFF
#undef SMEM_SMEM_ACT_SCALE_STAGE_BYTES
#undef SMEM_SMEM_ACT_SCALE_STRIDE
#undef SMEM_SMEM_PEER_AMAX_OFF
#undef SMEM_SMEM_PEER_AMAX_STAGE_BYTES
#undef SMEM_SMEM_PEER_AMAX_STRIDE
#undef SMEM_SMEM_OUT_OFF
#undef SMEM_SMEM_OUT_STAGE_BYTES
#undef SMEM_SMEM_OUT_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef LAUNCH_MIN_BLOCKS
#undef up_full_addr
#undef up_ready_addr
#undef up_free_addr
#undef down_full_addr
#undef down_ready_addr
#undef down_free_addr
#undef amax_ready_addr
#undef act_ready_addr
#undef kernel_alpha_moe_w8a8_up_down
constexpr int kGeneratedThreads = 192;
constexpr int kGeneratedSmemTotal = 112896;
}  // namespace w8a8_i2

namespace w8a8_j2 {
#define kernel_alpha_moe_w8a8_up_down kernel_alpha_moe_w8a8_up_down_w8a8_j2
#define TMEM_NCOLS 96
#define TMEM_UP_ACC_OFFSET 0
#define TMEM_DOWN_ACC_OFFSET 64
#define NUM_UP_PIPE_STAGES 4
#define NUM_DOWN_PIPE_STAGES 4
#define SMEM_SMEM_W1_OFF 1024
#define SMEM_SMEM_W1_STAGE_BYTES 32768
#define SMEM_SMEM_W1_STRIDE 33792
#define SMEM_SMEM_X_OFF 33792
#define SMEM_SMEM_X_STAGE_BYTES 1024
#define SMEM_SMEM_X_STRIDE 33792
#define SMEM_SMEM_W2_OFF 136192
#define SMEM_SMEM_W2_STAGE_BYTES 16384
#define SMEM_SMEM_W2_STRIDE 16384
#define SMEM_SMEM_ACT_OFF 201728
#define SMEM_SMEM_ACT_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_STRIDE 1024
#define SMEM_SMEM_AMAX_OFF 202752
#define SMEM_SMEM_AMAX_STAGE_BYTES 32
#define SMEM_SMEM_AMAX_STRIDE 32
#define SMEM_SMEM_ACT_SCALE_OFF 202784
#define SMEM_SMEM_ACT_SCALE_STAGE_BYTES 32
#define SMEM_SMEM_ACT_SCALE_STRIDE 32
#define SMEM_SMEM_OUT_OFF 202816
#define SMEM_SMEM_OUT_STAGE_BYTES 2048
#define SMEM_SMEM_OUT_STRIDE 2048
#define SMEM_TOTAL 204928
#define THREADS 192
#define LAUNCH_MIN_BLOCKS 1

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


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
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




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
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

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_alpha_moe_w8a8_up_down(const __grid_constant__ CUtensorMap x, const __grid_constant__ CUtensorMap W1, float* __restrict__ x_scale, float* __restrict__ w1_scale, const __grid_constant__ CUtensorMap W2, float* __restrict__ w2_scale, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, float* __restrict__ topk_weights, __nv_bfloat16* __restrict__ out, int M, int K, int top_k, int route_block_m, float scaling_factor)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define up_full_addr (mbar_base + 0)
    #define up_ready_addr (mbar_base + 32)
    #define up_free_addr (mbar_base + 64)
    #define down_full_addr (mbar_base + 96)
    #define down_ready_addr (mbar_base + 128)
    #define down_free_addr (mbar_base + 160)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_w1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_W1_OFF);
    const int smem_w1_addr = smem + SMEM_SMEM_W1_OFF;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_X_OFF);
    const int smem_x_addr = smem + SMEM_SMEM_X_OFF;
    uint8_t* smem_w2 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_W2_OFF);
    const int smem_w2_addr = smem + SMEM_SMEM_W2_OFF;
    uint8_t* smem_act = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_ACT_OFF);
    const int smem_act_addr = smem + SMEM_SMEM_ACT_OFF;
    unsigned int* smem_amax = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_AMAX_OFF);
    const int smem_amax_addr = smem + SMEM_SMEM_AMAX_OFF;
    float* smem_act_scale = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_ACT_SCALE_OFF);
    const int smem_act_scale_addr = smem + SMEM_SMEM_ACT_SCALE_OFF;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_OUT_OFF);
    const int smem_out_addr = smem + SMEM_SMEM_OUT_OFF;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'up_pipe' ---
            // up_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // up_ready: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // up_free: 4 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            // --- pipeline 'down_pipe' ---
            // down_full: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // down_ready: 4 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // down_free: 4 barriers, init_count=4
            mbarrier_init(smem + 160, 4);
            mbarrier_init(smem + 168, 4);
            mbarrier_init(smem + 176, 4);
            mbarrier_init(smem + 184, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 0) {
        int _tmem_hold = smem + 192;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_up_acc = taddr;
    const int tmem_down_acc = taddr + 64;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int _phase_up_free = 1;
            unsigned int _phase_down_free = 1;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int route_work = blockIdx.x;
                int intermediate_block = blockIdx.y;
                int subtiles_per_route_block = route_block_m / 8;
                int route_block = route_work / subtiles_per_route_block;
                int route_subtile = route_work % subtiles_per_route_block;
                int route_base = route_block * route_block_m + route_subtile * 8;
                bool route_active = route_base < num_tokens_post_padded[0];
                int expert = 0;
                if (route_active) {
                    expert = expert_ids[route_block];
                }
                int k_blocks = K / 128;
                unsigned int stage = 0;
                #pragma unroll 1
                for (int kb = 0; kb < k_blocks; kb++) {
                    mbarrier_wait(up_free_addr + (stage) * 8, _phase_up_free);
                    if (elect_sync()) {
                        int pair0 = M * top_k;
                        int pair1 = M * top_k;
                        int pair2 = M * top_k;
                        int pair3 = M * top_k;
                        int pair4 = M * top_k;
                        int pair5 = M * top_k;
                        int pair6 = M * top_k;
                        int pair7 = M * top_k;
                        if (route_active) {
                            pair0 = sorted_token_ids[route_base];
                            pair1 = sorted_token_ids[route_base + 1];
                            pair2 = sorted_token_ids[route_base + 2];
                            pair3 = sorted_token_ids[route_base + 3];
                            pair4 = sorted_token_ids[route_base + 4];
                            pair5 = sorted_token_ids[route_base + 5];
                            pair6 = sorted_token_ids[route_base + 6];
                            pair7 = sorted_token_ids[route_base + 7];
                        }
                        int _min_0 = ((pair0 / top_k) < (M - 1) ? (pair0 / top_k) : (M - 1));
                        int row0 = _min_0;
                        int _min_1 = ((pair1 / top_k) < (M - 1) ? (pair1 / top_k) : (M - 1));
                        int row1 = _min_1;
                        int _min_2 = ((pair2 / top_k) < (M - 1) ? (pair2 / top_k) : (M - 1));
                        int row2 = _min_2;
                        int _min_3 = ((pair3 / top_k) < (M - 1) ? (pair3 / top_k) : (M - 1));
                        int row3 = _min_3;
                        int _min_4 = ((pair4 / top_k) < (M - 1) ? (pair4 / top_k) : (M - 1));
                        int row4 = _min_4;
                        int _min_5 = ((pair5 / top_k) < (M - 1) ? (pair5 / top_k) : (M - 1));
                        int row5 = _min_5;
                        int _min_6 = ((pair6 / top_k) < (M - 1) ? (pair6 / top_k) : (M - 1));
                        int row6 = _min_6;
                        int _min_7 = ((pair7 / top_k) < (M - 1) ? (pair7 / top_k) : (M - 1));
                        int row7 = _min_7;
                        mbarrier_arrive_expect_tx(up_full_addr + (stage) * 8, 33792);
                        tma_gather4_gmem2smem(smem_x_addr + stage * 33792, (&x), kb * 128, row0, row1, row2, row3, up_full_addr + (stage) * 8);
                        tma_gather4_gmem2smem(smem_x_addr + stage * 33792 + 512, (&x), kb * 128, row4, row5, row6, row7, up_full_addr + (stage) * 8);
                        tma_4d_gmem2smem(smem_w1_addr + stage * 33792, (&W1), 0, intermediate_block * 256, kb, expert, up_full_addr + (stage) * 8);
                        tma_4d_gmem2smem(smem_w1_addr + stage * 33792 + 16384, (&W1), 0, intermediate_block * 256 + 128, kb, expert, up_full_addr + (stage) * 8);
                    }
                    stage += 1;
                    if (stage == 4) { stage = 0; _phase_up_free ^= 1; }
                }
                asm volatile("barrier.sync 14, 192;" ::: "memory");
                stage = 0;
                int out_blocks = K / 128;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int ob = blockIdx.z; ob < out_blocks; ob += gridDim.z) {
                        mbarrier_wait(down_free_addr + (stage) * 8, _phase_down_free);
                        mbarrier_arrive_expect_tx(down_full_addr + (stage) * 8, 16384);
                        tma_3d_gmem2smem(smem_w2_addr + stage * 16384, (&W2), intermediate_block * 128, ob * 128, expert, down_full_addr + (stage) * 8);
                        stage += 1;
                        if (stage == 4) { stage = 0; _phase_down_free ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int _phase_up_free_1 = 1;
            unsigned int _phase_up_full = 0;
            unsigned int _phase_down_free_1 = 1;
            unsigned int _phase_down_full = 0;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int k_blocks_mma = K / 128;
                unsigned int up_stage = 0;
                #pragma unroll 1
                for (int kb_mma = 0; kb_mma < k_blocks_mma; kb_mma++) {
                    mbarrier_wait(up_free_addr + (up_stage) * 8, _phase_up_free_1);
                    mbarrier_wait(up_full_addr + (up_stage) * 8, _phase_up_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_w1_addr) >> 4) & 0x3FFF) + (up_stage) * 2112);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (up_stage) * 2112);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                        }
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((smem_w1_addr + 16384) >> 4) & 0x3FFF) + (up_stage) * 2112);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (up_stage) * 2112);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                        }
                    }
                    elect_commit(up_ready_addr + (up_stage) * 8);
                    up_stage += 1;
                    if (up_stage == 4) { up_stage = 0; _phase_up_free_1 ^= 1; _phase_up_full ^= 1; }
                }
                asm volatile("barrier.sync 14, 192;" ::: "memory");
                int out_blocks_mma = K / 128;
                unsigned int down_stage = 0;
                #pragma unroll 1
                for (int ob_mma = blockIdx.z; ob_mma < out_blocks_mma; ob_mma += gridDim.z) {
                    mbarrier_wait(down_free_addr + (down_stage) * 8, _phase_down_free_1);
                    mbarrier_wait(down_full_addr + (down_stage) * 8, _phase_down_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_2 = make_warp_uniform((((smem_w2_addr) >> 4) & 0x3FFF) + (down_stage) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform(((smem_act_addr) >> 4) & 0x3FFF);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                        }
                    }
                    elect_commit(down_ready_addr + (down_stage) * 8);
                    down_stage += 1;
                    if (down_stage == 4) { down_stage = 0; _phase_down_free_1 ^= 1; _phase_down_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: consumer ----
    if (warp >= 2 && warp <= 5) {
        { // consumer_main
            unsigned int _phase_up_ready = 0;
            unsigned int _phase_down_ready = 0;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int route_work_c = blockIdx.x;
                int intermediate_blocks_c = gridDim.y;
                int intermediate_block_c = blockIdx.y;
                int subtiles_per_route_block_c = route_block_m / 8;
                int route_block_c = route_work_c / subtiles_per_route_block_c;
                int route_subtile_c = route_work_c % subtiles_per_route_block_c;
                int route_base_c = route_block_c * route_block_m + route_subtile_c * 8;
                bool route_active_c = route_base_c < num_tokens_post_padded[0];
                int expert_c = 0;
                if (route_active_c) {
                    expert_c = expert_ids[route_block_c];
                }
                int k_blocks_c = K / 128;
                const int consumer_lane = tid % 32;
                const int consumer_warp = warp % 4;
                const int consumer_tid = consumer_warp * 32 + consumer_lane;
                const int physical_row = consumer_warp * 32 + consumer_lane;
                const int scale_row = physical_row / 8 % 2;
                int lane_pair = M * top_k;
                int lane_token = M - 1;
                float lane_route_weight = 0.0f;
                if (consumer_lane < 8 && route_active_c) {
                    lane_pair = sorted_token_ids[route_base_c + consumer_lane];
                    int _min_8 = ((lane_pair / top_k) < (M - 1) ? (lane_pair / top_k) : (M - 1));
                    lane_token = _min_8;
                    if (lane_pair < M * top_k) {
                        lane_route_weight = topk_weights[lane_pair];
                    }
                }
                int route_pairs[8];
                float route_weights[8];
                #pragma unroll
                for (int token_slot_route = 0; token_slot_route < 8; token_slot_route++) {
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, lane_pair, token_slot_route);
                    route_pairs[token_slot_route] = _shfl_0;
                    float _shfl_1 = __shfl_sync(0xFFFFFFFF, lane_route_weight, token_slot_route);
                    route_weights[token_slot_route] = _shfl_1;
                }
                float panel0[8];
                float panel1[8];
                #pragma unroll
                for (int token_slot_init = 0; token_slot_init < 8; token_slot_init++) {
                    panel0[token_slot_init] = 0.0f;
                    panel1[token_slot_init] = 0.0f;
                }
                unsigned int up_stage_c = 0;
                #pragma unroll 1
                for (int kb_c = 0; kb_c < k_blocks_c; kb_c++) {
                    mbarrier_wait(up_ready_addr + (up_stage_c) * 8, _phase_up_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int row_base = consumer_warp * 32;
                    int panel0_addr = taddr + (unsigned int)(row_base << 16) + up_stage_c * 16;
                    int panel1_addr = panel0_addr + 8;
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], panel0_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_1[8];
                    tmem_ld_x8(&_tmem_load_1[0], panel1_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float wscale_lane = 0.0f;
                    if (consumer_lane < 2) {
                        wscale_lane = w1_scale[(expert_c * intermediate_blocks_c * 2 + intermediate_block_c * 2 + consumer_lane) * k_blocks_c + kb_c];
                    }
                    float _shfl_2 = __shfl_sync(0xFFFFFFFF, wscale_lane, scale_row);
                    float wscale = _shfl_2;
                    float xscale_lane = 0.0f;
                    if (consumer_lane < 8) {
                        xscale_lane = x_scale[lane_token * k_blocks_c + kb_c];
                    }
                    #pragma unroll
                    for (int token_slot_acc = 0; token_slot_acc < 8; token_slot_acc++) {
                        float _shfl_3 = __shfl_sync(0xFFFFFFFF, xscale_lane, token_slot_acc);
                        float xscale = _shfl_3;
                        float scale = xscale * wscale;
                        panel0[token_slot_acc] = panel0[token_slot_acc] + _tmem_load_0[token_slot_acc] * scale;
                        panel1[token_slot_acc] = panel1[token_slot_acc] + _tmem_load_1[token_slot_acc] * scale;
                    }
                    if (elect_sync()) {
                        mbarrier_arrive(up_free_addr + (up_stage_c) * 8);
                    }
                    up_stage_c += 1;
                    if (up_stage_c == 4) { up_stage_c = 0; _phase_up_ready ^= 1; }
                }
                if (consumer_tid < 8) {
                    smem_amax[consumer_tid] = 0;
                }
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                float act0[8];
                float act1[8];
                bool gate_lane = consumer_lane % 16 < 8;
                #pragma unroll
                for (int token_slot_act = 0; token_slot_act < 8; token_slot_act++) {
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, panel0[token_slot_act], 8);
                    float peer0 = _shfl_xor_0;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, panel1[token_slot_act], 8);
                    float peer1 = _shfl_xor_1;
                    float gate0 = panel0[token_slot_act];
                    float gate1 = panel1[token_slot_act];
                    float _expf_0 = __expf(-gate0);
                    float _rcp_0 = approx_rcp(1.0f + _expf_0);
                    float sig0 = _rcp_0;
                    float _expf_1 = __expf(-gate1);
                    float _rcp_1 = approx_rcp(1.0f + _expf_1);
                    float sig1 = _rcp_1;
                    act0[token_slot_act] = gate0 * sig0 * peer0;
                    act1[token_slot_act] = gate1 * sig1 * peer1;
                    if (gate_lane) {
                        float _fabs_0 = fabsf(act0[token_slot_act]);
                        float abs0 = _fabs_0;
                        float _fabs_1 = fabsf(act1[token_slot_act]);
                        float abs1 = _fabs_1;
                        float _max_0 = max_noftz(abs0, abs1);
                        float local_max = _max_0;
                        uint32_t _amf_u_0 = __float_as_uint(local_max);
                        uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
                        atomicMax(&smem_amax[token_slot_act], _amf_enc_0);
                    }
                }
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                if (consumer_tid < 8) {
                    uint32_t _amf_u_1 = smem_amax[consumer_tid];
                    uint32_t _amf_mask_1 = ((_amf_u_1 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_1 ^ _amf_mask_1);
                    smem_act_scale[consumer_tid] = ((_amf_dec_0 == 0.0f) ? 1.0f : _amf_dec_0 / 448.0f);
                }
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                const int feature_in_half = physical_row / 16 * 8 + physical_row % 8;
                bool even_feature_lane = gate_lane && consumer_lane % 2 == 0;
                #pragma unroll
                for (int token_slot_quant = 0; token_slot_quant < 8; token_slot_quant++) {
                    float _rcp_2 = approx_rcp(smem_act_scale[token_slot_quant]);
                    float inv_scale = _rcp_2;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, act0[token_slot_quant], 1);
                    float peer_feature0 = _shfl_xor_2;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, act1[token_slot_quant], 1);
                    float peer_feature1 = _shfl_xor_3;
                    if (even_feature_lane) {
                        {
                            uint16_t _fp8_pair_2;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_2) : "f"(0.0f), "f"(act0[token_slot_quant] * inv_scale));
                            uint32_t _byte_2 = (uint32_t)(_fp8_pair_2 & 0xFF);
                            uint32_t _addr_2 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + feature_in_half ^ (token_slot_quant * 128 + feature_in_half >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_2), "r"(_byte_2) : "memory");
                        }
                        {
                            uint16_t _fp8_pair_3;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_3) : "f"(0.0f), "f"(peer_feature0 * inv_scale));
                            uint32_t _byte_3 = (uint32_t)(_fp8_pair_3 & 0xFF);
                            uint32_t _addr_3 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + (feature_in_half + 1) ^ (token_slot_quant * 128 + (feature_in_half + 1) >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_3), "r"(_byte_3) : "memory");
                        }
                        {
                            uint16_t _fp8_pair_4;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_4) : "f"(0.0f), "f"(act1[token_slot_quant] * inv_scale));
                            uint32_t _byte_4 = (uint32_t)(_fp8_pair_4 & 0xFF);
                            uint32_t _addr_4 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + (64 + feature_in_half) ^ (token_slot_quant * 128 + (64 + feature_in_half) >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_4), "r"(_byte_4) : "memory");
                        }
                        {
                            uint16_t _fp8_pair_5;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_5) : "f"(0.0f), "f"(peer_feature1 * inv_scale));
                            uint32_t _byte_5 = (uint32_t)(_fp8_pair_5 & 0xFF);
                            uint32_t _addr_5 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + (64 + feature_in_half + 1) ^ (token_slot_quant * 128 + (64 + feature_in_half + 1) >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_5), "r"(_byte_5) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 15, 128;" ::: "memory");
                asm volatile("barrier.sync 14, 192;" ::: "memory");
                float down_act_scale[8];
                float down_token_scale[8];
                #pragma unroll
                for (int token_slot_scale = 0; token_slot_scale < 8; token_slot_scale++) {
                    down_act_scale[token_slot_scale] = smem_act_scale[token_slot_scale];
                    int pair_scale = route_pairs[token_slot_scale];
                    bool valid_scale = pair_scale < M * top_k;
                    float route_weight_scale = 0.0f;
                    if (valid_scale) {
                        route_weight_scale = route_weights[token_slot_scale];
                    }
                    down_token_scale[token_slot_scale] = down_act_scale[token_slot_scale] * route_weight_scale * scaling_factor;
                }
                int out_blocks_c = K / 128;
                unsigned int down_stage_c = 0;
                #pragma unroll 1
                for (int ob_c = blockIdx.z; ob_c < out_blocks_c; ob_c += gridDim.z) {
                    mbarrier_wait(down_ready_addr + (down_stage_c) * 8, _phase_down_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int down_addr = taddr + 64 + (unsigned int)(consumer_warp * 32 << 16) + down_stage_c * 8;
                    float _tmem_load_2[8];
                    tmem_ld_x8(&_tmem_load_2[0], down_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float w2s = w2_scale[(expert_c * out_blocks_c + ob_c) * intermediate_blocks_c + intermediate_block_c];
                    #pragma unroll
                    for (int token_slot_out = 0; token_slot_out < 8; token_slot_out++) {
                        float out_scale = down_token_scale[token_slot_out] * w2s;
                        smem_out[token_slot_out * 128 + physical_row] = _tmem_load_2[token_slot_out] * out_scale;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 15, 128;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            #pragma unroll
                            for (int token_slot_reduce = 0; token_slot_reduce < 8; token_slot_reduce++) {
                                int pair_reduce = route_pairs[token_slot_reduce];
                                if (pair_reduce < M * top_k) {
                                    int token_reduce = pair_reduce / top_k;
                                    {
                                        void* _cpred_dst_6 = reinterpret_cast<void*>(out + (token_reduce * K + ob_c * 128));
                                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"
                                            " [%0], [%1], %2;"
                                            :: "l"(_cpred_dst_6), "r"(smem_out_addr + (unsigned int)(token_slot_reduce * 128 * 2)), "r"((uint32_t)(256))
                                            : "memory");
                                    }
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                    }
                    asm volatile("barrier.sync 15, 128;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(down_free_addr + (down_stage_c) * 8);
                    }
                    down_stage_c += 1;
                    if (down_stage_c == 4) { down_stage_c = 0; _phase_down_ready ^= 1; }
                }
                if (warp == 2) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"

constexpr int kGeneratedThreads = THREADS;
constexpr int kGeneratedSmemTotal = SMEM_TOTAL;
#undef TMEM_NCOLS
#undef TMEM_UP_ACC_OFFSET
#undef TMEM_DOWN_ACC_OFFSET
#undef NUM_UP_PIPE_STAGES
#undef NUM_DOWN_PIPE_STAGES
#undef SMEM_SMEM_W1_OFF
#undef SMEM_SMEM_W1_STAGE_BYTES
#undef SMEM_SMEM_W1_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_SMEM_W2_OFF
#undef SMEM_SMEM_W2_STAGE_BYTES
#undef SMEM_SMEM_W2_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_AMAX_OFF
#undef SMEM_SMEM_AMAX_STAGE_BYTES
#undef SMEM_SMEM_AMAX_STRIDE
#undef SMEM_SMEM_ACT_SCALE_OFF
#undef SMEM_SMEM_ACT_SCALE_STAGE_BYTES
#undef SMEM_SMEM_ACT_SCALE_STRIDE
#undef SMEM_SMEM_OUT_OFF
#undef SMEM_SMEM_OUT_STAGE_BYTES
#undef SMEM_SMEM_OUT_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef LAUNCH_MIN_BLOCKS
#undef up_full_addr
#undef up_ready_addr
#undef up_free_addr
#undef down_full_addr
#undef down_ready_addr
#undef down_free_addr
#undef kernel_alpha_moe_w8a8_up_down
}  // namespace w8a8_j2

namespace w8a8_l2 {
#define kernel_alpha_moe_w8a8_up_down kernel_alpha_moe_w8a8_up_down_w8a8_l2
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
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else

#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 48
#define TMEM_UP_ACC_OFFSET 0
#define TMEM_DOWN_ACC_OFFSET 32
#define NUM_UP_PIPE_STAGES 2
#define NUM_DOWN_PIPE_STAGES 2
#define SMEM_SMEM_W1_OFF 1024
#define SMEM_SMEM_W1_STAGE_BYTES 32768
#define SMEM_SMEM_W1_STRIDE 33792
#define SMEM_SMEM_X_OFF 33792
#define SMEM_SMEM_X_STAGE_BYTES 1024
#define SMEM_SMEM_X_STRIDE 33792
#define SMEM_SMEM_W2_OFF 68608
#define SMEM_SMEM_W2_STAGE_BYTES 16384
#define SMEM_SMEM_W2_STRIDE 16384
#define SMEM_SMEM_ACT_OFF 101376
#define SMEM_SMEM_ACT_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_STRIDE 1024
#define SMEM_SMEM_AMAX_OFF 102400
#define SMEM_SMEM_AMAX_STAGE_BYTES 32
#define SMEM_SMEM_AMAX_STRIDE 32
#define SMEM_SMEM_ACT_SCALE_OFF 102432
#define SMEM_SMEM_ACT_SCALE_STAGE_BYTES 32
#define SMEM_SMEM_ACT_SCALE_STRIDE 32
#define SMEM_SMEM_OUT_OFF 102464
#define SMEM_SMEM_OUT_STAGE_BYTES 4096
#define SMEM_SMEM_OUT_STRIDE 4096
#define SMEM_TOTAL 106624
#define THREADS 192
#define LAUNCH_MIN_BLOCKS 2

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

__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
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

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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
    // Non-multicast, single-CTA gather4 in the .shared::cta window (the
    // cuda::ptx form): the destination and the barrier are the executing
    // CTA's. A CTA pair's gather takes the .shared::cluster window
    // (tma_gather4_gmem2smem_cta2), its barrier being the pair leader's.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_alpha_moe_w8a8_up_down(const __grid_constant__ CUtensorMap x, const __grid_constant__ CUtensorMap W1, float* __restrict__ x_scale, float* __restrict__ w1_scale, const __grid_constant__ CUtensorMap W2, float* __restrict__ w2_scale, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, float* __restrict__ topk_weights, __nv_bfloat16* __restrict__ out, int M, int K, int top_k, int route_block_m, float scaling_factor)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define up_full_addr (mbar_base + 0)
    #define up_ready_addr (mbar_base + 16)
    #define up_free_addr (mbar_base + 32)
    #define up_empty_addr (mbar_base + 48)
    #define down_full_addr (mbar_base + 64)
    #define down_ready_addr (mbar_base + 80)
    #define down_free_addr (mbar_base + 96)
    #define down_empty_addr (mbar_base + 112)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_w1 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w1_addr = smem + 1024;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_x_addr = smem + 33792;
    uint8_t* smem_w2 = reinterpret_cast<uint8_t*>(smem_raw + 68608);
    const int smem_w2_addr = smem + 68608;
    uint8_t* smem_act = reinterpret_cast<uint8_t*>(smem_raw + 101376);
    const int smem_act_addr = smem + 101376;
    unsigned int* smem_amax = reinterpret_cast<unsigned int*>(smem_raw + 102400);
    const int smem_amax_addr = smem + 102400;
    float* smem_act_scale = reinterpret_cast<float*>(smem_raw + 102432);
    const int smem_act_scale_addr = smem + 102432;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 102464);
    const int smem_out_addr = smem + 102464;
    if (__syncthreads_and(blockIdx.x * 8 >= num_tokens_post_padded[0])) return;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'up_pipe' ---
            // up_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // up_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // up_free: 2 barriers, init_count=4
            mbarrier_init(smem + 32, 4);
            mbarrier_init(smem + 40, 4);
            // up_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'down_pipe' ---
            // down_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // down_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // down_free: 2 barriers, init_count=4
            mbarrier_init(smem + 96, 4);
            mbarrier_init(smem + 104, 4);
            // down_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 48 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_up_acc = taddr;
    const int tmem_down_acc = taddr + 32;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int _phase_up_empty = 1;
            unsigned int _phase_down_empty = 1;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int route_work = blockIdx.x;
                int intermediate_block = blockIdx.y;
                int subtiles_per_route_block = route_block_m / 8;
                int route_block = route_work / subtiles_per_route_block;
                int route_subtile = route_work % subtiles_per_route_block;
                int route_base = route_block * route_block_m + route_subtile * 8;
                bool route_active = route_base < num_tokens_post_padded[0];
                int expert = 0;
                if (route_active) {
                    expert = expert_ids[route_block];
                }
                int k_blocks = K / 128;
                int pair0 = M * top_k;
                int pair1 = M * top_k;
                int pair2 = M * top_k;
                int pair3 = M * top_k;
                int pair4 = M * top_k;
                int pair5 = M * top_k;
                int pair6 = M * top_k;
                int pair7 = M * top_k;
                if (route_active) {
                    pair0 = sorted_token_ids[route_base];
                    pair1 = sorted_token_ids[route_base + 1];
                    pair2 = sorted_token_ids[route_base + 2];
                    pair3 = sorted_token_ids[route_base + 3];
                    pair4 = sorted_token_ids[route_base + 4];
                    pair5 = sorted_token_ids[route_base + 5];
                    pair6 = sorted_token_ids[route_base + 6];
                    pair7 = sorted_token_ids[route_base + 7];
                }
                int row0 = ((pair0 / top_k) < (M - 1) ? (pair0 / top_k) : (M - 1));
                int row1 = ((pair1 / top_k) < (M - 1) ? (pair1 / top_k) : (M - 1));
                int row2 = ((pair2 / top_k) < (M - 1) ? (pair2 / top_k) : (M - 1));
                int row3 = ((pair3 / top_k) < (M - 1) ? (pair3 / top_k) : (M - 1));
                int row4 = ((pair4 / top_k) < (M - 1) ? (pair4 / top_k) : (M - 1));
                int row5 = ((pair5 / top_k) < (M - 1) ? (pair5 / top_k) : (M - 1));
                int row6 = ((pair6 / top_k) < (M - 1) ? (pair6 / top_k) : (M - 1));
                int row7 = ((pair7 / top_k) < (M - 1) ? (pair7 / top_k) : (M - 1));
                unsigned int stage = 0;
                #pragma unroll 1
                for (int kb = 0; kb < k_blocks; kb++) {
                    mbarrier_wait(up_empty_addr + (stage) * 8, _phase_up_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(up_full_addr + (stage) * 8, 33792);
                        tma_gather4_gmem2smem(smem_x_addr + stage * 33792, (&x), kb * 128, row0, row1, row2, row3, up_full_addr + (stage) * 8);
                        tma_gather4_gmem2smem(smem_x_addr + stage * 33792 + 512, (&x), kb * 128, row4, row5, row6, row7, up_full_addr + (stage) * 8);
                        tma_4d_gmem2smem(smem_w1_addr + stage * 33792, (&W1), 0, intermediate_block * 256, kb, expert, up_full_addr + (stage) * 8);
                        tma_4d_gmem2smem(smem_w1_addr + stage * 33792 + 16384, (&W1), 0, intermediate_block * 256 + 128, kb, expert, up_full_addr + (stage) * 8);
                    }
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_up_empty ^= 1; }
                }
                unsigned int down_load_stage = 0;
                int out_blocks = K / 128;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int ob = blockIdx.z; ob < out_blocks; ob += gridDim.z) {
                        mbarrier_wait(down_empty_addr + (down_load_stage) * 8, _phase_down_empty);
                        mbarrier_arrive_expect_tx(down_full_addr + (down_load_stage) * 8, 16384);
                        tma_3d_gmem2smem(smem_w2_addr + down_load_stage * 16384, (&W2), intermediate_block * 128, ob * 128, expert, down_full_addr + (down_load_stage) * 8);
                        down_load_stage += 1;
                        if (down_load_stage == 2) { down_load_stage = 0; _phase_down_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int _phase_up_free = 1;
            unsigned int _phase_up_full = 0;
            unsigned int _phase_down_free = 1;
            unsigned int _phase_down_full = 0;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int k_blocks_mma = K / 128;
                unsigned int up_stage = 0;
                #pragma unroll 1
                for (int kb_mma = 0; kb_mma < k_blocks_mma; kb_mma++) {
                    mbarrier_wait(up_free_addr + (up_stage) * 8, _phase_up_free);
                    mbarrier_wait(up_full_addr + (up_stage) * 8, _phase_up_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_w1_addr) >> 4) & 0x3FFF) + (up_stage) * 2112);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (up_stage) * 2112);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                        }
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((smem_w1_addr + 16384) >> 4) & 0x3FFF) + (up_stage) * 2112);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (up_stage) * 2112);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                        }
                    }
                    elect_commit(up_ready_addr + (up_stage) * 8);
                    elect_commit(up_empty_addr + (up_stage) * 8);
                    up_stage += 1;
                    if (up_stage == 2) { up_stage = 0; _phase_up_free ^= 1; _phase_up_full ^= 1; }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 160;" ::: "memory");
                int out_blocks_mma = K / 128;
                unsigned int down_stage = 0;
                #pragma unroll 1
                for (int ob_mma = blockIdx.z; ob_mma < out_blocks_mma; ob_mma += gridDim.z) {
                    mbarrier_wait(down_free_addr + (down_stage) * 8, _phase_down_free);
                    mbarrier_wait(down_full_addr + (down_stage) * 8, _phase_down_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_2 = make_warp_uniform((((smem_w2_addr) >> 4) & 0x3FFF) + (down_stage) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform(((smem_act_addr) >> 4) & 0x3FFF);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                        }
                    }
                    elect_commit(down_ready_addr + (down_stage) * 8);
                    elect_commit(down_empty_addr + (down_stage) * 8);
                    down_stage += 1;
                    if (down_stage == 2) { down_stage = 0; _phase_down_free ^= 1; _phase_down_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: consumer ----
    if (warp >= 2 && warp <= 5) {
        { // consumer_main
            unsigned int _phase_up_ready = 0;
            unsigned int _phase_down_ready = 0;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int route_work_c = blockIdx.x;
                int intermediate_blocks_c = gridDim.y;
                int intermediate_block_c = blockIdx.y;
                int subtiles_per_route_block_c = route_block_m / 8;
                int route_block_c = route_work_c / subtiles_per_route_block_c;
                int route_subtile_c = route_work_c % subtiles_per_route_block_c;
                int route_base_c = route_block_c * route_block_m + route_subtile_c * 8;
                bool route_active_c = route_base_c < num_tokens_post_padded[0];
                int expert_c = 0;
                if (route_active_c) {
                    expert_c = expert_ids[route_block_c];
                }
                int k_blocks_c = K / 128;
                const int consumer_lane = tid % 32;
                const int consumer_warp = warp % 4;
                const int consumer_tid = consumer_warp * 32 + consumer_lane;
                const int physical_row = consumer_warp * 32 + consumer_lane;
                const int scale_row = physical_row / 8 % 2;
                int lane_pair = M * top_k;
                int lane_token = M - 1;
                float lane_route_weight = 0.0f;
                if (consumer_lane < 8 && route_active_c) {
                    lane_pair = sorted_token_ids[route_base_c + consumer_lane];
                    int _min_0 = ((lane_pair / top_k) < (M - 1) ? (lane_pair / top_k) : (M - 1));
                    lane_token = _min_0;
                    if (lane_pair < M * top_k) {
                        lane_route_weight = topk_weights[lane_pair];
                    }
                }
                int route_pairs[8];
                float route_weights[8];
                #pragma unroll
                for (int token_slot_route = 0; token_slot_route < 8; token_slot_route++) {
                    int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, lane_pair, token_slot_route);
                    route_pairs[token_slot_route] = _shfl_sync_1;
                    float _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, lane_route_weight, token_slot_route);
                    route_weights[token_slot_route] = _shfl_sync_2;
                }
                float panel0[8];
                float panel1[8];
                #pragma unroll
                for (int token_slot_init = 0; token_slot_init < 8; token_slot_init++) {
                    panel0[token_slot_init] = 0.0f;
                    panel1[token_slot_init] = 0.0f;
                }
                const int scale_slot = consumer_lane % 16;
                const int w1_scale_half = (expert_c * intermediate_blocks_c * 2 + intermediate_block_c * 2 + consumer_lane / 16) * k_blocks_c;
                int x_scale_rows[8];
                float x_scales[8];
                float x_scales_next[8];
                int _min_3 = ((scale_slot) < (k_blocks_c - 1) ? (scale_slot) : (k_blocks_c - 1));
                const int first_scale_kb = _min_3;
                #pragma unroll
                for (int token_slot_xs = 0; token_slot_xs < 8; token_slot_xs++) {
                    int _min_4 = ((route_pairs[token_slot_xs] / top_k) < (M - 1) ? (route_pairs[token_slot_xs] / top_k) : (M - 1));
                    x_scale_rows[token_slot_xs] = _min_4 * k_blocks_c;
                    x_scales[token_slot_xs] = 0.0f;
                    x_scales_next[token_slot_xs] = x_scale[x_scale_rows[token_slot_xs] + first_scale_kb];
                }
                float w1_scales = 0.0f;
                float w1_scales_next = w1_scale[w1_scale_half + first_scale_kb];
                unsigned int up_stage_c = 0;
                #pragma unroll 1
                for (int kb_c = 0; kb_c < k_blocks_c; kb_c++) {
                    mbarrier_wait(up_ready_addr + (up_stage_c) * 8, _phase_up_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int row_base = consumer_warp * 32;
                    int panel0_addr = taddr + (unsigned int)(row_base << 16) + up_stage_c * 16;
                    int panel1_addr = panel0_addr + 8;
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], panel0_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_1[8];
                    tmem_ld_x8(&_tmem_load_1[0], panel1_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(up_free_addr + (up_stage_c) * 8);
                    }
                    int slot_kb = kb_c % 16;
                    if (slot_kb == 0) {
                        #pragma unroll
                        for (int token_slot_cp = 0; token_slot_cp < 8; token_slot_cp++) {
                            x_scales[token_slot_cp] = x_scales_next[token_slot_cp];
                        }
                        w1_scales = w1_scales_next;
                        int next_kb = ((kb_c + 16 + scale_slot) < (k_blocks_c - 1) ? (kb_c + 16 + scale_slot) : (k_blocks_c - 1));
                        #pragma unroll
                        for (int token_slot_nx = 0; token_slot_nx < 8; token_slot_nx++) {
                            x_scales_next[token_slot_nx] = x_scale[x_scale_rows[token_slot_nx] + next_kb];
                        }
                        w1_scales_next = w1_scale[w1_scale_half + next_kb];
                    }
                    float wscale = __shfl_sync(0xFFFFFFFF, w1_scales, slot_kb + 16 * scale_row);
                    #pragma unroll
                    for (int token_slot_acc = 0; token_slot_acc < 8; token_slot_acc++) {
                        float xscale = __shfl_sync(0xFFFFFFFF, x_scales[token_slot_acc], slot_kb);
                        float scale = xscale * wscale;
                        panel0[token_slot_acc] = __fmaf_rn(_tmem_load_0[token_slot_acc], scale, panel0[token_slot_acc]);
                        panel1[token_slot_acc] = __fmaf_rn(_tmem_load_1[token_slot_acc], scale, panel1[token_slot_acc]);
                    }
                    up_stage_c += 1;
                    if (up_stage_c == 2) { up_stage_c = 0; _phase_up_ready ^= 1; }
                }
                const int w2s_blocks = K / 128;
                const int w2s_rank = blockIdx.z;
                const int w2s_ranks = gridDim.z;
                const int w2s_base = expert_c * w2s_blocks;
                int _min_5 = ((w2s_rank + consumer_lane * w2s_ranks) < (w2s_blocks - 1) ? (w2s_rank + consumer_lane * w2s_ranks) : (w2s_blocks - 1));
                const int w2s_first = _min_5;
                float w2_scales = w2_scale[(w2s_base + w2s_first) * intermediate_blocks_c + intermediate_block_c];
                int w2s_count = 0;
                if (consumer_tid < 8) {
                    smem_amax[consumer_tid] = 0;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                float act0[8];
                float act1[8];
                bool gate_lane = consumer_lane % 16 < 8;
                #pragma unroll
                for (int token_slot_act = 0; token_slot_act < 8; token_slot_act++) {
                    float peer0 = __shfl_xor_sync(0xFFFFFFFF, panel0[token_slot_act], 8);
                    float peer1 = __shfl_xor_sync(0xFFFFFFFF, panel1[token_slot_act], 8);
                    float gate0 = panel0[token_slot_act];
                    float gate1 = panel1[token_slot_act];
                    float _exp_6 = __expf(-gate0);
                    float sig0 = approx_rcp(1.0f + _exp_6);
                    float _exp_7 = __expf(-gate1);
                    float sig1 = approx_rcp(1.0f + _exp_7);
                    act0[token_slot_act] = gate0 * sig0 * peer0;
                    act1[token_slot_act] = gate1 * sig1 * peer1;
                    if (gate_lane) {
                        float abs0 = fabsf(act0[token_slot_act]);
                        float abs1 = fabsf(act1[token_slot_act]);
                        float local_max;
                        asm("max.f32 %0, %1, %2;" : "=f"(local_max) : "f"(abs0), "f"(abs1));
                        uint32_t _amf_u_9 = __float_as_uint(local_max);
                        uint32_t _amf_mask_9 = -int32_t(_amf_u_9 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_8 = _amf_u_9 ^ _amf_mask_9;
                        atomicMax(&smem_amax[token_slot_act], _atomic_max_float_encode_8);
                    }
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (consumer_tid < 8) {
                    uint32_t _amf_u_10 = smem_amax[consumer_tid];
                    uint32_t _amf_mask_10 = ((_amf_u_10 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_10 ^ _amf_mask_10);
                    smem_act_scale[consumer_tid] = ((_amf_dec_0 == 0.0f) ? 1.0f : _amf_dec_0 / 448.0f);
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                const int feature_in_half = physical_row / 16 * 8 + physical_row % 8;
                bool even_feature_lane = gate_lane && consumer_lane % 2 == 0;
                #pragma unroll
                for (int token_slot_quant = 0; token_slot_quant < 8; token_slot_quant++) {
                    float inv_scale = approx_rcp(smem_act_scale[token_slot_quant]);
                    float peer_feature0 = __shfl_xor_sync(0xFFFFFFFF, act0[token_slot_quant], 1);
                    float peer_feature1 = __shfl_xor_sync(0xFFFFFFFF, act1[token_slot_quant], 1);
                    if (even_feature_lane) {
                        {
                            uint16_t _fp8_pair_11;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_11) : "f"(0.0f), "f"(act0[token_slot_quant] * inv_scale));
                            uint32_t _byte_11 = (uint32_t)(_fp8_pair_11 & 0xFF);
                            uint32_t _addr_11 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + feature_in_half ^ (token_slot_quant * 128 + feature_in_half >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_11), "r"(_byte_11) : "memory");
                        }
                        {
                            uint16_t _fp8_pair_12;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_12) : "f"(0.0f), "f"(peer_feature0 * inv_scale));
                            uint32_t _byte_12 = (uint32_t)(_fp8_pair_12 & 0xFF);
                            uint32_t _addr_12 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + (feature_in_half + 1) ^ (token_slot_quant * 128 + (feature_in_half + 1) >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_12), "r"(_byte_12) : "memory");
                        }
                        {
                            uint16_t _fp8_pair_13;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_13) : "f"(0.0f), "f"(act1[token_slot_quant] * inv_scale));
                            uint32_t _byte_13 = (uint32_t)(_fp8_pair_13 & 0xFF);
                            uint32_t _addr_13 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + (64 + feature_in_half) ^ (token_slot_quant * 128 + (64 + feature_in_half) >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_13), "r"(_byte_13) : "memory");
                        }
                        {
                            uint16_t _fp8_pair_14;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_14) : "f"(0.0f), "f"(peer_feature1 * inv_scale));
                            uint32_t _byte_14 = (uint32_t)(_fp8_pair_14 & 0xFF);
                            uint32_t _addr_14 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + (64 + feature_in_half + 1) ^ (token_slot_quant * 128 + (64 + feature_in_half + 1) >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_14), "r"(_byte_14) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                asm volatile("barrier.sync 1, 160;" ::: "memory");
                float down_act_scale[8];
                float down_token_scale[8];
                #pragma unroll
                for (int token_slot_scale = 0; token_slot_scale < 8; token_slot_scale++) {
                    down_act_scale[token_slot_scale] = smem_act_scale[token_slot_scale];
                    int pair_scale = route_pairs[token_slot_scale];
                    bool valid_scale = pair_scale < M * top_k;
                    float route_weight_scale = 0.0f;
                    if (valid_scale) {
                        route_weight_scale = route_weights[token_slot_scale];
                    }
                    down_token_scale[token_slot_scale] = down_act_scale[token_slot_scale] * route_weight_scale * scaling_factor;
                }
                int out_blocks_c = K / 128;
                unsigned int down_stage_c = 0;
                #pragma unroll 1
                for (int ob_c = blockIdx.z; ob_c < out_blocks_c; ob_c += gridDim.z) {
                    mbarrier_wait(down_ready_addr + (down_stage_c) * 8, _phase_down_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int down_addr = taddr + 32 + (unsigned int)(consumer_warp * 32 << 16) + down_stage_c * 8;
                    float _tmem_load_2[8];
                    tmem_ld_x8(&_tmem_load_2[0], down_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(down_free_addr + (down_stage_c) * 8);
                    }
                    int w2s_slot = w2s_count % 32;
                    float w2s = __shfl_sync(0xFFFFFFFF, w2_scales, w2s_slot);
                    if (w2s_slot == 31) {
                        int w2s_next = ((ob_c + (consumer_lane + 1) * w2s_ranks) < (w2s_blocks - 1) ? (ob_c + (consumer_lane + 1) * w2s_ranks) : (w2s_blocks - 1));
                        w2_scales = w2_scale[(w2s_base + w2s_next) * intermediate_blocks_c + intermediate_block_c];
                    }
                    w2s_count = w2s_count + 1;
                    #pragma unroll
                    for (int token_slot_out = 0; token_slot_out < 8; token_slot_out++) {
                        float out_scale = down_token_scale[token_slot_out] * w2s;
                        smem_out[((int)down_stage_c * 8 + token_slot_out) * 128 + physical_row] = _tmem_load_2[token_slot_out] * out_scale;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            #pragma unroll
                            for (int token_slot_reduce = 0; token_slot_reduce < 8; token_slot_reduce++) {
                                int pair_reduce = route_pairs[token_slot_reduce];
                                if (pair_reduce < M * top_k) {
                                    int token_reduce = pair_reduce / top_k;
                                    {
                                        void* _cpred_dst_15 = reinterpret_cast<void*>(out + (token_reduce * K + ob_c * 128));
                                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"
                                            " [%0], [%1], %2;"
                                            :: "l"(_cpred_dst_15), "r"(smem_out_addr + (unsigned int)(((int)down_stage_c * 8 + token_slot_reduce) * 128 * 2)), "r"((uint32_t)(256))
                                            : "memory");
                                    }
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    down_stage_c += 1;
                    if (down_stage_c == 2) { down_stage_c = 0; _phase_down_ready ^= 1; }
                }
                if (warp == 2) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
    }
}

} // extern "C"

#undef CAKE_INF
#undef TMEM_NCOLS
#undef TMEM_UP_ACC_OFFSET
#undef TMEM_DOWN_ACC_OFFSET
#undef NUM_UP_PIPE_STAGES
#undef NUM_DOWN_PIPE_STAGES
#undef SMEM_SMEM_W1_OFF
#undef SMEM_SMEM_W1_STAGE_BYTES
#undef SMEM_SMEM_W1_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_SMEM_W2_OFF
#undef SMEM_SMEM_W2_STAGE_BYTES
#undef SMEM_SMEM_W2_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_AMAX_OFF
#undef SMEM_SMEM_AMAX_STAGE_BYTES
#undef SMEM_SMEM_AMAX_STRIDE
#undef SMEM_SMEM_ACT_SCALE_OFF
#undef SMEM_SMEM_ACT_SCALE_STAGE_BYTES
#undef SMEM_SMEM_ACT_SCALE_STRIDE
#undef SMEM_SMEM_OUT_OFF
#undef SMEM_SMEM_OUT_STAGE_BYTES
#undef SMEM_SMEM_OUT_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef LAUNCH_MIN_BLOCKS
#undef up_full_addr
#undef up_ready_addr
#undef up_free_addr
#undef up_empty_addr
#undef down_full_addr
#undef down_ready_addr
#undef down_free_addr
#undef down_empty_addr
#undef kernel_alpha_moe_w8a8_up_down
constexpr int kGeneratedThreads = 192;
constexpr int kGeneratedSmemTotal = 106624;
}  // namespace w8a8_l2

namespace w8a8_p2 {
#define kernel_alpha_moe_w8a8_up_down kernel_alpha_moe_w8a8_up_down_w8a8_p2
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
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else

#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 48
#define TMEM_UP_ACC_OFFSET 0
#define TMEM_DOWN_ACC_OFFSET 32
#define NUM_UP_PIPE_STAGES 4
#define NUM_DOWN_PIPE_STAGES 2
#define SMEM_SMEM_W1_OFF 1024
#define SMEM_SMEM_W1_STAGE_BYTES 16384
#define SMEM_SMEM_W1_STRIDE 17408
#define SMEM_SMEM_X_OFF 17408
#define SMEM_SMEM_X_STAGE_BYTES 1024
#define SMEM_SMEM_X_STRIDE 17408
#define SMEM_SMEM_W2_OFF 70656
#define SMEM_SMEM_W2_STAGE_BYTES 16384
#define SMEM_SMEM_W2_STRIDE 16384
#define SMEM_SMEM_ACT_OFF 103424
#define SMEM_SMEM_ACT_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_STRIDE 1024
#define SMEM_SMEM_AMAX_OFF 104448
#define SMEM_SMEM_AMAX_STAGE_BYTES 32
#define SMEM_SMEM_AMAX_STRIDE 32
#define SMEM_SMEM_ACT_SCALE_OFF 104480
#define SMEM_SMEM_ACT_SCALE_STAGE_BYTES 32
#define SMEM_SMEM_ACT_SCALE_STRIDE 32
#define SMEM_SMEM_PEER_AMAX_OFF 104512
#define SMEM_SMEM_PEER_AMAX_STAGE_BYTES 32
#define SMEM_SMEM_PEER_AMAX_STRIDE 32
#define SMEM_SMEM_OUT_OFF 104544
#define SMEM_SMEM_OUT_STAGE_BYTES 4096
#define SMEM_SMEM_OUT_STRIDE 4096
#define SMEM_TOTAL 108672
#define THREADS 192
#define LAUNCH_MIN_BLOCKS 2

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

__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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

__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
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

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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
    // Non-multicast, single-CTA gather4 in the .shared::cta window (the
    // cuda::ptx form): the destination and the barrier are the executing
    // CTA's. A CTA pair's gather takes the .shared::cluster window
    // (tma_gather4_gmem2smem_cta2), its barrier being the pair leader's.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) __cluster_dims__(1,1,2) void
kernel_alpha_moe_w8a8_up_down(const __grid_constant__ CUtensorMap x, const __grid_constant__ CUtensorMap W1, float* __restrict__ x_scale, float* __restrict__ w1_scale, const __grid_constant__ CUtensorMap W2, float* __restrict__ w2_scale, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, float* __restrict__ topk_weights, __nv_bfloat16* __restrict__ out, int M, int K, int top_k, int route_block_m, float scaling_factor)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define up_full_addr (mbar_base + 0)
    #define up_ready_addr (mbar_base + 32)
    #define up_free_addr (mbar_base + 64)
    #define up_empty_addr (mbar_base + 96)
    #define down_full_addr (mbar_base + 128)
    #define down_ready_addr (mbar_base + 144)
    #define down_free_addr (mbar_base + 160)
    #define down_empty_addr (mbar_base + 176)
    #define amax_ready_addr (mbar_base + 192)
    #define act_ready_addr (mbar_base + 200)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = (((blockIdx.z / 2) * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * gridDim.y * (gridDim.z / 2);

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_w1 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w1_addr = smem + 1024;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_x_addr = smem + 17408;
    uint8_t* smem_w2 = reinterpret_cast<uint8_t*>(smem_raw + 70656);
    const int smem_w2_addr = smem + 70656;
    uint8_t* smem_act = reinterpret_cast<uint8_t*>(smem_raw + 103424);
    const int smem_act_addr = smem + 103424;
    unsigned int* smem_amax = reinterpret_cast<unsigned int*>(smem_raw + 104448);
    const int smem_amax_addr = smem + 104448;
    float* smem_act_scale = reinterpret_cast<float*>(smem_raw + 104480);
    const int smem_act_scale_addr = smem + 104480;
    unsigned int* smem_peer_amax = reinterpret_cast<unsigned int*>(smem_raw + 104512);
    const int smem_peer_amax_addr = smem + 104512;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 104544);
    const int smem_out_addr = smem + 104544;
    if (blockIdx.x * 8 >= num_tokens_post_padded[0]) {
        return;
    }

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'up_pipe' ---
            // up_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // up_ready: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // up_free: 4 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            // up_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // --- pipeline 'down_pipe' ---
            // down_full: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // down_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // down_free: 2 barriers, init_count=4
            mbarrier_init(smem + 160, 4);
            mbarrier_init(smem + 168, 4);
            // down_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // amax_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // act_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 192, 32);
            mbarrier_expect_tx(smem + 200, 512);
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 48 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 0) {
        int _tmem_hold = smem + 208;
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
    const int tmem_up_acc = taddr;
    const int tmem_down_acc = taddr + 32;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int _phase_up_empty = 1;
            unsigned int _phase_down_empty = 1;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int route_work = blockIdx.x;
                int intermediate_block = blockIdx.y;
                int subtiles_per_route_block = route_block_m / 8;
                int route_block = route_work / subtiles_per_route_block;
                int route_subtile = route_work % subtiles_per_route_block;
                int route_base = route_block * route_block_m + route_subtile * 8;
                bool route_active = route_base < num_tokens_post_padded[0];
                int expert = 0;
                if (route_active) {
                    expert = expert_ids[route_block];
                }
                int k_blocks = K / 128;
                const int panel = cta_rank;
                int pair0 = M * top_k;
                int pair1 = M * top_k;
                int pair2 = M * top_k;
                int pair3 = M * top_k;
                int pair4 = M * top_k;
                int pair5 = M * top_k;
                int pair6 = M * top_k;
                int pair7 = M * top_k;
                if (route_active) {
                    pair0 = sorted_token_ids[route_base];
                    pair1 = sorted_token_ids[route_base + 1];
                    pair2 = sorted_token_ids[route_base + 2];
                    pair3 = sorted_token_ids[route_base + 3];
                    pair4 = sorted_token_ids[route_base + 4];
                    pair5 = sorted_token_ids[route_base + 5];
                    pair6 = sorted_token_ids[route_base + 6];
                    pair7 = sorted_token_ids[route_base + 7];
                }
                int row0 = ((pair0 / top_k) < (M - 1) ? (pair0 / top_k) : (M - 1));
                int row1 = ((pair1 / top_k) < (M - 1) ? (pair1 / top_k) : (M - 1));
                int row2 = ((pair2 / top_k) < (M - 1) ? (pair2 / top_k) : (M - 1));
                int row3 = ((pair3 / top_k) < (M - 1) ? (pair3 / top_k) : (M - 1));
                int row4 = ((pair4 / top_k) < (M - 1) ? (pair4 / top_k) : (M - 1));
                int row5 = ((pair5 / top_k) < (M - 1) ? (pair5 / top_k) : (M - 1));
                int row6 = ((pair6 / top_k) < (M - 1) ? (pair6 / top_k) : (M - 1));
                int row7 = ((pair7 / top_k) < (M - 1) ? (pair7 / top_k) : (M - 1));
                unsigned int stage = 0;
                #pragma unroll 1
                for (int kb = 0; kb < k_blocks; kb++) {
                    mbarrier_wait(up_empty_addr + (stage) * 8, _phase_up_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(up_full_addr + (stage) * 8, 17408);
                        tma_gather4_gmem2smem(smem_x_addr + stage * 17408, (&x), kb * 128, row0, row1, row2, row3, up_full_addr + (stage) * 8);
                        tma_gather4_gmem2smem(smem_x_addr + stage * 17408 + 512, (&x), kb * 128, row4, row5, row6, row7, up_full_addr + (stage) * 8);
                        tma_4d_gmem2smem(smem_w1_addr + stage * 17408, (&W1), 0, intermediate_block * 256 + panel * 128, kb, expert, up_full_addr + (stage) * 8);
                    }
                    stage += 1;
                    if (stage == 4) { stage = 0; _phase_up_empty ^= 1; }
                }
                unsigned int down_load_stage = 0;
                int out_blocks = K / 128;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int ob = blockIdx.z; ob < out_blocks; ob += gridDim.z) {
                        mbarrier_wait(down_empty_addr + (down_load_stage) * 8, _phase_down_empty);
                        mbarrier_arrive_expect_tx(down_full_addr + (down_load_stage) * 8, 16384);
                        tma_3d_gmem2smem(smem_w2_addr + down_load_stage * 16384, (&W2), intermediate_block * 128, ob * 128, expert, down_full_addr + (down_load_stage) * 8);
                        down_load_stage += 1;
                        if (down_load_stage == 2) { down_load_stage = 0; _phase_down_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int _phase_up_free = 1;
            unsigned int _phase_up_full = 0;
            unsigned int _phase_down_free = 1;
            unsigned int _phase_down_full = 0;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int k_blocks_mma = K / 128;
                unsigned int up_stage = 0;
                #pragma unroll 1
                for (int kb_mma = 0; kb_mma < k_blocks_mma; kb_mma++) {
                    mbarrier_wait(up_free_addr + (up_stage) * 8, _phase_up_free);
                    mbarrier_wait(up_full_addr + (up_stage) * 8, _phase_up_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_w1_addr) >> 4) & 0x3FFF) + (up_stage) * 1088);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (up_stage) * 1088);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                        }
                    }
                    elect_commit(up_ready_addr + (up_stage) * 8);
                    elect_commit(up_empty_addr + (up_stage) * 8);
                    up_stage += 1;
                    if (up_stage == 4) { up_stage = 0; _phase_up_free ^= 1; _phase_up_full ^= 1; }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 160;" ::: "memory");
                mbarrier_wait_cluster_hint(act_ready_addr, 0, 10000000);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int out_blocks_mma = K / 128;
                unsigned int down_stage = 0;
                #pragma unroll 1
                for (int ob_mma = blockIdx.z; ob_mma < out_blocks_mma; ob_mma += gridDim.z) {
                    mbarrier_wait(down_free_addr + (down_stage) * 8, _phase_down_free);
                    mbarrier_wait(down_full_addr + (down_stage) * 8, _phase_down_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_1 = make_warp_uniform((((smem_w2_addr) >> 4) & 0x3FFF) + (down_stage) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform(((smem_act_addr) >> 4) & 0x3FFF);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                        }
                    }
                    elect_commit(down_ready_addr + (down_stage) * 8);
                    elect_commit(down_empty_addr + (down_stage) * 8);
                    down_stage += 1;
                    if (down_stage == 2) { down_stage = 0; _phase_down_free ^= 1; _phase_down_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: consumer ----
    if (warp >= 2 && warp <= 5) {
        { // consumer_main
            unsigned int _phase_up_ready = 0;
            unsigned int _phase_down_ready = 0;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int route_work_c = blockIdx.x;
                int intermediate_blocks_c = gridDim.y;
                int intermediate_block_c = blockIdx.y;
                int subtiles_per_route_block_c = route_block_m / 8;
                int route_block_c = route_work_c / subtiles_per_route_block_c;
                int route_subtile_c = route_work_c % subtiles_per_route_block_c;
                int route_base_c = route_block_c * route_block_m + route_subtile_c * 8;
                bool route_active_c = route_base_c < num_tokens_post_padded[0];
                int expert_c = 0;
                if (route_active_c) {
                    expert_c = expert_ids[route_block_c];
                }
                int k_blocks_c = K / 128;
                const int consumer_lane = tid % 32;
                const int consumer_warp = warp % 4;
                const int consumer_tid = consumer_warp * 32 + consumer_lane;
                const int physical_row = consumer_warp * 32 + consumer_lane;
                const int scale_row = physical_row / 8 % 2;
                int lane_pair = M * top_k;
                int lane_token = M - 1;
                float lane_route_weight = 0.0f;
                if (consumer_lane < 8 && route_active_c) {
                    lane_pair = sorted_token_ids[route_base_c + consumer_lane];
                    int _min_0 = ((lane_pair / top_k) < (M - 1) ? (lane_pair / top_k) : (M - 1));
                    lane_token = _min_0;
                    if (lane_pair < M * top_k) {
                        lane_route_weight = topk_weights[lane_pair];
                    }
                }
                int route_pairs[8];
                float route_weights[8];
                #pragma unroll
                for (int token_slot_route = 0; token_slot_route < 8; token_slot_route++) {
                    int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, lane_pair, token_slot_route);
                    route_pairs[token_slot_route] = _shfl_sync_1;
                    float _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, lane_route_weight, token_slot_route);
                    route_weights[token_slot_route] = _shfl_sync_2;
                }
                const int panel_c = cta_rank;
                float acc[8];
                #pragma unroll
                for (int token_slot_init = 0; token_slot_init < 8; token_slot_init++) {
                    acc[token_slot_init] = 0.0f;
                }
                int w1_scale_row = (expert_c * intermediate_blocks_c * 2 + intermediate_block_c * 2 + consumer_lane) * k_blocks_c;
                const int scale_slot = consumer_lane % 16;
                const int w1_scale_half = (expert_c * intermediate_blocks_c * 2 + intermediate_block_c * 2 + consumer_lane / 16) * k_blocks_c;
                int x_scale_rows[8];
                float x_scales[8];
                float x_scales_next[8];
                int _min_3 = ((scale_slot) < (k_blocks_c - 1) ? (scale_slot) : (k_blocks_c - 1));
                const int first_scale_kb = _min_3;
                #pragma unroll
                for (int token_slot_xs = 0; token_slot_xs < 8; token_slot_xs++) {
                    int _min_4 = ((route_pairs[token_slot_xs] / top_k) < (M - 1) ? (route_pairs[token_slot_xs] / top_k) : (M - 1));
                    x_scale_rows[token_slot_xs] = _min_4 * k_blocks_c;
                    x_scales[token_slot_xs] = 0.0f;
                    x_scales_next[token_slot_xs] = x_scale[x_scale_rows[token_slot_xs] + first_scale_kb];
                }
                float w1_scales = 0.0f;
                float w1_scales_next = w1_scale[w1_scale_half + first_scale_kb];
                if (consumer_tid < 8) {
                    smem_amax[consumer_tid] = 0;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                unsigned int up_stage_c = 0;
                #pragma unroll 1
                for (int kb_c = 0; kb_c < k_blocks_c; kb_c++) {
                    mbarrier_wait(up_ready_addr + (up_stage_c) * 8, _phase_up_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int row_base = consumer_warp * 32;
                    int panel_addr = taddr + (unsigned int)(row_base << 16) + up_stage_c * 8;
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], panel_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(up_free_addr + (up_stage_c) * 8);
                    }
                    int slot_kb = kb_c % 16;
                    if (slot_kb == 0) {
                        #pragma unroll
                        for (int token_slot_cp = 0; token_slot_cp < 8; token_slot_cp++) {
                            x_scales[token_slot_cp] = x_scales_next[token_slot_cp];
                        }
                        w1_scales = w1_scales_next;
                        int next_kb = ((kb_c + 16 + scale_slot) < (k_blocks_c - 1) ? (kb_c + 16 + scale_slot) : (k_blocks_c - 1));
                        #pragma unroll
                        for (int token_slot_nx = 0; token_slot_nx < 8; token_slot_nx++) {
                            x_scales_next[token_slot_nx] = x_scale[x_scale_rows[token_slot_nx] + next_kb];
                        }
                        w1_scales_next = w1_scale[w1_scale_half + next_kb];
                    }
                    float wscale = __shfl_sync(0xFFFFFFFF, w1_scales, slot_kb + 16 * scale_row);
                    #pragma unroll
                    for (int token_slot_acc = 0; token_slot_acc < 8; token_slot_acc++) {
                        float xscale = __shfl_sync(0xFFFFFFFF, x_scales[token_slot_acc], slot_kb);
                        float scale = xscale * wscale;
                        acc[token_slot_acc] = __fmaf_rn(_tmem_load_0[token_slot_acc], scale, acc[token_slot_acc]);
                    }
                    up_stage_c += 1;
                    if (up_stage_c == 4) { up_stage_c = 0; _phase_up_ready ^= 1; }
                }
                const int w2s_blocks = K / 128;
                const int w2s_rank = blockIdx.z;
                const int w2s_ranks = gridDim.z;
                const int w2s_base = expert_c * w2s_blocks;
                int _min_5 = ((w2s_rank + consumer_lane * w2s_ranks) < (w2s_blocks - 1) ? (w2s_rank + consumer_lane * w2s_ranks) : (w2s_blocks - 1));
                const int w2s_first = _min_5;
                float w2_scales = w2_scale[(w2s_base + w2s_first) * intermediate_blocks_c + intermediate_block_c];
                int w2s_count = 0;
                float act[8];
                unsigned int amax_words[8];
                bool gate_lane = consumer_lane % 16 < 8;
                #pragma unroll
                for (int token_slot_act = 0; token_slot_act < 8; token_slot_act++) {
                    float peer = __shfl_xor_sync(0xFFFFFFFF, acc[token_slot_act], 8);
                    float gate = acc[token_slot_act];
                    float _exp_6 = __expf(-gate);
                    float sig = approx_rcp(1.0f + _exp_6);
                    act[token_slot_act] = gate * sig * peer;
                    unsigned int gate_word = 0;
                    if (gate_lane) {
                        uint32_t _amf_u_8 = __float_as_uint(fabsf(act[token_slot_act]));
                        uint32_t _amf_mask_8 = -int32_t(_amf_u_8 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_7 = _amf_u_8 ^ _amf_mask_8;
                        gate_word = _atomic_max_float_encode_7;
                    }
                    unsigned int _warp_redux_9;
                    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_9) : "r"(gate_word));
                    amax_words[token_slot_act] = _warp_redux_9;
                }
                if (consumer_lane < 8) {
                    unsigned int lane_word = amax_words[0];
                    #pragma unroll
                    for (int token_slot_sel = 1; token_slot_sel < 8; token_slot_sel++) {
                        if (consumer_lane == token_slot_sel) {
                            lane_word = amax_words[token_slot_sel];
                        }
                    }
                    atomicMax(&smem_amax[consumer_lane], lane_word);
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (consumer_tid < 8) {
                    unsigned int local_amax = smem_amax[consumer_tid];
                    if (panel_c != 0) {
                        uint32_t _cluster_mapa_10;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_10) : "r"(smem_peer_amax_addr + (unsigned int)(consumer_tid * 4)), "r"(0));
                        uint32_t _mbar_peer_11;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_11) : "r"(amax_ready_addr), "r"(0));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_10), "r"(local_amax), "r"(_mbar_peer_11) : "memory");
                    }
                    if (panel_c != 1) {
                        uint32_t _cluster_mapa_12;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_12) : "r"(smem_peer_amax_addr + (unsigned int)(consumer_tid * 4)), "r"(1));
                        uint32_t _mbar_peer_13;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_13) : "r"(amax_ready_addr), "r"(1));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_12), "r"(local_amax), "r"(_mbar_peer_13) : "memory");
                    }
                }
                if (warp == 2) {
                    if (elect_sync()) {
                        mbarrier_arrive(amax_ready_addr);
                        mbarrier_arrive(act_ready_addr);
                    }
                }
                mbarrier_wait_cluster_hint(amax_ready_addr, 0, 10000000);
                if (consumer_tid < 8) {
                    unsigned int _max_14 = ((smem_amax[consumer_tid]) > (smem_peer_amax[consumer_tid]) ? (smem_amax[consumer_tid]) : (smem_peer_amax[consumer_tid]));
                    uint32_t _amf_u_15 = _max_14;
                    uint32_t _amf_mask_15 = ((_amf_u_15 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_15 ^ _amf_mask_15);
                    smem_act_scale[consumer_tid] = ((_amf_dec_0 == 0.0f) ? 1.0f : _amf_dec_0 / 448.0f);
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                const int feature = panel_c * 64 + physical_row / 16 * 8 + physical_row % 8;
                bool even_feature_lane = gate_lane && consumer_lane % 2 == 0;
                #pragma unroll
                for (int token_slot_quant = 0; token_slot_quant < 8; token_slot_quant++) {
                    float inv_scale = approx_rcp(smem_act_scale[token_slot_quant]);
                    float peer_feature = __shfl_xor_sync(0xFFFFFFFF, act[token_slot_quant], 1);
                    if (even_feature_lane) {
                        {
                            uint16_t _fp8_pair_16;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_16) : "f"(0.0f), "f"(act[token_slot_quant] * inv_scale));
                            uint32_t _byte_16 = (uint32_t)(_fp8_pair_16 & 0xFF);
                            uint32_t _addr_16 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + feature ^ (token_slot_quant * 128 + feature >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_16), "r"(_byte_16) : "memory");
                        }
                        {
                            uint16_t _fp8_pair_17;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_17) : "f"(0.0f), "f"(peer_feature * inv_scale));
                            uint32_t _byte_17 = (uint32_t)(_fp8_pair_17 & 0xFF);
                            uint32_t _addr_17 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + (feature + 1) ^ (token_slot_quant * 128 + (feature + 1) >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_17), "r"(_byte_17) : "memory");
                        }
                    }
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (consumer_warp == 0) {
                    unsigned int chunk_addr = (unsigned int)(smem_act_addr + (unsigned int)(consumer_lane / 4 * 128 + (panel_c * 4 + consumer_lane % 4) * 16 ^ (consumer_lane / 4 * 128 + (panel_c * 4 + consumer_lane % 4) * 16 >> 7 & 7) << 4));
                    unsigned int chunk[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&chunk[0])), "=r"(*reinterpret_cast<uint32_t*>(&chunk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&chunk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&chunk[(0) + 3]))
                        : "r"(chunk_addr));
                    if (panel_c != 0) {
                        uint32_t _cluster_mapa_18;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_18) : "r"(chunk_addr), "r"(0));
                        uint32_t _mbar_peer_19;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_19) : "r"(act_ready_addr), "r"(0));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_18), "r"(chunk[0]), "r"(chunk[1]), "r"(chunk[2]), "r"(chunk[3]), "r"(_mbar_peer_19) : "memory");
                    }
                    if (panel_c != 1) {
                        uint32_t _cluster_mapa_20;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_20) : "r"(chunk_addr), "r"(1));
                        uint32_t _mbar_peer_21;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_21) : "r"(act_ready_addr), "r"(1));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_20), "r"(chunk[0]), "r"(chunk[1]), "r"(chunk[2]), "r"(chunk[3]), "r"(_mbar_peer_21) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                asm volatile("barrier.sync 1, 160;" ::: "memory");
                float down_act_scale[8];
                float down_token_scale[8];
                #pragma unroll
                for (int token_slot_scale = 0; token_slot_scale < 8; token_slot_scale++) {
                    down_act_scale[token_slot_scale] = smem_act_scale[token_slot_scale];
                    int pair_scale = route_pairs[token_slot_scale];
                    bool valid_scale = pair_scale < M * top_k;
                    float route_weight_scale = 0.0f;
                    if (valid_scale) {
                        route_weight_scale = route_weights[token_slot_scale];
                    }
                    down_token_scale[token_slot_scale] = down_act_scale[token_slot_scale] * route_weight_scale * scaling_factor;
                }
                int out_blocks_c = K / 128;
                unsigned int down_stage_c = 0;
                #pragma unroll 1
                for (int ob_c = blockIdx.z; ob_c < out_blocks_c; ob_c += gridDim.z) {
                    mbarrier_wait(down_ready_addr + (down_stage_c) * 8, _phase_down_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int down_addr = taddr + 32 + (unsigned int)(consumer_warp * 32 << 16) + down_stage_c * 8;
                    float _tmem_load_1[8];
                    tmem_ld_x8(&_tmem_load_1[0], down_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(down_free_addr + (down_stage_c) * 8);
                    }
                    int w2s_slot = w2s_count % 32;
                    float w2s = __shfl_sync(0xFFFFFFFF, w2_scales, w2s_slot);
                    if (w2s_slot == 31) {
                        int w2s_next = ((ob_c + (consumer_lane + 1) * w2s_ranks) < (w2s_blocks - 1) ? (ob_c + (consumer_lane + 1) * w2s_ranks) : (w2s_blocks - 1));
                        w2_scales = w2_scale[(w2s_base + w2s_next) * intermediate_blocks_c + intermediate_block_c];
                    }
                    w2s_count = w2s_count + 1;
                    #pragma unroll
                    for (int token_slot_out = 0; token_slot_out < 8; token_slot_out++) {
                        float out_scale = down_token_scale[token_slot_out] * w2s;
                        smem_out[((int)down_stage_c * 8 + token_slot_out) * 128 + physical_row] = _tmem_load_1[token_slot_out] * out_scale;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            #pragma unroll
                            for (int token_slot_reduce = 0; token_slot_reduce < 8; token_slot_reduce++) {
                                int pair_reduce = route_pairs[token_slot_reduce];
                                if (pair_reduce < M * top_k) {
                                    int token_reduce = pair_reduce / top_k;
                                    {
                                        void* _cpred_dst_22 = reinterpret_cast<void*>(out + (token_reduce * K + ob_c * 128));
                                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"
                                            " [%0], [%1], %2;"
                                            :: "l"(_cpred_dst_22), "r"(smem_out_addr + (unsigned int)(((int)down_stage_c * 8 + token_slot_reduce) * 128 * 2)), "r"((uint32_t)(256))
                                            : "memory");
                                    }
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    down_stage_c += 1;
                    if (down_stage_c == 2) { down_stage_c = 0; _phase_down_ready ^= 1; }
                }
                if (warp == 2) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
    }
}

} // extern "C"

#undef CAKE_INF
#undef TMEM_NCOLS
#undef TMEM_UP_ACC_OFFSET
#undef TMEM_DOWN_ACC_OFFSET
#undef NUM_UP_PIPE_STAGES
#undef NUM_DOWN_PIPE_STAGES
#undef SMEM_SMEM_W1_OFF
#undef SMEM_SMEM_W1_STAGE_BYTES
#undef SMEM_SMEM_W1_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_SMEM_W2_OFF
#undef SMEM_SMEM_W2_STAGE_BYTES
#undef SMEM_SMEM_W2_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_AMAX_OFF
#undef SMEM_SMEM_AMAX_STAGE_BYTES
#undef SMEM_SMEM_AMAX_STRIDE
#undef SMEM_SMEM_ACT_SCALE_OFF
#undef SMEM_SMEM_ACT_SCALE_STAGE_BYTES
#undef SMEM_SMEM_ACT_SCALE_STRIDE
#undef SMEM_SMEM_PEER_AMAX_OFF
#undef SMEM_SMEM_PEER_AMAX_STAGE_BYTES
#undef SMEM_SMEM_PEER_AMAX_STRIDE
#undef SMEM_SMEM_OUT_OFF
#undef SMEM_SMEM_OUT_STAGE_BYTES
#undef SMEM_SMEM_OUT_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef LAUNCH_MIN_BLOCKS
#undef up_full_addr
#undef up_ready_addr
#undef up_free_addr
#undef up_empty_addr
#undef down_full_addr
#undef down_ready_addr
#undef down_free_addr
#undef down_empty_addr
#undef amax_ready_addr
#undef act_ready_addr
#undef kernel_alpha_moe_w8a8_up_down
constexpr int kGeneratedThreads = 192;
constexpr int kGeneratedSmemTotal = 108672;
}  // namespace w8a8_p2

namespace w8a8_pl {
#define kernel_alpha_moe_w8a8_up_down kernel_alpha_moe_w8a8_up_down_w8a8_pl
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
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else

#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 48
#define TMEM_UP_ACC_OFFSET 0
#define TMEM_DOWN_ACC_OFFSET 32
#define NUM_UP_PIPE_STAGES 4
#define NUM_DOWN_PIPE_STAGES 2
#define SMEM_SMEM_W1_OFF 1024
#define SMEM_SMEM_W1_STAGE_BYTES 16384
#define SMEM_SMEM_W1_STRIDE 17408
#define SMEM_SMEM_X_OFF 17408
#define SMEM_SMEM_X_STAGE_BYTES 1024
#define SMEM_SMEM_X_STRIDE 17408
#define SMEM_SMEM_W2_OFF 70656
#define SMEM_SMEM_W2_STAGE_BYTES 16384
#define SMEM_SMEM_W2_STRIDE 16384
#define SMEM_SMEM_ACT_OFF 103424
#define SMEM_SMEM_ACT_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_STRIDE 1024
#define SMEM_SMEM_AMAX_OFF 104448
#define SMEM_SMEM_AMAX_STAGE_BYTES 32
#define SMEM_SMEM_AMAX_STRIDE 32
#define SMEM_SMEM_ACT_SCALE_OFF 104480
#define SMEM_SMEM_ACT_SCALE_STAGE_BYTES 32
#define SMEM_SMEM_ACT_SCALE_STRIDE 32
#define SMEM_SMEM_PEER_AMAX_OFF 104512
#define SMEM_SMEM_PEER_AMAX_STAGE_BYTES 32
#define SMEM_SMEM_PEER_AMAX_STRIDE 32
#define SMEM_SMEM_OUT_OFF 104544
#define SMEM_SMEM_OUT_STAGE_BYTES 4096
#define SMEM_SMEM_OUT_STRIDE 4096
#define SMEM_TOTAL 108672
#define THREADS 192
#define LAUNCH_MIN_BLOCKS 2

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

__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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

__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
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

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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
    // Non-multicast, single-CTA gather4 in the .shared::cta window (the
    // cuda::ptx form): the destination and the barrier are the executing
    // CTA's. A CTA pair's gather takes the .shared::cluster window
    // (tma_gather4_gmem2smem_cta2), its barrier being the pair leader's.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) __cluster_dims__(1,1,2) void
kernel_alpha_moe_w8a8_up_down(const __grid_constant__ CUtensorMap x, const __grid_constant__ CUtensorMap W1, float* __restrict__ x_scale, float* __restrict__ w1_scale, const __grid_constant__ CUtensorMap W2, float* __restrict__ w2_scale, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, float* __restrict__ topk_weights, __nv_bfloat16* __restrict__ out, int M, int K, int top_k, int route_block_m, float scaling_factor)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define up_full_addr (mbar_base + 0)
    #define up_ready_addr (mbar_base + 32)
    #define up_free_addr (mbar_base + 64)
    #define down_full_addr (mbar_base + 96)
    #define down_ready_addr (mbar_base + 112)
    #define down_free_addr (mbar_base + 128)
    #define amax_ready_addr (mbar_base + 144)
    #define act_ready_addr (mbar_base + 152)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = (((blockIdx.z / 2) * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * gridDim.y * (gridDim.z / 2);

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_w1 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w1_addr = smem + 1024;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_x_addr = smem + 17408;
    uint8_t* smem_w2 = reinterpret_cast<uint8_t*>(smem_raw + 70656);
    const int smem_w2_addr = smem + 70656;
    uint8_t* smem_act = reinterpret_cast<uint8_t*>(smem_raw + 103424);
    const int smem_act_addr = smem + 103424;
    unsigned int* smem_amax = reinterpret_cast<unsigned int*>(smem_raw + 104448);
    const int smem_amax_addr = smem + 104448;
    float* smem_act_scale = reinterpret_cast<float*>(smem_raw + 104480);
    const int smem_act_scale_addr = smem + 104480;
    unsigned int* smem_peer_amax = reinterpret_cast<unsigned int*>(smem_raw + 104512);
    const int smem_peer_amax_addr = smem + 104512;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 104544);
    const int smem_out_addr = smem + 104544;
    if (blockIdx.x * 8 >= num_tokens_post_padded[0]) {
        return;
    }

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'up_pipe' ---
            // up_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // up_ready: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // up_free: 4 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            // --- pipeline 'down_pipe' ---
            // down_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // down_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // down_free: 2 barriers, init_count=4
            mbarrier_init(smem + 128, 4);
            mbarrier_init(smem + 136, 4);
            // amax_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // act_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 144, 32);
            mbarrier_expect_tx(smem + 152, 512);
        }
    }

    __syncwarp();

    // TMEM alloc (64 columns, 48 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
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
    const int tmem_up_acc = taddr;
    const int tmem_down_acc = taddr + 32;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int _phase_up_free = 1;
            unsigned int _phase_down_free = 1;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int route_work = blockIdx.x;
                int intermediate_block = blockIdx.y;
                int subtiles_per_route_block = route_block_m / 8;
                int route_block = route_work / subtiles_per_route_block;
                int route_subtile = route_work % subtiles_per_route_block;
                int route_base = route_block * route_block_m + route_subtile * 8;
                bool route_active = route_base < num_tokens_post_padded[0];
                int expert = 0;
                if (route_active) {
                    expert = expert_ids[route_block];
                }
                int k_blocks = K / 128;
                const int panel = cta_rank;
                int pair0 = M * top_k;
                int pair1 = M * top_k;
                int pair2 = M * top_k;
                int pair3 = M * top_k;
                int pair4 = M * top_k;
                int pair5 = M * top_k;
                int pair6 = M * top_k;
                int pair7 = M * top_k;
                if (route_active) {
                    pair0 = sorted_token_ids[route_base];
                    pair1 = sorted_token_ids[route_base + 1];
                    pair2 = sorted_token_ids[route_base + 2];
                    pair3 = sorted_token_ids[route_base + 3];
                    pair4 = sorted_token_ids[route_base + 4];
                    pair5 = sorted_token_ids[route_base + 5];
                    pair6 = sorted_token_ids[route_base + 6];
                    pair7 = sorted_token_ids[route_base + 7];
                }
                int row0 = ((pair0 / top_k) < (M - 1) ? (pair0 / top_k) : (M - 1));
                int row1 = ((pair1 / top_k) < (M - 1) ? (pair1 / top_k) : (M - 1));
                int row2 = ((pair2 / top_k) < (M - 1) ? (pair2 / top_k) : (M - 1));
                int row3 = ((pair3 / top_k) < (M - 1) ? (pair3 / top_k) : (M - 1));
                int row4 = ((pair4 / top_k) < (M - 1) ? (pair4 / top_k) : (M - 1));
                int row5 = ((pair5 / top_k) < (M - 1) ? (pair5 / top_k) : (M - 1));
                int row6 = ((pair6 / top_k) < (M - 1) ? (pair6 / top_k) : (M - 1));
                int row7 = ((pair7 / top_k) < (M - 1) ? (pair7 / top_k) : (M - 1));
                unsigned int stage = 0;
                const int pf_rank = blockIdx.z;
                const int pf_ranks = gridDim.z;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int pf0 = 4; pf0 < 16; pf0++) {
                        if (k_blocks > pf0) {
                            asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&W1))), "r"((int)(0)), "r"((int)(intermediate_block * 256 + panel * 128)), "r"((int)(pf0)), "r"((int)(expert)) : "memory");
                        }
                    }
                }
                #pragma unroll 1
                for (int kb = 0; kb < k_blocks; kb++) {
                    mbarrier_wait(up_free_addr + (stage) * 8, _phase_up_free);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(up_full_addr + (stage) * 8, 17408);
                        tma_gather4_gmem2smem(smem_x_addr + stage * 17408, (&x), kb * 128, row0, row1, row2, row3, up_full_addr + (stage) * 8);
                        tma_gather4_gmem2smem(smem_x_addr + stage * 17408 + 512, (&x), kb * 128, row4, row5, row6, row7, up_full_addr + (stage) * 8);
                        tma_4d_gmem2smem(smem_w1_addr + stage * 17408, (&W1), 0, intermediate_block * 256 + panel * 128, kb, expert, up_full_addr + (stage) * 8);
                        int pf_kb = kb + 16;
                        if (pf_kb < k_blocks) {
                            asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&W1))), "r"((int)(0)), "r"((int)(intermediate_block * 256 + panel * 128)), "r"((int)(pf_kb)), "r"((int)(expert)) : "memory");
                        }
                        int pf_ob = pf_rank + kb * pf_ranks;
                        if (pf_ob < K / 128) {
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W2))), "r"((int)(intermediate_block * 128)), "r"((int)(pf_ob * 128)), "r"((int)(expert)) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 4) { stage = 0; _phase_up_free ^= 1; }
                }
                unsigned int down_load_stage = 0;
                int out_blocks = K / 128;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int ob = blockIdx.z; ob < out_blocks; ob += gridDim.z) {
                        mbarrier_wait(down_free_addr + (down_load_stage) * 8, _phase_down_free);
                        mbarrier_arrive_expect_tx(down_full_addr + (down_load_stage) * 8, 16384);
                        tma_3d_gmem2smem(smem_w2_addr + down_load_stage * 16384, (&W2), intermediate_block * 128, ob * 128, expert, down_full_addr + (down_load_stage) * 8);
                        down_load_stage += 1;
                        if (down_load_stage == 2) { down_load_stage = 0; _phase_down_free ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int _phase_up_free_1 = 1;
            unsigned int _phase_up_full = 0;
            unsigned int _phase_down_free_1 = 1;
            unsigned int _phase_down_full = 0;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int k_blocks_mma = K / 128;
                unsigned int up_stage = 0;
                #pragma unroll 1
                for (int kb_mma = 0; kb_mma < k_blocks_mma; kb_mma++) {
                    mbarrier_wait(up_free_addr + (up_stage) * 8, _phase_up_free_1);
                    mbarrier_wait(up_full_addr + (up_stage) * 8, _phase_up_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_w1_addr) >> 4) & 0x3FFF) + (up_stage) * 1088);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (up_stage) * 1088);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 8)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                        }
                    }
                    elect_commit(up_ready_addr + (up_stage) * 8);
                    up_stage += 1;
                    if (up_stage == 4) { up_stage = 0; _phase_up_free_1 ^= 1; _phase_up_full ^= 1; }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 160;" ::: "memory");
                mbarrier_wait_cluster_hint(act_ready_addr, 0, 10000000);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int out_blocks_mma = K / 128;
                unsigned int down_stage = 0;
                #pragma unroll 1
                for (int ob_mma = blockIdx.z; ob_mma < out_blocks_mma; ob_mma += gridDim.z) {
                    mbarrier_wait(down_free_addr + (down_stage) * 8, _phase_down_free_1);
                    mbarrier_wait(down_full_addr + (down_stage) * 8, _phase_down_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_1 = make_warp_uniform((((smem_w2_addr) >> 4) & 0x3FFF) + (down_stage) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform(((smem_act_addr) >> 4) & 0x3FFF);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                        }
                    }
                    elect_commit(down_ready_addr + (down_stage) * 8);
                    down_stage += 1;
                    if (down_stage == 2) { down_stage = 0; _phase_down_free_1 ^= 1; _phase_down_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: consumer ----
    if (warp >= 2 && warp <= 5) {
        { // consumer_main
            unsigned int _phase_up_ready = 0;
            unsigned int _phase_down_ready = 0;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int route_work_c = blockIdx.x;
                int intermediate_blocks_c = gridDim.y;
                int intermediate_block_c = blockIdx.y;
                int subtiles_per_route_block_c = route_block_m / 8;
                int route_block_c = route_work_c / subtiles_per_route_block_c;
                int route_subtile_c = route_work_c % subtiles_per_route_block_c;
                int route_base_c = route_block_c * route_block_m + route_subtile_c * 8;
                bool route_active_c = route_base_c < num_tokens_post_padded[0];
                int expert_c = 0;
                if (route_active_c) {
                    expert_c = expert_ids[route_block_c];
                }
                int k_blocks_c = K / 128;
                const int consumer_lane = tid % 32;
                const int consumer_warp = warp % 4;
                const int consumer_tid = consumer_warp * 32 + consumer_lane;
                const int physical_row = consumer_warp * 32 + consumer_lane;
                const int scale_row = physical_row / 8 % 2;
                int lane_pair = M * top_k;
                int lane_token = M - 1;
                float lane_route_weight = 0.0f;
                if (consumer_lane < 8 && route_active_c) {
                    lane_pair = sorted_token_ids[route_base_c + consumer_lane];
                    int _min_0 = ((lane_pair / top_k) < (M - 1) ? (lane_pair / top_k) : (M - 1));
                    lane_token = _min_0;
                    if (lane_pair < M * top_k) {
                        lane_route_weight = topk_weights[lane_pair];
                    }
                }
                int route_pairs[8];
                float route_weights[8];
                #pragma unroll
                for (int token_slot_route = 0; token_slot_route < 8; token_slot_route++) {
                    int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, lane_pair, token_slot_route);
                    route_pairs[token_slot_route] = _shfl_sync_1;
                    float _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, lane_route_weight, token_slot_route);
                    route_weights[token_slot_route] = _shfl_sync_2;
                }
                const int panel_c = cta_rank;
                float acc[8];
                #pragma unroll
                for (int token_slot_init = 0; token_slot_init < 8; token_slot_init++) {
                    acc[token_slot_init] = 0.0f;
                }
                int w1_scale_row = (expert_c * intermediate_blocks_c * 2 + intermediate_block_c * 2 + consumer_lane) * k_blocks_c;
                const int scale_slot = consumer_lane % 16;
                const int w1_scale_half = (expert_c * intermediate_blocks_c * 2 + intermediate_block_c * 2 + consumer_lane / 16) * k_blocks_c;
                int x_scale_rows[8];
                float x_scales[8];
                float x_scales_next[8];
                int _min_3 = ((scale_slot) < (k_blocks_c - 1) ? (scale_slot) : (k_blocks_c - 1));
                const int first_scale_kb = _min_3;
                #pragma unroll
                for (int token_slot_xs = 0; token_slot_xs < 8; token_slot_xs++) {
                    int _min_4 = ((route_pairs[token_slot_xs] / top_k) < (M - 1) ? (route_pairs[token_slot_xs] / top_k) : (M - 1));
                    x_scale_rows[token_slot_xs] = _min_4 * k_blocks_c;
                    x_scales[token_slot_xs] = 0.0f;
                    x_scales_next[token_slot_xs] = x_scale[x_scale_rows[token_slot_xs] + first_scale_kb];
                }
                float w1_scales = 0.0f;
                float w1_scales_next = w1_scale[w1_scale_half + first_scale_kb];
                if (consumer_tid < 8) {
                    smem_amax[consumer_tid] = 0;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                unsigned int up_stage_c = 0;
                #pragma unroll 1
                for (int kb_c = 0; kb_c < k_blocks_c; kb_c++) {
                    mbarrier_wait(up_ready_addr + (up_stage_c) * 8, _phase_up_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int row_base = consumer_warp * 32;
                    int panel_addr = taddr + (unsigned int)(row_base << 16) + up_stage_c * 8;
                    float _tmem_load_0[8];
                    tmem_ld_x8(&_tmem_load_0[0], panel_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    int slot_kb = kb_c % 16;
                    if (slot_kb == 0) {
                        #pragma unroll
                        for (int token_slot_cp = 0; token_slot_cp < 8; token_slot_cp++) {
                            x_scales[token_slot_cp] = x_scales_next[token_slot_cp];
                        }
                        w1_scales = w1_scales_next;
                        int next_kb = ((kb_c + 16 + scale_slot) < (k_blocks_c - 1) ? (kb_c + 16 + scale_slot) : (k_blocks_c - 1));
                        #pragma unroll
                        for (int token_slot_nx = 0; token_slot_nx < 8; token_slot_nx++) {
                            x_scales_next[token_slot_nx] = x_scale[x_scale_rows[token_slot_nx] + next_kb];
                        }
                        w1_scales_next = w1_scale[w1_scale_half + next_kb];
                    }
                    float wscale = __shfl_sync(0xFFFFFFFF, w1_scales, slot_kb + 16 * scale_row);
                    #pragma unroll
                    for (int token_slot_acc = 0; token_slot_acc < 8; token_slot_acc++) {
                        float xscale = __shfl_sync(0xFFFFFFFF, x_scales[token_slot_acc], slot_kb);
                        float scale = xscale * wscale;
                        acc[token_slot_acc] = __fmaf_rn(_tmem_load_0[token_slot_acc], scale, acc[token_slot_acc]);
                    }
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(up_free_addr + (up_stage_c) * 8);
                    }
                    up_stage_c += 1;
                    if (up_stage_c == 4) { up_stage_c = 0; _phase_up_ready ^= 1; }
                }
                const int w2s_blocks = K / 128;
                const int w2s_rank = blockIdx.z;
                const int w2s_ranks = gridDim.z;
                const int w2s_base = expert_c * w2s_blocks;
                int _min_5 = ((w2s_rank + consumer_lane * w2s_ranks) < (w2s_blocks - 1) ? (w2s_rank + consumer_lane * w2s_ranks) : (w2s_blocks - 1));
                const int w2s_first = _min_5;
                float w2_scales = w2_scale[(w2s_base + w2s_first) * intermediate_blocks_c + intermediate_block_c];
                int w2s_count = 0;
                float act[8];
                unsigned int amax_words[8];
                bool gate_lane = consumer_lane % 16 < 8;
                #pragma unroll
                for (int token_slot_act = 0; token_slot_act < 8; token_slot_act++) {
                    float peer = __shfl_xor_sync(0xFFFFFFFF, acc[token_slot_act], 8);
                    float gate = acc[token_slot_act];
                    float _exp_6 = __expf(-gate);
                    float sig = approx_rcp(1.0f + _exp_6);
                    act[token_slot_act] = gate * sig * peer;
                    unsigned int gate_word = 0;
                    if (gate_lane) {
                        uint32_t _amf_u_8 = __float_as_uint(fabsf(act[token_slot_act]));
                        uint32_t _amf_mask_8 = -int32_t(_amf_u_8 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_7 = _amf_u_8 ^ _amf_mask_8;
                        gate_word = _atomic_max_float_encode_7;
                    }
                    unsigned int _warp_redux_9;
                    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_9) : "r"(gate_word));
                    amax_words[token_slot_act] = _warp_redux_9;
                }
                if (consumer_lane < 8) {
                    unsigned int lane_word = amax_words[0];
                    #pragma unroll
                    for (int token_slot_sel = 1; token_slot_sel < 8; token_slot_sel++) {
                        if (consumer_lane == token_slot_sel) {
                            lane_word = amax_words[token_slot_sel];
                        }
                    }
                    atomicMax(&smem_amax[consumer_lane], lane_word);
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (consumer_tid < 8) {
                    unsigned int local_amax = smem_amax[consumer_tid];
                    if (panel_c != 0) {
                        uint32_t _cluster_mapa_10;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_10) : "r"(smem_peer_amax_addr + (unsigned int)(consumer_tid * 4)), "r"(0));
                        uint32_t _mbar_peer_11;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_11) : "r"(amax_ready_addr), "r"(0));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_10), "r"(local_amax), "r"(_mbar_peer_11) : "memory");
                    }
                    if (panel_c != 1) {
                        uint32_t _cluster_mapa_12;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_12) : "r"(smem_peer_amax_addr + (unsigned int)(consumer_tid * 4)), "r"(1));
                        uint32_t _mbar_peer_13;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_13) : "r"(amax_ready_addr), "r"(1));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_12), "r"(local_amax), "r"(_mbar_peer_13) : "memory");
                    }
                }
                if (warp == 2) {
                    if (elect_sync()) {
                        mbarrier_arrive(amax_ready_addr);
                        mbarrier_arrive(act_ready_addr);
                    }
                }
                mbarrier_wait_cluster_hint(amax_ready_addr, 0, 10000000);
                if (consumer_tid < 8) {
                    unsigned int _max_14 = ((smem_amax[consumer_tid]) > (smem_peer_amax[consumer_tid]) ? (smem_amax[consumer_tid]) : (smem_peer_amax[consumer_tid]));
                    uint32_t _amf_u_15 = _max_14;
                    uint32_t _amf_mask_15 = ((_amf_u_15 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_15 ^ _amf_mask_15);
                    smem_act_scale[consumer_tid] = ((_amf_dec_0 == 0.0f) ? 1.0f : _amf_dec_0 / 448.0f);
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                const int feature = panel_c * 64 + physical_row / 16 * 8 + physical_row % 8;
                bool even_feature_lane = gate_lane && consumer_lane % 2 == 0;
                #pragma unroll
                for (int token_slot_quant = 0; token_slot_quant < 8; token_slot_quant++) {
                    float inv_scale = approx_rcp(smem_act_scale[token_slot_quant]);
                    float peer_feature = __shfl_xor_sync(0xFFFFFFFF, act[token_slot_quant], 1);
                    if (even_feature_lane) {
                        {
                            uint16_t _fp8_pair_16;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_16) : "f"(0.0f), "f"(act[token_slot_quant] * inv_scale));
                            uint32_t _byte_16 = (uint32_t)(_fp8_pair_16 & 0xFF);
                            uint32_t _addr_16 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + feature ^ (token_slot_quant * 128 + feature >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_16), "r"(_byte_16) : "memory");
                        }
                        {
                            uint16_t _fp8_pair_17;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_17) : "f"(0.0f), "f"(peer_feature * inv_scale));
                            uint32_t _byte_17 = (uint32_t)(_fp8_pair_17 & 0xFF);
                            uint32_t _addr_17 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + (feature + 1) ^ (token_slot_quant * 128 + (feature + 1) >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_17), "r"(_byte_17) : "memory");
                        }
                    }
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (consumer_warp == 0) {
                    unsigned int chunk_addr = (unsigned int)(smem_act_addr + (unsigned int)(consumer_lane / 4 * 128 + (panel_c * 4 + consumer_lane % 4) * 16 ^ (consumer_lane / 4 * 128 + (panel_c * 4 + consumer_lane % 4) * 16 >> 7 & 7) << 4));
                    unsigned int chunk[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&chunk[0])), "=r"(*reinterpret_cast<uint32_t*>(&chunk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&chunk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&chunk[(0) + 3]))
                        : "r"(chunk_addr));
                    if (panel_c != 0) {
                        uint32_t _cluster_mapa_18;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_18) : "r"(chunk_addr), "r"(0));
                        uint32_t _mbar_peer_19;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_19) : "r"(act_ready_addr), "r"(0));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_18), "r"(chunk[0]), "r"(chunk[1]), "r"(chunk[2]), "r"(chunk[3]), "r"(_mbar_peer_19) : "memory");
                    }
                    if (panel_c != 1) {
                        uint32_t _cluster_mapa_20;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_20) : "r"(chunk_addr), "r"(1));
                        uint32_t _mbar_peer_21;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_21) : "r"(act_ready_addr), "r"(1));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_20), "r"(chunk[0]), "r"(chunk[1]), "r"(chunk[2]), "r"(chunk[3]), "r"(_mbar_peer_21) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                asm volatile("barrier.sync 1, 160;" ::: "memory");
                float down_act_scale[8];
                float down_token_scale[8];
                #pragma unroll
                for (int token_slot_scale = 0; token_slot_scale < 8; token_slot_scale++) {
                    down_act_scale[token_slot_scale] = smem_act_scale[token_slot_scale];
                    int pair_scale = route_pairs[token_slot_scale];
                    bool valid_scale = pair_scale < M * top_k;
                    float route_weight_scale = 0.0f;
                    if (valid_scale) {
                        route_weight_scale = route_weights[token_slot_scale];
                    }
                    down_token_scale[token_slot_scale] = down_act_scale[token_slot_scale] * route_weight_scale * scaling_factor;
                }
                int out_blocks_c = K / 128;
                unsigned int down_stage_c = 0;
                #pragma unroll 1
                for (int ob_c = blockIdx.z; ob_c < out_blocks_c; ob_c += gridDim.z) {
                    mbarrier_wait(down_ready_addr + (down_stage_c) * 8, _phase_down_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int down_addr = taddr + 32 + (unsigned int)(consumer_warp * 32 << 16) + down_stage_c * 8;
                    float _tmem_load_1[8];
                    tmem_ld_x8(&_tmem_load_1[0], down_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(down_free_addr + (down_stage_c) * 8);
                    }
                    int w2s_slot = w2s_count % 32;
                    float w2s = __shfl_sync(0xFFFFFFFF, w2_scales, w2s_slot);
                    if (w2s_slot == 31) {
                        int w2s_next = ((ob_c + (consumer_lane + 1) * w2s_ranks) < (w2s_blocks - 1) ? (ob_c + (consumer_lane + 1) * w2s_ranks) : (w2s_blocks - 1));
                        w2_scales = w2_scale[(w2s_base + w2s_next) * intermediate_blocks_c + intermediate_block_c];
                    }
                    w2s_count = w2s_count + 1;
                    #pragma unroll
                    for (int token_slot_out = 0; token_slot_out < 8; token_slot_out++) {
                        float out_scale = down_token_scale[token_slot_out] * w2s;
                        smem_out[((int)down_stage_c * 8 + token_slot_out) * 128 + physical_row] = _tmem_load_1[token_slot_out] * out_scale;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            #pragma unroll
                            for (int token_slot_reduce = 0; token_slot_reduce < 8; token_slot_reduce++) {
                                int pair_reduce = route_pairs[token_slot_reduce];
                                if (pair_reduce < M * top_k) {
                                    int token_reduce = pair_reduce / top_k;
                                    {
                                        void* _cpred_dst_22 = reinterpret_cast<void*>(out + (token_reduce * K + ob_c * 128));
                                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"
                                            " [%0], [%1], %2;"
                                            :: "l"(_cpred_dst_22), "r"(smem_out_addr + (unsigned int)(((int)down_stage_c * 8 + token_slot_reduce) * 128 * 2)), "r"((uint32_t)(256))
                                            : "memory");
                                    }
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    down_stage_c += 1;
                    if (down_stage_c == 2) { down_stage_c = 0; _phase_down_ready ^= 1; }
                }
                if (warp == 2) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
    }
}

} // extern "C"

#undef CAKE_INF
#undef TMEM_NCOLS
#undef TMEM_UP_ACC_OFFSET
#undef TMEM_DOWN_ACC_OFFSET
#undef NUM_UP_PIPE_STAGES
#undef NUM_DOWN_PIPE_STAGES
#undef SMEM_SMEM_W1_OFF
#undef SMEM_SMEM_W1_STAGE_BYTES
#undef SMEM_SMEM_W1_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_SMEM_W2_OFF
#undef SMEM_SMEM_W2_STAGE_BYTES
#undef SMEM_SMEM_W2_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_AMAX_OFF
#undef SMEM_SMEM_AMAX_STAGE_BYTES
#undef SMEM_SMEM_AMAX_STRIDE
#undef SMEM_SMEM_ACT_SCALE_OFF
#undef SMEM_SMEM_ACT_SCALE_STAGE_BYTES
#undef SMEM_SMEM_ACT_SCALE_STRIDE
#undef SMEM_SMEM_PEER_AMAX_OFF
#undef SMEM_SMEM_PEER_AMAX_STAGE_BYTES
#undef SMEM_SMEM_PEER_AMAX_STRIDE
#undef SMEM_SMEM_OUT_OFF
#undef SMEM_SMEM_OUT_STAGE_BYTES
#undef SMEM_SMEM_OUT_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef LAUNCH_MIN_BLOCKS
#undef up_full_addr
#undef up_ready_addr
#undef up_free_addr
#undef down_full_addr
#undef down_ready_addr
#undef down_free_addr
#undef amax_ready_addr
#undef act_ready_addr
#undef kernel_alpha_moe_w8a8_up_down
constexpr int kGeneratedThreads = 192;
constexpr int kGeneratedSmemTotal = 108672;
}  // namespace w8a8_pl

namespace w8a8_im {
#define kernel_alpha_moe_w8a8_up_down kernel_alpha_moe_w8a8_up_down_w8a8_im
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
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else

#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 80
#define TMEM_UP_ACC_OFFSET 0
#define TMEM_DOWN_ACC_OFFSET 64
#define NUM_UP_PIPE_STAGES 4
#define NUM_DOWN_PIPE_STAGES 2
#define SMEM_SMEM_W1_OFF 1024
#define SMEM_SMEM_W1_STAGE_BYTES 8192
#define SMEM_SMEM_W1_STRIDE 18432
#define SMEM_SMEM_X_OFF 17408
#define SMEM_SMEM_X_STAGE_BYTES 1024
#define SMEM_SMEM_X_STRIDE 18432
#define SMEM_SMEM_W1B_OFF 9216
#define SMEM_SMEM_W1B_STAGE_BYTES 8192
#define SMEM_SMEM_W1B_STRIDE 18432
#define SMEM_SMEM_XB_OFF 18432
#define SMEM_SMEM_XB_STAGE_BYTES 1024
#define SMEM_SMEM_XB_STRIDE 18432
#define SMEM_SMEM_W2_OFF 74752
#define SMEM_SMEM_W2_STAGE_BYTES 16384
#define SMEM_SMEM_W2_STRIDE 16384
#define SMEM_SMEM_ACT_OFF 107520
#define SMEM_SMEM_ACT_STAGE_BYTES 1024
#define SMEM_SMEM_ACT_STRIDE 1024
#define SMEM_SMEM_AMAX_OFF 108544
#define SMEM_SMEM_AMAX_STAGE_BYTES 32
#define SMEM_SMEM_AMAX_STRIDE 32
#define SMEM_SMEM_ACT_SCALE_OFF 108576
#define SMEM_SMEM_ACT_SCALE_STAGE_BYTES 32
#define SMEM_SMEM_ACT_SCALE_STRIDE 32
#define SMEM_SMEM_PEER_AMAX_OFF 108608
#define SMEM_SMEM_PEER_AMAX_STAGE_BYTES 128
#define SMEM_SMEM_PEER_AMAX_STRIDE 128
#define SMEM_SMEM_OUT_OFF 108736
#define SMEM_SMEM_OUT_STAGE_BYTES 4096
#define SMEM_SMEM_OUT_STRIDE 4096
#define SMEM_TOTAL 112896
#define THREADS 192
#define LAUNCH_MIN_BLOCKS 2

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

__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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

__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
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

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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
    // Non-multicast, single-CTA gather4 in the .shared::cta window (the
    // cuda::ptx form): the destination and the barrier are the executing
    // CTA's. A CTA pair's gather takes the .shared::cluster window
    // (tma_gather4_gmem2smem_cta2), its barrier being the pair leader's.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) __cluster_dims__(1,1,4) void
kernel_alpha_moe_w8a8_up_down(const __grid_constant__ CUtensorMap x, const __grid_constant__ CUtensorMap W1, float* __restrict__ x_scale, float* __restrict__ w1_scale, const __grid_constant__ CUtensorMap W2, float* __restrict__ w2_scale, int* __restrict__ sorted_token_ids, int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_padded, float* __restrict__ topk_weights, __nv_bfloat16* __restrict__ out, int M, int K, int top_k, int route_block_m, float scaling_factor)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define up_full_addr (mbar_base + 0)
    #define up_ready_addr (mbar_base + 32)
    #define up_free_addr (mbar_base + 64)
    #define up_empty_addr (mbar_base + 96)
    #define down_full_addr (mbar_base + 128)
    #define down_ready_addr (mbar_base + 144)
    #define down_free_addr (mbar_base + 160)
    #define down_empty_addr (mbar_base + 176)
    #define amax_ready_addr (mbar_base + 192)
    #define act_ready_addr (mbar_base + 200)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = (((blockIdx.z / 4) * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * gridDim.y * (gridDim.z / 4);

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_w1 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w1_addr = smem + 1024;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_x_addr = smem + 17408;
    uint8_t* smem_w1b = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_w1b_addr = smem + 9216;
    uint8_t* smem_xb = reinterpret_cast<uint8_t*>(smem_raw + 18432);
    const int smem_xb_addr = smem + 18432;
    uint8_t* smem_w2 = reinterpret_cast<uint8_t*>(smem_raw + 74752);
    const int smem_w2_addr = smem + 74752;
    uint8_t* smem_act = reinterpret_cast<uint8_t*>(smem_raw + 107520);
    const int smem_act_addr = smem + 107520;
    unsigned int* smem_amax = reinterpret_cast<unsigned int*>(smem_raw + 108544);
    const int smem_amax_addr = smem + 108544;
    float* smem_act_scale = reinterpret_cast<float*>(smem_raw + 108576);
    const int smem_act_scale_addr = smem + 108576;
    unsigned int* smem_peer_amax = reinterpret_cast<unsigned int*>(smem_raw + 108608);
    const int smem_peer_amax_addr = smem + 108608;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 108736);
    const int smem_out_addr = smem + 108736;
    if (blockIdx.x * 8 >= num_tokens_post_padded[0]) {
        return;
    }

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'up_pipe' ---
            // up_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // up_ready: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // up_free: 4 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            // up_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // --- pipeline 'down_pipe' ---
            // down_full: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // down_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // down_free: 2 barriers, init_count=4
            mbarrier_init(smem + 160, 4);
            mbarrier_init(smem + 168, 4);
            // down_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // amax_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // act_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 192, 96);
            mbarrier_expect_tx(smem + 200, 768);
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 80 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 0) {
        int _tmem_hold = smem + 208;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_up_acc = taddr;
    const int tmem_down_acc = taddr + 64;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int _phase_up_empty = 1;
            unsigned int _phase_down_empty = 1;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int route_work = blockIdx.x;
                int intermediate_block = blockIdx.y;
                int subtiles_per_route_block = route_block_m / 8;
                int route_block = route_work / subtiles_per_route_block;
                int route_subtile = route_work % subtiles_per_route_block;
                int route_base = route_block * route_block_m + route_subtile * 8;
                bool route_active = route_base < num_tokens_post_padded[0];
                int expert = 0;
                if (route_active) {
                    expert = expert_ids[route_block];
                }
                int k_blocks = K / 128;
                const int panel = cta_rank;
                int pair0 = M * top_k;
                int pair1 = M * top_k;
                int pair2 = M * top_k;
                int pair3 = M * top_k;
                int pair4 = M * top_k;
                int pair5 = M * top_k;
                int pair6 = M * top_k;
                int pair7 = M * top_k;
                if (route_active) {
                    pair0 = sorted_token_ids[route_base];
                    pair1 = sorted_token_ids[route_base + 1];
                    pair2 = sorted_token_ids[route_base + 2];
                    pair3 = sorted_token_ids[route_base + 3];
                    pair4 = sorted_token_ids[route_base + 4];
                    pair5 = sorted_token_ids[route_base + 5];
                    pair6 = sorted_token_ids[route_base + 6];
                    pair7 = sorted_token_ids[route_base + 7];
                }
                int row0 = ((pair0 / top_k) < (M - 1) ? (pair0 / top_k) : (M - 1));
                int row1 = ((pair1 / top_k) < (M - 1) ? (pair1 / top_k) : (M - 1));
                int row2 = ((pair2 / top_k) < (M - 1) ? (pair2 / top_k) : (M - 1));
                int row3 = ((pair3 / top_k) < (M - 1) ? (pair3 / top_k) : (M - 1));
                int row4 = ((pair4 / top_k) < (M - 1) ? (pair4 / top_k) : (M - 1));
                int row5 = ((pair5 / top_k) < (M - 1) ? (pair5 / top_k) : (M - 1));
                int row6 = ((pair6 / top_k) < (M - 1) ? (pair6 / top_k) : (M - 1));
                int row7 = ((pair7 / top_k) < (M - 1) ? (pair7 / top_k) : (M - 1));
                unsigned int stage = 0;
                const int pf_rank = blockIdx.z;
                const int pf_ranks = gridDim.z;
                bool l2_on = 1;
                l2_on = num_tokens_post_padded[0] <= 192;
                if (l2_on) {
                    if (elect_sync()) {
                        #pragma unroll 1
                        for (int pf0 = 8; pf0 < 16; pf0++) {
                            if (k_blocks > pf0) {
                                asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&W1))), "r"((int)(0)), "r"((int)(intermediate_block * 256 + panel * 64)), "r"((int)(pf0)), "r"((int)(expert)) : "memory");
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int ks = 0; ks < (k_blocks + 1) / 2; ks++) {
                        mbarrier_wait(up_empty_addr + (stage) * 8, _phase_up_empty);
                        if (elect_sync()) {
                            int kb_second = ((ks * 2 + 1) < (k_blocks - 1) ? (ks * 2 + 1) : (k_blocks - 1));
                            mbarrier_arrive_expect_tx(up_full_addr + (stage) * 8, 18432);
                            tma_gather4_gmem2smem(smem_x_addr + stage * 18432, (&x), ks * 2 * 128, row0, row1, row2, row3, up_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(smem_x_addr + stage * 18432 + 512, (&x), ks * 2 * 128, row4, row5, row6, row7, up_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(smem_xb_addr + stage * 18432, (&x), kb_second * 128, row0, row1, row2, row3, up_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(smem_xb_addr + stage * 18432 + 512, (&x), kb_second * 128, row4, row5, row6, row7, up_full_addr + (stage) * 8);
                            tma_4d_gmem2smem(smem_w1_addr + stage * 18432, (&W1), 0, intermediate_block * 256 + panel * 64, ks * 2, expert, up_full_addr + (stage) * 8);
                            tma_4d_gmem2smem(smem_w1b_addr + stage * 18432, (&W1), 0, intermediate_block * 256 + panel * 64, kb_second, expert, up_full_addr + (stage) * 8);
                            int pf_kb = ks * 2 + 16;
                            if (pf_kb < k_blocks) {
                                asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&W1))), "r"((int)(0)), "r"((int)(intermediate_block * 256 + panel * 64)), "r"((int)(pf_kb)), "r"((int)(expert)) : "memory");
                            }
                            if (k_blocks > pf_kb + 1) {
                                asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&W1))), "r"((int)(0)), "r"((int)(intermediate_block * 256 + panel * 64)), "r"((int)(pf_kb + 1)), "r"((int)(expert)) : "memory");
                            }
                            int pf_ob = pf_rank + ks * pf_ranks;
                            if (pf_ob < K / 128) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&W2))), "r"((int)(intermediate_block * 128)), "r"((int)(pf_ob * 128)), "r"((int)(expert)) : "memory");
                            }
                        }
                        stage += 1;
                        if (stage == 4) { stage = 0; _phase_up_empty ^= 1; }
                    }
                } else {
                    #pragma unroll 1
                    for (int ks_1 = 0; ks_1 < (k_blocks + 1) / 2; ks_1++) {
                        mbarrier_wait(up_empty_addr + (stage) * 8, _phase_up_empty);
                        if (elect_sync()) {
                            int kb_second_1 = ((ks_1 * 2 + 1) < (k_blocks - 1) ? (ks_1 * 2 + 1) : (k_blocks - 1));
                            mbarrier_arrive_expect_tx(up_full_addr + (stage) * 8, 18432);
                            tma_gather4_gmem2smem(smem_x_addr + stage * 18432, (&x), ks_1 * 2 * 128, row0, row1, row2, row3, up_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(smem_x_addr + stage * 18432 + 512, (&x), ks_1 * 2 * 128, row4, row5, row6, row7, up_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(smem_xb_addr + stage * 18432, (&x), kb_second_1 * 128, row0, row1, row2, row3, up_full_addr + (stage) * 8);
                            tma_gather4_gmem2smem(smem_xb_addr + stage * 18432 + 512, (&x), kb_second_1 * 128, row4, row5, row6, row7, up_full_addr + (stage) * 8);
                            tma_4d_gmem2smem(smem_w1_addr + stage * 18432, (&W1), 0, intermediate_block * 256 + panel * 64, ks_1 * 2, expert, up_full_addr + (stage) * 8);
                            tma_4d_gmem2smem(smem_w1b_addr + stage * 18432, (&W1), 0, intermediate_block * 256 + panel * 64, kb_second_1, expert, up_full_addr + (stage) * 8);
                        }
                        stage += 1;
                        if (stage == 4) { stage = 0; _phase_up_empty ^= 1; }
                    }
                }
                unsigned int down_load_stage = 0;
                int out_blocks = K / 128;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int ob = blockIdx.z; ob < out_blocks; ob += gridDim.z) {
                        mbarrier_wait(down_empty_addr + (down_load_stage) * 8, _phase_down_empty);
                        mbarrier_arrive_expect_tx(down_full_addr + (down_load_stage) * 8, 16384);
                        tma_3d_gmem2smem(smem_w2_addr + down_load_stage * 16384, (&W2), intermediate_block * 128, ob * 128, expert, down_full_addr + (down_load_stage) * 8);
                        down_load_stage += 1;
                        if (down_load_stage == 2) { down_load_stage = 0; _phase_down_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int _phase_up_free = 1;
            unsigned int _phase_up_full = 0;
            unsigned int _phase_down_free = 1;
            unsigned int _phase_down_full = 0;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int k_blocks_mma = K / 128;
                unsigned int up_stage = 0;
                #pragma unroll 1
                for (int ks_mma = 0; ks_mma < (k_blocks_mma + 1) / 2; ks_mma++) {
                    mbarrier_wait(up_free_addr + (up_stage) * 8, _phase_up_free);
                    mbarrier_wait(up_full_addr + (up_stage) * 8, _phase_up_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_w1_addr) >> 4) & 0x3FFF) + (up_stage) * 1152);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_x_addr) >> 4) & 0x3FFF) + (up_stage) * 1152);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67239952, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67239952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67239952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 67239952, 1);
                        }
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((smem_w1b_addr) >> 4) & 0x3FFF) + (up_stage) * 1152);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_xb_addr) >> 4) & 0x3FFF) + (up_stage) * 1152);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67239952, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67239952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67239952, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_up_acc + (up_stage * 16 + 8)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 67239952, 1);
                        }
                    }
                    elect_commit(up_ready_addr + (up_stage) * 8);
                    elect_commit(up_empty_addr + (up_stage) * 8);
                    up_stage += 1;
                    if (up_stage == 4) { up_stage = 0; _phase_up_free ^= 1; _phase_up_full ^= 1; }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 160;" ::: "memory");
                mbarrier_wait_cluster_hint(act_ready_addr, 0, 10000000);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int out_blocks_mma = K / 128;
                unsigned int down_stage = 0;
                #pragma unroll 1
                for (int ob_mma = blockIdx.z; ob_mma < out_blocks_mma; ob_mma += gridDim.z) {
                    mbarrier_wait(down_free_addr + (down_stage) * 8, _phase_down_free);
                    mbarrier_wait(down_full_addr + (down_stage) * 8, _phase_down_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_2 = make_warp_uniform((((smem_w2_addr) >> 4) & 0x3FFF) + (down_stage) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform(((smem_act_addr) >> 4) & 0x3FFF);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_down_acc + (down_stage * 8)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                        }
                    }
                    elect_commit(down_ready_addr + (down_stage) * 8);
                    elect_commit(down_empty_addr + (down_stage) * 8);
                    down_stage += 1;
                    if (down_stage == 2) { down_stage = 0; _phase_down_free ^= 1; _phase_down_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: consumer ----
    if (warp >= 2 && warp <= 5) {
        { // consumer_main
            unsigned int _phase_up_ready = 0;
            unsigned int _phase_down_ready = 0;
            if (blockIdx.x * 8 < num_tokens_post_padded[0]) {
                int route_work_c = blockIdx.x;
                int intermediate_blocks_c = gridDim.y;
                int intermediate_block_c = blockIdx.y;
                int subtiles_per_route_block_c = route_block_m / 8;
                int route_block_c = route_work_c / subtiles_per_route_block_c;
                int route_subtile_c = route_work_c % subtiles_per_route_block_c;
                int route_base_c = route_block_c * route_block_m + route_subtile_c * 8;
                bool route_active_c = route_base_c < num_tokens_post_padded[0];
                int expert_c = 0;
                if (route_active_c) {
                    expert_c = expert_ids[route_block_c];
                }
                int k_blocks_c = K / 128;
                const int consumer_lane = tid % 32;
                const int consumer_warp = warp % 4;
                const int consumer_tid = consumer_warp * 32 + consumer_lane;
                const int physical_row = consumer_warp * 32 + consumer_lane;
                const int scale_row = consumer_lane / 8 % 2;
                int lane_pair = M * top_k;
                int lane_token = M - 1;
                float lane_route_weight = 0.0f;
                if (consumer_lane < 8 && route_active_c) {
                    lane_pair = sorted_token_ids[route_base_c + consumer_lane];
                    int _min_0 = ((lane_pair / top_k) < (M - 1) ? (lane_pair / top_k) : (M - 1));
                    lane_token = _min_0;
                    if (lane_pair < M * top_k) {
                        lane_route_weight = topk_weights[lane_pair];
                    }
                }
                int route_pairs[8];
                float route_weights[8];
                #pragma unroll
                for (int token_slot_route = 0; token_slot_route < 8; token_slot_route++) {
                    int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, lane_pair, token_slot_route);
                    route_pairs[token_slot_route] = _shfl_sync_1;
                    float _shfl_sync_2 = __shfl_sync(0xFFFFFFFF, lane_route_weight, token_slot_route);
                    route_weights[token_slot_route] = _shfl_sync_2;
                }
                const int panel_c = cta_rank;
                float acc[8];
                #pragma unroll
                for (int token_slot_init = 0; token_slot_init < 8; token_slot_init++) {
                    acc[token_slot_init] = 0.0f;
                }
                int w1_scale_row = (expert_c * intermediate_blocks_c * 2 + intermediate_block_c * 2 + consumer_lane) * k_blocks_c;
                const int scale_slot = consumer_lane % 16;
                const int w1_scale_half = (expert_c * intermediate_blocks_c * 2 + intermediate_block_c * 2 + consumer_lane / 16) * k_blocks_c;
                int x_scale_rows[8];
                float x_scales[8];
                float x_scales_next[8];
                int _min_3 = ((scale_slot) < (k_blocks_c - 1) ? (scale_slot) : (k_blocks_c - 1));
                const int first_scale_kb = _min_3;
                #pragma unroll
                for (int token_slot_xs = 0; token_slot_xs < 8; token_slot_xs++) {
                    int _min_4 = ((route_pairs[token_slot_xs] / top_k) < (M - 1) ? (route_pairs[token_slot_xs] / top_k) : (M - 1));
                    x_scale_rows[token_slot_xs] = _min_4 * k_blocks_c;
                    x_scales[token_slot_xs] = 0.0f;
                    x_scales_next[token_slot_xs] = x_scale[x_scale_rows[token_slot_xs] + first_scale_kb];
                }
                float w1_scales = 0.0f;
                float w1_scales_next = w1_scale[w1_scale_half + first_scale_kb];
                if (consumer_tid < 8) {
                    smem_amax[consumer_tid] = 0;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                unsigned int up_stage_c = 0;
                #pragma unroll 1
                for (int ks_c = 0; ks_c < (k_blocks_c + 1) / 2; ks_c++) {
                    mbarrier_wait(up_ready_addr + (up_stage_c) * 8, _phase_up_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int pack_addr = taddr + (unsigned int)(consumer_warp * 32 << 16) + up_stage_c * 16;
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], pack_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(up_free_addr + (up_stage_c) * 8);
                    }
                    up_stage_c += 1;
                    if (up_stage_c == 4) { up_stage_c = 0; _phase_up_ready ^= 1; }
                    #pragma unroll
                    for (int pack_half = 0; pack_half < 2; pack_half++) {
                        int kb_k = ks_c * 2 + pack_half;
                        if (kb_k < k_blocks_c) {
                            int slot_k = kb_k % 16;
                            if (slot_k == 0) {
                                #pragma unroll
                                for (int token_slot_nk_cp = 0; token_slot_nk_cp < 8; token_slot_nk_cp++) {
                                    x_scales[token_slot_nk_cp] = x_scales_next[token_slot_nk_cp];
                                }
                                w1_scales = w1_scales_next;
                                int next_kb_k = ((kb_k + 16 + scale_slot) < (k_blocks_c - 1) ? (kb_k + 16 + scale_slot) : (k_blocks_c - 1));
                                #pragma unroll
                                for (int token_slot_nk = 0; token_slot_nk < 8; token_slot_nk++) {
                                    x_scales_next[token_slot_nk] = x_scale[x_scale_rows[token_slot_nk] + next_kb_k];
                                }
                                w1_scales_next = w1_scale[w1_scale_half + next_kb_k];
                            }
                            float wscale_k = __shfl_sync(0xFFFFFFFF, w1_scales, slot_k + 16 * scale_row);
                            #pragma unroll
                            for (int token_slot_kr = 0; token_slot_kr < 8; token_slot_kr++) {
                                float xscale_kr = __shfl_sync(0xFFFFFFFF, x_scales[token_slot_kr], slot_k);
                                float scale_kr = xscale_kr * wscale_k;
                                acc[token_slot_kr] = __fmaf_rn(_tmem_load_0[pack_half * 8 + token_slot_kr], scale_kr, acc[token_slot_kr]);
                            }
                        }
                    }
                }
                const int w2s_blocks = K / 128;
                const int w2s_rank = blockIdx.z;
                const int w2s_ranks = gridDim.z;
                const int w2s_base = expert_c * w2s_blocks;
                int _min_5 = ((w2s_rank + consumer_lane * w2s_ranks) < (w2s_blocks - 1) ? (w2s_rank + consumer_lane * w2s_ranks) : (w2s_blocks - 1));
                const int w2s_first = _min_5;
                float w2_scales = w2_scale[(w2s_base + w2s_first) * intermediate_blocks_c + intermediate_block_c];
                int w2s_count = 0;
                float act[8];
                unsigned int amax_words[8];
                bool gate_lane = consumer_lane < 8;
                #pragma unroll
                for (int token_slot_act = 0; token_slot_act < 8; token_slot_act++) {
                    float peer = __shfl_xor_sync(0xFFFFFFFF, acc[token_slot_act], 8);
                    float gate = acc[token_slot_act];
                    float _exp_6 = __expf(-gate);
                    float sig = approx_rcp(1.0f + _exp_6);
                    act[token_slot_act] = gate * sig * peer;
                    unsigned int gate_word = 0;
                    if (gate_lane) {
                        uint32_t _amf_u_8 = __float_as_uint(fabsf(act[token_slot_act]));
                        uint32_t _amf_mask_8 = -int32_t(_amf_u_8 >> 31) | 0x80000000u;
                        unsigned int _atomic_max_float_encode_7 = _amf_u_8 ^ _amf_mask_8;
                        gate_word = _atomic_max_float_encode_7;
                    }
                    unsigned int _warp_redux_9;
                    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_9) : "r"(gate_word));
                    amax_words[token_slot_act] = _warp_redux_9;
                }
                if (consumer_lane < 8) {
                    unsigned int lane_word = amax_words[0];
                    #pragma unroll
                    for (int token_slot_sel = 1; token_slot_sel < 8; token_slot_sel++) {
                        if (consumer_lane == token_slot_sel) {
                            lane_word = amax_words[token_slot_sel];
                        }
                    }
                    atomicMax(&smem_amax[consumer_lane], lane_word);
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (consumer_tid < 8) {
                    unsigned int local_amax = smem_amax[consumer_tid];
                    if (panel_c != 0) {
                        uint32_t _cluster_mapa_10;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_10) : "r"(smem_peer_amax_addr + (unsigned int)((panel_c * 8 + consumer_tid) * 4)), "r"(0));
                        uint32_t _mbar_peer_11;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_11) : "r"(amax_ready_addr), "r"(0));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_10), "r"(local_amax), "r"(_mbar_peer_11) : "memory");
                    }
                    if (panel_c != 1) {
                        uint32_t _cluster_mapa_12;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_12) : "r"(smem_peer_amax_addr + (unsigned int)((panel_c * 8 + consumer_tid) * 4)), "r"(1));
                        uint32_t _mbar_peer_13;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_13) : "r"(amax_ready_addr), "r"(1));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_12), "r"(local_amax), "r"(_mbar_peer_13) : "memory");
                    }
                    if (panel_c != 2) {
                        uint32_t _cluster_mapa_14;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_14) : "r"(smem_peer_amax_addr + (unsigned int)((panel_c * 8 + consumer_tid) * 4)), "r"(2));
                        uint32_t _mbar_peer_15;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_15) : "r"(amax_ready_addr), "r"(2));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_14), "r"(local_amax), "r"(_mbar_peer_15) : "memory");
                    }
                    if (panel_c != 3) {
                        uint32_t _cluster_mapa_16;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_16) : "r"(smem_peer_amax_addr + (unsigned int)((panel_c * 8 + consumer_tid) * 4)), "r"(3));
                        uint32_t _mbar_peer_17;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_17) : "r"(amax_ready_addr), "r"(3));
                        asm volatile(
                            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                            :: "r"(_cluster_mapa_16), "r"(local_amax), "r"(_mbar_peer_17) : "memory");
                    }
                }
                if (warp == 2) {
                    if (elect_sync()) {
                        mbarrier_arrive(amax_ready_addr);
                        mbarrier_arrive(act_ready_addr);
                    }
                }
                mbarrier_wait_cluster_hint(amax_ready_addr, 0, 10000000);
                if (consumer_tid < 8) {
                    unsigned int merged = smem_amax[consumer_tid];
                    if (panel_c != 0) {
                        unsigned int _max_18 = ((merged) > (smem_peer_amax[consumer_tid]) ? (merged) : (smem_peer_amax[consumer_tid]));
                        merged = _max_18;
                    }
                    if (panel_c != 1) {
                        unsigned int _max_19 = ((merged) > (smem_peer_amax[8 + consumer_tid]) ? (merged) : (smem_peer_amax[8 + consumer_tid]));
                        merged = _max_19;
                    }
                    if (panel_c != 2) {
                        unsigned int _max_20 = ((merged) > (smem_peer_amax[16 + consumer_tid]) ? (merged) : (smem_peer_amax[16 + consumer_tid]));
                        merged = _max_20;
                    }
                    if (panel_c != 3) {
                        unsigned int _max_21 = ((merged) > (smem_peer_amax[24 + consumer_tid]) ? (merged) : (smem_peer_amax[24 + consumer_tid]));
                        merged = _max_21;
                    }
                    uint32_t _amf_u_22 = merged;
                    uint32_t _amf_mask_22 = ((_amf_u_22 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_22 ^ _amf_mask_22);
                    smem_act_scale[consumer_tid] = ((_amf_dec_0 == 0.0f) ? 1.0f : _amf_dec_0 / 448.0f);
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                const int feature = panel_c * 32 + consumer_warp * 8 + consumer_lane % 8;
                bool even_feature_lane = gate_lane && consumer_lane % 2 == 0;
                #pragma unroll
                for (int token_slot_quant = 0; token_slot_quant < 8; token_slot_quant++) {
                    float inv_scale = approx_rcp(smem_act_scale[token_slot_quant]);
                    float peer_feature = __shfl_xor_sync(0xFFFFFFFF, act[token_slot_quant], 1);
                    if (even_feature_lane) {
                        {
                            uint16_t _fp8_pair_23;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_23) : "f"(0.0f), "f"(act[token_slot_quant] * inv_scale));
                            uint32_t _byte_23 = (uint32_t)(_fp8_pair_23 & 0xFF);
                            uint32_t _addr_23 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + feature ^ (token_slot_quant * 128 + feature >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_23), "r"(_byte_23) : "memory");
                        }
                        {
                            uint16_t _fp8_pair_24;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                                : "=h"(_fp8_pair_24) : "f"(0.0f), "f"(peer_feature * inv_scale));
                            uint32_t _byte_24 = (uint32_t)(_fp8_pair_24 & 0xFF);
                            uint32_t _addr_24 = static_cast<uint32_t>((smem_act_addr + (unsigned int)(token_slot_quant * 128 + (feature + 1) ^ (token_slot_quant * 128 + (feature + 1) >> 7 & 7) << 4)));
                            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_24), "r"(_byte_24) : "memory");
                        }
                    }
                }
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                if (consumer_warp == 0 && consumer_lane < 16) {
                    unsigned int chunk_addr = (unsigned int)(smem_act_addr + (unsigned int)(consumer_lane / 2 * 128 + (panel_c * 2 + consumer_lane % 2) * 16 ^ (consumer_lane / 2 * 128 + (panel_c * 2 + consumer_lane % 2) * 16 >> 7 & 7) << 4));
                    unsigned int chunk[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&chunk[0])), "=r"(*reinterpret_cast<uint32_t*>(&chunk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&chunk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&chunk[(0) + 3]))
                        : "r"(chunk_addr));
                    if (panel_c != 0) {
                        uint32_t _cluster_mapa_25;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_25) : "r"(chunk_addr), "r"(0));
                        uint32_t _mbar_peer_26;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_26) : "r"(act_ready_addr), "r"(0));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_25), "r"(chunk[0]), "r"(chunk[1]), "r"(chunk[2]), "r"(chunk[3]), "r"(_mbar_peer_26) : "memory");
                    }
                    if (panel_c != 1) {
                        uint32_t _cluster_mapa_27;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_27) : "r"(chunk_addr), "r"(1));
                        uint32_t _mbar_peer_28;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_28) : "r"(act_ready_addr), "r"(1));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_27), "r"(chunk[0]), "r"(chunk[1]), "r"(chunk[2]), "r"(chunk[3]), "r"(_mbar_peer_28) : "memory");
                    }
                    if (panel_c != 2) {
                        uint32_t _cluster_mapa_29;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_29) : "r"(chunk_addr), "r"(2));
                        uint32_t _mbar_peer_30;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_30) : "r"(act_ready_addr), "r"(2));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_29), "r"(chunk[0]), "r"(chunk[1]), "r"(chunk[2]), "r"(chunk[3]), "r"(_mbar_peer_30) : "memory");
                    }
                    if (panel_c != 3) {
                        uint32_t _cluster_mapa_31;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_31) : "r"(chunk_addr), "r"(3));
                        uint32_t _mbar_peer_32;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_32) : "r"(act_ready_addr), "r"(3));
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_cluster_mapa_31), "r"(chunk[0]), "r"(chunk[1]), "r"(chunk[2]), "r"(chunk[3]), "r"(_mbar_peer_32) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                asm volatile("barrier.sync 1, 160;" ::: "memory");
                float down_act_scale[8];
                float down_token_scale[8];
                #pragma unroll
                for (int token_slot_scale = 0; token_slot_scale < 8; token_slot_scale++) {
                    down_act_scale[token_slot_scale] = smem_act_scale[token_slot_scale];
                    int pair_scale = route_pairs[token_slot_scale];
                    bool valid_scale = pair_scale < M * top_k;
                    float route_weight_scale = 0.0f;
                    if (valid_scale) {
                        route_weight_scale = route_weights[token_slot_scale];
                    }
                    down_token_scale[token_slot_scale] = down_act_scale[token_slot_scale] * route_weight_scale * scaling_factor;
                }
                int out_blocks_c = K / 128;
                unsigned int down_stage_c = 0;
                #pragma unroll 1
                for (int ob_c = blockIdx.z; ob_c < out_blocks_c; ob_c += gridDim.z) {
                    mbarrier_wait(down_ready_addr + (down_stage_c) * 8, _phase_down_ready);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int down_addr = taddr + 64 + (unsigned int)(consumer_warp * 32 << 16) + down_stage_c * 8;
                    float _tmem_load_1[8];
                    tmem_ld_x8(&_tmem_load_1[0], down_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(down_free_addr + (down_stage_c) * 8);
                    }
                    int w2s_slot = w2s_count % 32;
                    float w2s = __shfl_sync(0xFFFFFFFF, w2_scales, w2s_slot);
                    if (w2s_slot == 31) {
                        int w2s_next = ((ob_c + (consumer_lane + 1) * w2s_ranks) < (w2s_blocks - 1) ? (ob_c + (consumer_lane + 1) * w2s_ranks) : (w2s_blocks - 1));
                        w2_scales = w2_scale[(w2s_base + w2s_next) * intermediate_blocks_c + intermediate_block_c];
                    }
                    w2s_count = w2s_count + 1;
                    #pragma unroll
                    for (int token_slot_out = 0; token_slot_out < 8; token_slot_out++) {
                        float out_scale = down_token_scale[token_slot_out] * w2s;
                        smem_out[((int)down_stage_c * 8 + token_slot_out) * 128 + physical_row] = _tmem_load_1[token_slot_out] * out_scale;
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    if (warp == 2) {
                        if (elect_sync()) {
                            #pragma unroll
                            for (int token_slot_reduce = 0; token_slot_reduce < 8; token_slot_reduce++) {
                                int pair_reduce = route_pairs[token_slot_reduce];
                                if (pair_reduce < M * top_k) {
                                    int token_reduce = pair_reduce / top_k;
                                    {
                                        void* _cpred_dst_33 = reinterpret_cast<void*>(out + (token_reduce * K + ob_c * 128));
                                        asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"
                                            " [%0], [%1], %2;"
                                            :: "l"(_cpred_dst_33), "r"(smem_out_addr + (unsigned int)(((int)down_stage_c * 8 + token_slot_reduce) * 128 * 2)), "r"((uint32_t)(256))
                                            : "memory");
                                    }
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    down_stage_c += 1;
                    if (down_stage_c == 2) { down_stage_c = 0; _phase_down_ready ^= 1; }
                }
                if (warp == 2) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"

#undef CAKE_INF
#undef TMEM_NCOLS
#undef TMEM_UP_ACC_OFFSET
#undef TMEM_DOWN_ACC_OFFSET
#undef NUM_UP_PIPE_STAGES
#undef NUM_DOWN_PIPE_STAGES
#undef SMEM_SMEM_W1_OFF
#undef SMEM_SMEM_W1_STAGE_BYTES
#undef SMEM_SMEM_W1_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_SMEM_W1B_OFF
#undef SMEM_SMEM_W1B_STAGE_BYTES
#undef SMEM_SMEM_W1B_STRIDE
#undef SMEM_SMEM_XB_OFF
#undef SMEM_SMEM_XB_STAGE_BYTES
#undef SMEM_SMEM_XB_STRIDE
#undef SMEM_SMEM_W2_OFF
#undef SMEM_SMEM_W2_STAGE_BYTES
#undef SMEM_SMEM_W2_STRIDE
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_AMAX_OFF
#undef SMEM_SMEM_AMAX_STAGE_BYTES
#undef SMEM_SMEM_AMAX_STRIDE
#undef SMEM_SMEM_ACT_SCALE_OFF
#undef SMEM_SMEM_ACT_SCALE_STAGE_BYTES
#undef SMEM_SMEM_ACT_SCALE_STRIDE
#undef SMEM_SMEM_PEER_AMAX_OFF
#undef SMEM_SMEM_PEER_AMAX_STAGE_BYTES
#undef SMEM_SMEM_PEER_AMAX_STRIDE
#undef SMEM_SMEM_OUT_OFF
#undef SMEM_SMEM_OUT_STAGE_BYTES
#undef SMEM_SMEM_OUT_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef LAUNCH_MIN_BLOCKS
#undef up_full_addr
#undef up_ready_addr
#undef up_free_addr
#undef up_empty_addr
#undef down_full_addr
#undef down_ready_addr
#undef down_free_addr
#undef down_empty_addr
#undef amax_ready_addr
#undef act_ready_addr
#undef kernel_alpha_moe_w8a8_up_down
constexpr int kGeneratedThreads = 192;
constexpr int kGeneratedSmemTotal = 112896;
}  // namespace w8a8_im



// Capture generated launch resources before isolating its macros.
namespace flashinfer {
namespace alphamoe_sm100 {
constexpr int kGeneratedThreads = w8a8_i2::kGeneratedThreads;  // all six W8A8 blocks are 192 threads
constexpr int kGeneratedSmemTotal = w8a8_i2::kGeneratedSmemTotal;  // i2's block (not used by the launch); the launch uses the selected kernel's own smem
static_assert(w8a8_i2::kGeneratedThreads == w8a8_j2::kGeneratedThreads && w8a8_i2::kGeneratedThreads == w8a8_l2::kGeneratedThreads && w8a8_i2::kGeneratedThreads == w8a8_p2::kGeneratedThreads && w8a8_i2::kGeneratedThreads == w8a8_pl::kGeneratedThreads && w8a8_i2::kGeneratedThreads == w8a8_im::kGeneratedThreads, "the six W8A8 kernels must share the block size");
}  // namespace alphamoe_sm100
}  // namespace flashinfer
#undef NUM_DOWN_PIPE_STAGES
#undef NUM_UP_PIPE_STAGES
#undef SMEM_SMEM_ACT_OFF
#undef SMEM_SMEM_ACT_SCALE_OFF
#undef SMEM_SMEM_ACT_SCALE_STAGE_BYTES
#undef SMEM_SMEM_ACT_SCALE_STRIDE
#undef SMEM_SMEM_ACT_STAGE_BYTES
#undef SMEM_SMEM_ACT_STRIDE
#undef SMEM_SMEM_AMAX_OFF
#undef SMEM_SMEM_AMAX_STAGE_BYTES
#undef SMEM_SMEM_AMAX_STRIDE
#undef SMEM_SMEM_OUT_OFF
#undef SMEM_SMEM_OUT_STAGE_BYTES
#undef SMEM_SMEM_OUT_STRIDE
#undef SMEM_SMEM_W1_OFF
#undef SMEM_SMEM_W1_STAGE_BYTES
#undef SMEM_SMEM_W1_STRIDE
#undef SMEM_SMEM_W2_OFF
#undef SMEM_SMEM_W2_STAGE_BYTES
#undef SMEM_SMEM_W2_STRIDE
#undef SMEM_SMEM_X_OFF
#undef SMEM_SMEM_X_STAGE_BYTES
#undef SMEM_SMEM_X_STRIDE
#undef SMEM_TOTAL
#undef THREADS
#undef TMEM_DOWN_ACC_OFFSET
#undef TMEM_NCOLS
#undef TMEM_UP_ACC_OFFSET
#undef down_free_addr
#undef down_full_addr
#undef down_ready_addr
#undef up_free_addr
#undef up_full_addr
#undef up_ready_addr

// clang-format on

namespace flashinfer {
namespace alphamoe_sm100 {

using tvm::ffi::TensorView;

// Fixed geometry of the generated schedule. The kernel owns one 8-row route
// subtile x one 128-wide intermediate block per CTA; gate/up weight rows are
// consumed as two 128-row TMA boxes (256 interleaved rows) per intermediate
// block, and every FP8 scale is a 128-group block scale.
constexpr int kThreads = kGeneratedThreads;
constexpr int kSmemTotal = kGeneratedSmemTotal;
constexpr int kRouteSubtile = 8;    // route rows owned by one CTA
constexpr int kGroup = 128;         // FP8 scale group / TMA box edge
constexpr int kRowsPerInter = 256;  // interleaved gate+up rows per 128 intermediate columns

struct ProblemDims {
  int m;                     // number of tokens
  int k;                     // hidden size
  int n;                     // gate+up rows per expert (2 * intermediate)
  int intermediate;          // n / 2
  int top_k;                 // routes per token
  int block_m;               // routing-plan block size (route_block_m)
  int64_t num_route_blocks;  // expert_ids.numel()
  int64_t num_experts;       // gemm1_weights.shape[0]
  int64_t num_sms;           // cudaDevAttrMultiProcessorCount of the device
  int64_t quad_clusters;     // QuadClusterCapacity of the device
};

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

// Streaming multiprocessors of a device, read once per device: the attribute
// query enqueues no work, so the variant dispatch stays CUDA-graph safe.
inline int64_t NumSms(int device_id) {
  constexpr int kCachedDevices = 64;
  static std::atomic<int> cached[kCachedDevices];
  const bool cacheable = device_id >= 0 && device_id < kCachedDevices;
  int sms = cacheable ? cached[device_id].load(std::memory_order_relaxed) : 0;
  if (sms == 0) {
    CheckCuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id),
              "cudaDeviceGetAttribute(multiprocessor count)");
    if (cacheable) cached[device_id].store(sms, std::memory_order_relaxed);
  }
  return sms;
}

// Never launched: the occupancy calculator's subject for QuadClusterCapacity.
// It has the quad variants' cluster shape, block size and shared memory but no
// tensor memory, for which the calculator answers one CTA per SM.
__global__ void __cluster_dims__(1, 1, 4) QuadClusterProbe() {}

// The four-CTA quad clusters (i2 / im) a device runs at once at the quad's
// footprint of two CTAs per SM, as the occupancy calculator places them on the
// device's SM groups. Read once per device (Run's device guard makes it current);
// the query enqueues no work. A failed query leaves the quad uncapped. Measured:
// 74 on GB300 (152 SMs), which runs 72 busy clusters in one wave and loses to the
// pair at 80; 71 on B200 (148 SMs), where the quad wins at 70 busy clusters and
// loses at 72.
inline int64_t QuadClusterCapacity(int device_id) {
  constexpr int kCachedDevices = 64;
  static std::atomic<int> cached[kCachedDevices];
  const bool cacheable = device_id >= 0 && device_id < kCachedDevices;
  int capacity = cacheable ? cached[device_id].load(std::memory_order_relaxed) : 0;
  if (capacity == 0) {
    int clusters = 0;
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(1, 1, 4);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = std::max(w8a8_i2::kGeneratedSmemTotal, w8a8_im::kGeneratedSmemTotal);
    bool ok = cudaFuncSetAttribute(QuadClusterProbe, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(config.dynamicSmemBytes)) == cudaSuccess;
    ok = ok && cudaOccupancyMaxActiveClusters(&clusters, QuadClusterProbe, &config) == cudaSuccess;
    if (!ok) (void)cudaGetLastError();
    capacity = ok && clusters > 0 ? clusters : std::numeric_limits<int>::max();
    if (cacheable) cached[device_id].store(capacity, std::memory_order_relaxed);
  }
  return capacity;
}

inline void CheckSm100Family(int device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(compute capability major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(compute capability minor)");
  TVM_FFI_ICHECK(major == 10 && (minor == 0 || minor == 3))
      << "alphamoe_sm100 requires an SM100/SM103 (B200/B300 class) device, got sm_" << major
      << minor;
}

inline void CheckDtypeIs(const TensorView& t, DLDataType want, const char* name,
                         const char* want_name) {
  const DLDataType d = t.dtype();
  TVM_FFI_ICHECK(d.code == want.code && d.bits == want.bits && d.lanes == want.lanes)
      << name << " must be " << want_name << ", got (code=" << int(d.code)
      << ", bits=" << int(d.bits) << ", lanes=" << int(d.lanes) << ")";
}

inline void CheckCudaTensor(const TensorView& t, DLDataType want, int ndim, bool contiguous,
                            const char* name, const char* want_name) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(t.ndim() == ndim) << name << " must be " << ndim << "D, got ndim=" << t.ndim();
  if (contiguous) {
    TVM_FFI_ICHECK(t.IsContiguous()) << name << " must be contiguous";
  }
  CheckDtypeIs(t, want, name, want_name);
}

// A logical axis.outer(trailing) folds every source dim above the trailing
// dimensions. Shape products are independent of physical strides, so verify
// the leading dimensions form one dense row-major chain instead of inventing
// a "folded stride". The descriptor reads its exact adjacent physical step
// separately through stride[-(trailing + 1)]. (Verbatim from the Loom host
// shim; a no-op for the 2D hidden_states tensor.)
inline void CheckDenseLeadingFold(const TensorView& t, int trailing, const char* name) {
  TVM_FFI_ICHECK(trailing > 0 && t.ndim() >= trailing)
      << name << " cannot fold leading dimensions above " << trailing
      << " trailing dims from ndim=" << t.ndim();
  int outer_last = static_cast<int>(t.ndim()) - trailing - 1;
  if (outer_last <= 0) {
    return;
  }
  int64_t step = t.stride(outer_last);
  TVM_FFI_ICHECK(step > 0) << name << " physical strides must be positive";
  int64_t expected = step;
  for (int axis = outer_last - 1; axis >= 0; --axis) {
    expected *= t.size(axis + 1);
    if (t.size(axis) > 1) {
      TVM_FFI_ICHECK(t.stride(axis) == expected)
          << name << " leading dims are not physically foldable above " << trailing
          << " trailing dims: stride(" << axis << ")=" << t.stride(axis) << ", expected "
          << expected;
    }
  }
}

// 2D TMA descriptor for hidden_states (the shim's buffer 'x') —
// field-for-field the descriptor the Loom host shim encodes: raw UINT8 view,
// box (128, 1) for the gather4 row loads, 128B swizzle, no L2 promotion, no
// OOB fill. Only the innermost stride must be 1; the row stride may exceed K
// (row-sliced activations are valid).
inline CUtensorMap EncodeHiddenStatesTma(const TensorView& t) {
  TVM_FFI_ICHECK(t.ndim() >= 2) << "TMA source 'hidden_states' must have at least 2 dimensions, "
                                   "got ndim="
                                << t.ndim();
  TVM_FFI_ICHECK(t.stride(-1) == 1)
      << "TMA source 'hidden_states' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_ICHECK(d1 > 0) << "TMA source 'hidden_states' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "hidden_states");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_ICHECK(s2 > 0) << "TMA source 'hidden_states' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_ICHECK(global_dim[0] > 0 && global_dim[1] > 0)
      << "TMA descriptor for 'hidden_states' resolved a non-positive global dim";
  TVM_FFI_ICHECK(128u <= global_dim[0] && 1u <= global_dim[1])
      << "TMA box (128, 1) exceeds resolved global dims for 'hidden_states'";
  uint64_t global_strides[1] = {
      (uint64_t)((s2 * 8) / 8),
  };
  uint32_t box_dim[2] = {128u, 1u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, t.data_ptr(), global_dim, global_strides, box_dim,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(r == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled (2D, 'hidden_states') failed: CUresult=" << int(r);
  return tm;
}

// 4D TMA descriptor for gemm1_weights (the shim's buffer 'W1') — the logical
// (E, N, K) tensor is presented to TMA as (128, N, K/128, E) so one box is a
// 128-wide K-major tile of box_rows interleaved gate/up rows (128, or 64 for
// the four-CTA i2 variant).
inline CUtensorMap EncodeGemm1WeightsTma(const TensorView& t, uint32_t box_rows) {
  TVM_FFI_ICHECK(t.ndim() >= 2) << "TMA source 'gemm1_weights' must have at least 2 dimensions, "
                                   "got ndim="
                                << t.ndim();
  TVM_FFI_ICHECK(t.stride(-1) == 1)
      << "TMA source 'gemm1_weights' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_ICHECK(d1 > 0 && d2 > 0) << "TMA source 'gemm1_weights' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_ICHECK(d1 % 128 == 0) << "TMA source 'gemm1_weights' extent " << d1
                                << " must divide exactly by " << 128;
  uint64_t global_dim[4] = {(uint64_t)(128), (uint64_t)(d2), (uint64_t)((d1 / 128)),
                            (uint64_t)(outer2)};
  TVM_FFI_ICHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0)
      << "TMA descriptor for 'gemm1_weights' resolved a non-positive global dim";
  TVM_FFI_ICHECK(128u <= global_dim[0] && box_rows <= global_dim[1] && 1u <= global_dim[2] &&
                 1u <= global_dim[3])
      << "TMA box (128, " << box_rows << ", 1, 1) exceeds resolved global dims for 'gemm1_weights'";
  uint64_t global_strides[3] = {
      (uint64_t)((d1 * 8) / 8),
      (uint64_t)((128 * 8) / 8),
      (uint64_t)(((d2 * d1) * 8) / 8),
  };
  uint32_t box_dim[4] = {128u, box_rows, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, t.data_ptr(), global_dim, global_strides, box_dim,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(r == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled (4D, 'gemm1_weights') failed: CUresult=" << int(r);
  return tm;
}

// 3D TMA descriptor for gemm2_weights (the shim's buffer 'W2') — the logical
// (E, K, N/2) tensor maps directly to (N/2, K, E) with 128x128 boxes.
inline CUtensorMap EncodeGemm2WeightsTma(const TensorView& t) {
  TVM_FFI_ICHECK(t.ndim() >= 2) << "TMA source 'gemm2_weights' must have at least 2 dimensions, "
                                   "got ndim="
                                << t.ndim();
  TVM_FFI_ICHECK(t.stride(-1) == 1)
      << "TMA source 'gemm2_weights' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_ICHECK(d1 > 0 && d2 > 0) << "TMA source 'gemm2_weights' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_ICHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0)
      << "TMA descriptor for 'gemm2_weights' resolved a non-positive global dim";
  TVM_FFI_ICHECK(128u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2])
      << "TMA box (128, 128, 1) exceeds resolved global dims for 'gemm2_weights'";
  uint64_t global_strides[2] = {
      (uint64_t)((d1 * 8) / 8),
      (uint64_t)(((d2 * d1) * 8) / 8),
  };
  uint32_t box_dim[3] = {128u, 128u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, t.data_ptr(), global_dim, global_strides, box_dim,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(r == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled (3D, 'gemm2_weights') failed: CUresult=" << int(r);
  return tm;
}

// Validate the public contract. Dtype/contiguity requirements mirror the Loom
// host shim (the shim types FP8 tensors as raw uint8 views; here they carry
// their real float8_e4m3fn dtype); the shape/divisibility constraints are the
// generated schedule's coverage guards: K and N/2 positive multiples of 128
// (N % 256 == 0), block_m a positive multiple of the 8-row route subtile.
inline ProblemDims CheckInputs(
    const TensorView& hidden_states, const TensorView& hidden_states_scale,
    const TensorView& gemm1_weights, const TensorView& gemm1_weights_scale,
    const TensorView& gemm2_weights, const TensorView& gemm2_weights_scale,
    const TensorView& sorted_token_ids, const TensorView& expert_ids,
    const TensorView& num_tokens_post_padded, const TensorView& topk_weights, const TensorView& out,
    int64_t top_k, int64_t block_m) {
  // hidden_states deliberately skips the contiguity check: the TMA descriptor
  // reads the row stride, so row-sliced activations are valid.
  CheckCudaTensor(hidden_states, dl_float8_e4m3fn, 2, /*contiguous=*/false, "hidden_states",
                  "float8_e4m3fn");
  CheckCudaTensor(hidden_states_scale, dl_float32, 2, /*contiguous=*/true, "hidden_states_scale",
                  "float32");
  CheckCudaTensor(gemm1_weights, dl_float8_e4m3fn, 3, /*contiguous=*/true, "gemm1_weights",
                  "float8_e4m3fn");
  CheckCudaTensor(gemm1_weights_scale, dl_float32, 3, /*contiguous=*/true, "gemm1_weights_scale",
                  "float32");
  CheckCudaTensor(gemm2_weights, dl_float8_e4m3fn, 3, /*contiguous=*/true, "gemm2_weights",
                  "float8_e4m3fn");
  CheckCudaTensor(gemm2_weights_scale, dl_float32, 3, /*contiguous=*/true, "gemm2_weights_scale",
                  "float32");
  CheckCudaTensor(sorted_token_ids, dl_int32, 1, /*contiguous=*/true, "sorted_token_ids", "int32");
  CheckCudaTensor(expert_ids, dl_int32, 1, /*contiguous=*/true, "expert_ids", "int32");
  CheckCudaTensor(num_tokens_post_padded, dl_int32, 1, /*contiguous=*/true,
                  "num_tokens_post_padded", "int32");
  CheckCudaTensor(topk_weights, dl_float32, 2, /*contiguous=*/true, "topk_weights", "float32");
  CheckCudaTensor(out, dl_bfloat16, 2, /*contiguous=*/true, "out", "bfloat16");

  TVM_FFI_ICHECK(hidden_states.stride(1) == 1)
      << "hidden_states must have unit innermost stride, got " << hidden_states.stride(1);
  TVM_FFI_ICHECK(hidden_states.stride(0) > 0 && hidden_states.stride(0) >= hidden_states.size(1) &&
                 hidden_states.stride(0) % 16 == 0)
      << "hidden_states row stride must be positive, non-overlapping, and 16-byte aligned, got "
      << hidden_states.stride(0) << " for row width " << hidden_states.size(1);
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(hidden_states.data_ptr()) % 16 == 0)
      << "hidden_states data pointer must be 16-byte aligned for TMA";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0)
      << "out data pointer must be 16-byte aligned for bulk reduction";

  const int device_id = hidden_states.device().device_id;
  const TensorView* views[] = {&hidden_states_scale, &gemm1_weights,
                               &gemm1_weights_scale, &gemm2_weights,
                               &gemm2_weights_scale, &sorted_token_ids,
                               &expert_ids,          &num_tokens_post_padded,
                               &topk_weights,        &out};
  for (const TensorView* view : views) {
    TVM_FFI_ICHECK(view->device().device_id == device_id)
        << "all alphamoe_sm100 tensors must live on the same CUDA device";
  }
  CheckSm100Family(device_id);

  const int64_t m = hidden_states.size(0);
  const int64_t k = hidden_states.size(1);
  const int64_t num_experts = gemm1_weights.size(0);
  const int64_t n = gemm1_weights.size(1);
  TVM_FFI_ICHECK(m > 0) << "hidden_states must have at least one token, got M=" << m;
  TVM_FFI_ICHECK(k >= kGroup && k % kGroup == 0)
      << "K (" << k << ") must be a positive multiple of " << kGroup << " and at least " << kGroup;
  TVM_FFI_ICHECK(n >= kRowsPerInter && n % kRowsPerInter == 0)
      << "gemm1_weights.shape[1] (" << n << ") must be a positive multiple of " << kRowsPerInter
      << " (gate+up rows per 128-wide intermediate block)";
  TVM_FFI_ICHECK(gemm1_weights.size(2) == k)
      << "gemm1_weights.shape[2] (" << gemm1_weights.size(2) << ") must equal K (" << k << ")";
  const int64_t intermediate = n / 2;

  TVM_FFI_ICHECK(hidden_states_scale.size(0) == m && hidden_states_scale.size(1) == k / kGroup)
      << "hidden_states_scale must have shape (" << m << ", " << k / kGroup << "), got ("
      << hidden_states_scale.size(0) << ", " << hidden_states_scale.size(1) << ")";
  TVM_FFI_ICHECK(gemm1_weights_scale.size(0) == num_experts &&
                 gemm1_weights_scale.size(1) == n / kGroup &&
                 gemm1_weights_scale.size(2) == k / kGroup)
      << "gemm1_weights_scale must have shape (" << num_experts << ", " << n / kGroup << ", "
      << k / kGroup << "), got (" << gemm1_weights_scale.size(0) << ", "
      << gemm1_weights_scale.size(1) << ", " << gemm1_weights_scale.size(2) << ")";
  TVM_FFI_ICHECK(gemm2_weights.size(0) == num_experts && gemm2_weights.size(1) == k &&
                 gemm2_weights.size(2) == intermediate)
      << "gemm2_weights must have shape (" << num_experts << ", " << k << ", " << intermediate
      << "), got (" << gemm2_weights.size(0) << ", " << gemm2_weights.size(1) << ", "
      << gemm2_weights.size(2) << ")";
  TVM_FFI_ICHECK(gemm2_weights_scale.size(0) == num_experts &&
                 gemm2_weights_scale.size(1) == k / kGroup &&
                 gemm2_weights_scale.size(2) == intermediate / kGroup)
      << "gemm2_weights_scale must have shape (" << num_experts << ", " << k / kGroup << ", "
      << intermediate / kGroup << "), got (" << gemm2_weights_scale.size(0) << ", "
      << gemm2_weights_scale.size(1) << ", " << gemm2_weights_scale.size(2) << ")";

  TVM_FFI_ICHECK(top_k >= 1) << "top_k must be positive, got " << top_k;
  TVM_FFI_ICHECK(block_m >= kRouteSubtile && block_m % kRouteSubtile == 0)
      << "block_m (" << block_m << ") must be a positive multiple of " << kRouteSubtile;
  TVM_FFI_ICHECK(num_tokens_post_padded.numel() == 1)
      << "num_tokens_post_padded must be a device int32 tensor with exactly one element, got "
      << num_tokens_post_padded.numel();
  TVM_FFI_ICHECK(expert_ids.numel() > 0) << "expert_ids must not be empty";
  // Divide instead of multiplying the required capacity to avoid overflow.
  TVM_FFI_ICHECK(expert_ids.numel() <= sorted_token_ids.numel() / block_m)
      << "sorted_token_ids must contain at least expert_ids.numel() * block_m entries";
  TVM_FFI_ICHECK(topk_weights.size(0) == m && topk_weights.size(1) == top_k)
      << "topk_weights must have shape (" << m << ", " << top_k << "), got ("
      << topk_weights.size(0) << ", " << topk_weights.size(1) << ")";
  TVM_FFI_ICHECK(out.size(0) == m && out.size(1) == k)
      << "out must have shape (" << m << ", " << k << "), got (" << out.size(0) << ", "
      << out.size(1) << ")";

  // The kernel receives M/K/top_k/block_m as i32 scalars and uses M * top_k
  // as the routing-plan padding sentinel; all must fit in i32.
  TVM_FFI_ICHECK(m <= std::numeric_limits<int>::max() && k <= std::numeric_limits<int>::max() &&
                 m * top_k <= std::numeric_limits<int>::max())
      << "problem dimensions exceed the kernel's i32 scalar range";

  return ProblemDims{static_cast<int>(m),     static_cast<int>(k),
                     static_cast<int>(n),     static_cast<int>(intermediate),
                     static_cast<int>(top_k), static_cast<int>(block_m),
                     expert_ids.numel(),      num_experts,
                     NumSms(device_id),       QuadClusterCapacity(device_id)};
}

// An upper bound of the routing plan's busy 8-row route subtiles from host-known sizes only (no
// device read): each of the at most min(E, M * top_k) experts with routes adds at most one partly
// filled block_m block to its full ones.
inline int64_t MaxBusySubtiles(const ProblemDims& dims) {
  const int64_t routes = static_cast<int64_t>(dims.m) * dims.top_k;
  const int64_t experts = std::min(dims.num_experts, routes);
  return (routes + (dims.block_m - 1) * experts) / dims.block_m * (dims.block_m / kRouteSubtile);
}

// The most busy subtiles i2 takes for M > 8: a share of the SMs that grows with K, since the
// four-CTA split repays its activation exchange over a longer K loop (7/16 of the SMs from
// K 4096, 11/32 below it, 7/32 below K 2048). Below K 2048 it runs only with at most 256
// experts: with more, the idle clusters of the plan's capacity cost more than it gains.
inline int64_t QuarterPanelMaxSubtiles(const ProblemDims& dims) {
  if (dims.k >= 4096) return dims.num_sms * 7 / 16;
  if (dims.k >= 2048) return dims.num_sms * 11 / 32;
  return dims.num_experts <= 256 ? dims.num_sms * 7 / 32 : 0;
}

// The i2 plans that run im, i2 whose MMA commit frees each operand stage (its
// consumers free only the TMEM product): K from 7168, the measured range
// where it is faster; every other i2 plan keeps i2.
inline bool UsesQuarterMmaRelease(const ProblemDims& dims) { return dims.k >= 7168; }

// i2 splits each expert block over four CTAs, each streaming a quarter of the
// 256 interleaved W1 rows of an intermediate block: its W1 tensor map uses
// 64-row boxes. It runs M <= 8 from K 4096 while the plan's busy subtiles
// (one cluster each) fit the device's co-resident quad clusters, and M > 8
// from K 1024 up to QuarterPanelMaxSubtiles. Past either bound the pair is
// faster. The other variants load 128-row boxes. This one predicate selects
// both the map and the kernel.
inline bool UsesQuarterW1Panels(const ProblemDims& dims) {
  if (dims.m <= 8) return dims.k >= 4096 && MaxBusySubtiles(dims) <= dims.quad_clusters;
  return dims.block_m == kRouteSubtile && dims.k >= 1024 &&
         MaxBusySubtiles(dims) <= QuarterPanelMaxSubtiles(dims);
}

// The p2 plans that run pl, the pair with an L2 prefetch of the weights ahead
// of its ring: K from 1024, at least 64 of K per route (M * top_k) and
// M * top_k * K at most 40960: a few tokens, each streaming a long enough
// K. With more routes the prefetch competes with the ring's own loads; below
// K 1024 the ring's first lap holds the whole K.
inline bool UsesPairL2Prefetch(const ProblemDims& dims) {
  const int64_t routes = static_cast<int64_t>(dims.m) * dims.top_k;
  return dims.k >= 1024 && routes * 64 <= dims.k && routes * dims.k <= 40960;
}

// M > 8 below K 4096: p2 (a CTA pair per expert block, two CTAs per SM) while
// its grid fits one wave (at most one busy subtile per SM) from K 2048, and from
// K 1024 with at most 32 experts; with more, up to 5/8 of the SMs from K 1024,
// where the pair's exchange outweighs its gain once the weights are L2-resident;
// l2 (one CTA per block) above.
inline bool UsesPairBelowK4096(const ProblemDims& dims) {
  if (dims.k >= 2048) return MaxBusySubtiles(dims) <= dims.num_sms;
  if (dims.k < 1024) return false;
  const int64_t cap = dims.num_experts <= 32 ? dims.num_sms : dims.num_sms * 5 / 8;
  return MaxBusySubtiles(dims) <= cap;
}

// Launch the generated kernel. TMap is the generated TU's by-value
// __grid_constant__ tensor-map parameter type (native CUtensorMap); it is
// layout-compatible with CUtensorMap. The kernel variant and its dynamic smem
// footprint are selected by (M, K); every variant launches 192-thread blocks,
// one CTA per (8-row route subtile, 128-wide intermediate block and
// output-column slice): grid.x/y are unchanged; grid.z selects disjoint
// down-output blocks in the generated grid-stride loop.
template <typename TMap>
inline void LaunchKernel(void (*kernel)(TMap, TMap, float*, float*, TMap, float*, int*, int*, int*,
                                        float*, __nv_bfloat16*, int, int, int, int, float),
                         const CUtensorMap& hidden_states_map, const CUtensorMap& gemm1_map,
                         const CUtensorMap& gemm2_map, float* hidden_states_scale,
                         float* gemm1_weights_scale, float* gemm2_weights_scale,
                         int* sorted_token_ids, int* expert_ids, int* num_tokens_post_padded,
                         float* topk_weights, __nv_bfloat16* out, const ProblemDims& dims,
                         float scaling_factor, cudaStream_t stream) {
  static_assert(sizeof(TMap) == sizeof(CUtensorMap),
                "generated tensor-map parameter must be layout-compatible with CUtensorMap");
  TMap x_param, w1_param, w2_param;
  std::memcpy(&x_param, &hidden_states_map, sizeof(TMap));
  std::memcpy(&w1_param, &gemm1_map, sizeof(TMap));
  std::memcpy(&w2_param, &gemm2_map, sizeof(TMap));

  // Variant dispatch. The six variants share the kernel signature and block
  // size. i2 splits each expert block over four CTAs (a static (1, 1, 4)
  // cluster along grid z, launched at grid z 4), each streaming a quarter of
  // the W1 rows (UsesQuarterW1Panels: M <= 8 with K >= 4096 up to the quad
  // clusters the device co-schedules, or M > 8 with K >= 1024 up to
  // QuarterPanelMaxSubtiles busy subtiles). p2 splits each expert block over a
  // CTA pair (a static (1, 1, 2) cluster, grid z 2) for the other M <= 8
  // plans from K 256, for M > 8 from K 4096, and for M > 8 below
  // K 4096 up to a busy-subtile bound (UsesPairBelowK4096). j2 (UP 4 /
  // DOWN 4, grid z 1) keeps M <= 8 below K 256, where the pair's fixed
  // hand-off cost exceeds its gain. l2 (one CTA per expert block, two CTAs per
  // SM, grid z 1) runs the other M > 8 cases below K 4096. The selected
  // variant replaces the kernel passed in for type deduction. pl is p2 with
  // an L2 prefetch of the weights ahead of its ring, for the p2 plans of
  // UsesPairL2Prefetch.
  // im is i2 with each operand stage released by the MMA's commit, for the
  // i2 plans of UsesQuarterMmaRelease.
  int smem_total;
  unsigned int grid_z;
  if (UsesQuarterW1Panels(dims) && UsesQuarterMmaRelease(dims)) {
    kernel = &w8a8_im::kernel_alpha_moe_w8a8_up_down_w8a8_im;
    smem_total = w8a8_im::kGeneratedSmemTotal;
    grid_z = 4u;
  } else if (UsesQuarterW1Panels(dims)) {
    kernel = &w8a8_i2::kernel_alpha_moe_w8a8_up_down_w8a8_i2;
    smem_total = w8a8_i2::kGeneratedSmemTotal;
    grid_z = 4u;
  } else if (dims.m <= 8 && dims.k < 256) {
    kernel = &w8a8_j2::kernel_alpha_moe_w8a8_up_down_w8a8_j2;
    smem_total = w8a8_j2::kGeneratedSmemTotal;
    grid_z = 1u;
  } else if (dims.m > 8 && dims.k < 4096 && !UsesPairBelowK4096(dims)) {
    kernel = &w8a8_l2::kernel_alpha_moe_w8a8_up_down_w8a8_l2;
    smem_total = w8a8_l2::kGeneratedSmemTotal;
    grid_z = 1u;
  } else if (UsesPairL2Prefetch(dims)) {
    kernel = &w8a8_pl::kernel_alpha_moe_w8a8_up_down_w8a8_pl;
    smem_total = w8a8_pl::kGeneratedSmemTotal;
    grid_z = 2u;
  } else {
    kernel = &w8a8_p2::kernel_alpha_moe_w8a8_up_down_w8a8_p2;
    smem_total = w8a8_p2::kGeneratedSmemTotal;
    grid_z = 2u;
  }
  CheckCuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_total),
            "cudaFuncSetAttribute(alphamoe_sm100 dynamic smem)");

  const int64_t grid_x = dims.num_route_blocks * (dims.block_m / kRouteSubtile);
  const int64_t grid_y = dims.n / kRowsPerInter;
  TVM_FFI_ICHECK(grid_x > 0 && grid_x <= std::numeric_limits<int>::max())
      << "launch grid.x (" << grid_x << ") out of range";
  TVM_FFI_ICHECK(grid_y > 0 && grid_y <= 65535) << "launch grid.y (" << grid_y << ") out of range";
  const dim3 grid(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y), grid_z);
  const dim3 block(kThreads);

  kernel<<<grid, block, smem_total, stream>>>(
      x_param, w1_param, hidden_states_scale, gemm1_weights_scale, w2_param, gemm2_weights_scale,
      sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, out, dims.m, dims.k,
      dims.top_k, dims.block_m, scaling_factor);
  CheckCuda(cudaGetLastError(), "alphamoe_sm100 kernel launch");
}

}  // namespace alphamoe_sm100
}  // namespace flashinfer

namespace flashinfer {
namespace alphamoe_sm100 {

// Fused W8A8 block-scale MoE (gate/up + SwiGLU + per-token FP8 requant +
// down) over a moe_align_block_size routing plan. `out` is a caller-owned
// BF16 accumulator: the kernel asynchronously reduce-adds every routed
// expert contribution into it, so it must be zeroed (or hold the intended
// initial values) before launch.
void Run(TensorView hidden_states, TensorView hidden_states_scale, TensorView gemm1_weights,
         TensorView gemm1_weights_scale, TensorView gemm2_weights, TensorView gemm2_weights_scale,
         TensorView sorted_token_ids, TensorView expert_ids, TensorView num_tokens_post_padded,
         TensorView topk_weights, TensorView out, int64_t top_k, int64_t block_m,
         double routed_scaling_factor) {
  TVM_FFI_ICHECK(hidden_states.device().device_type == kDLCUDA)
      << "hidden_states must be a CUDA tensor";
  ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
  const ProblemDims dims =
      CheckInputs(hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale,
                  gemm2_weights, gemm2_weights_scale, sorted_token_ids, expert_ids,
                  num_tokens_post_padded, topk_weights, out, top_k, block_m);
  const CUtensorMap hidden_states_map = EncodeHiddenStatesTma(hidden_states);
  const CUtensorMap gemm1_map =
      EncodeGemm1WeightsTma(gemm1_weights, UsesQuarterW1Panels(dims) ? 64u : 128u);
  const CUtensorMap gemm2_map = EncodeGemm2WeightsTma(gemm2_weights);
  const cudaStream_t stream = get_stream(hidden_states.device());
  LaunchKernel(&w8a8_i2::kernel_alpha_moe_w8a8_up_down_w8a8_i2, hidden_states_map, gemm1_map,
               gemm2_map, static_cast<float*>(hidden_states_scale.data_ptr()),
               static_cast<float*>(gemm1_weights_scale.data_ptr()),
               static_cast<float*>(gemm2_weights_scale.data_ptr()),
               static_cast<int*>(sorted_token_ids.data_ptr()),
               static_cast<int*>(expert_ids.data_ptr()),
               static_cast<int*>(num_tokens_post_padded.data_ptr()),
               static_cast<float*>(topk_weights.data_ptr()),
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), dims,
               static_cast<float>(routed_scaling_factor), stream);
}

}  // namespace alphamoe_sm100
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_block_scale_aligned_moe_op, flashinfer::alphamoe_sm100::Run);
