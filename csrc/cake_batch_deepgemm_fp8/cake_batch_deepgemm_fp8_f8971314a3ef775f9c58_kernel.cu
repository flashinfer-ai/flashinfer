/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
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
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFB_OFFSET 64
#define TMEM_TMEM_SFA_OFFSET 68
#define NUM_TMA_PIPE_STAGES 6
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_EPI_TMA_PIPE_STAGES 1
#define SMEM_EPI_STAGING_OFF 1024
#define SMEM_EPI_STAGING_STAGE_BYTES 4096
#define SMEM_EPI_STAGING_STRIDE 4096
#define SMEM_SMEM_A_OFF 5120
#define SMEM_SMEM_A_STAGE_BYTES 4096
#define SMEM_SMEM_A_STRIDE 37888
#define SMEM_SMEM_B_OFF 9216
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 37888
#define SMEM_SMEM_SFA_OFF 41984
#define SMEM_SMEM_SFA_STAGE_BYTES 512
#define SMEM_SMEM_SFA_STRIDE 37888
#define SMEM_SMEM_SFB_OFF 42496
#define SMEM_SMEM_SFB_STAGE_BYTES 512
#define SMEM_SMEM_SFB_STRIDE 37888
#define SMEM_TOTAL 232448
#define THREADS 256
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



__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
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


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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



__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
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



extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_batch_deepgemm_fp8_f8971314a3ef775f9c58(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C_tma, const __grid_constant__ CUtensorMap SFA_packed, const __grid_constant__ CUtensorMap SFB_packed, int* __restrict__ masked_m, unsigned int num_groups, unsigned int shape_m, unsigned int N, unsigned int K)
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
    #define tma_full_addr (mbar_base + 0)
    #define tma_free_addr (mbar_base + 48)
    #define scale_ready_addr (mbar_base + 96)
    #define mainloop_done_addr (mbar_base + 144)
    #define epilogue_done_addr (mbar_base + 160)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int epi_staging_addr = smem + 1024;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_a_addr = smem + 5120;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_b_addr = smem + 9216;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 41984);
    const int smem_sfa_addr = smem + 41984;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 42496);
    const int smem_sfb_addr = smem + 42496;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA_packed))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB_packed))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C_tma))) : "memory");

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 22 barriers)
    // Mbarriers at smem_raw[0..176)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 6 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // tma_free: 6 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // scale_ready: 6 barriers, init_count=64
            mbarrier_init(smem + 96, 64);
            mbarrier_init(smem + 104, 64);
            mbarrier_init(smem + 112, 64);
            mbarrier_init(smem + 120, 64);
            mbarrier_init(smem + 128, 64);
            mbarrier_init(smem + 136, 64);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 160, 8);
            mbarrier_init(smem + 168, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 72 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 176);
    if (warp == 2) {
        int _tmem_hold = smem + 176;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
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
    const int tmem_tmem_sfb = taddr + 64;
    const int tmem_tmem_sfa = taddr + 68;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int grid_n = ((1) ? (unsigned int)32 : N / 128);
            unsigned int num_k_blocks = ((1) ? (unsigned int)28 : K / 256);
            unsigned int sf_packed_cols = ((1) ? (unsigned int)14 : K / 512);
            unsigned int cta_rank_0 = (unsigned int)cta_rank;
            unsigned int num_bids_u = (unsigned int)num_bids;
            unsigned int bid_u = (unsigned int)bid;
            unsigned int current_group_idx = 0;
            unsigned int current_m_cumsum = 0;
            unsigned int group_bound = ((1) ? (unsigned int)64 : num_groups);
            unsigned int max_m_blocks = (shape_m + 32 - 1) / 32;
            unsigned int max_total_tiles = group_bound * max_m_blocks * grid_n;
            unsigned int max_scheduler_iters = (max_total_tiles + num_bids_u - 1) / num_bids_u;
            unsigned int _phase_tma_free = 1;
            #pragma unroll 1
            for (int current_iter = 0; current_iter < max_scheduler_iters; current_iter++) {
                unsigned int next_block_idx = (unsigned int)current_iter * num_bids_u + bid_u;
                int has_block = 0;
                unsigned int m_blocks_g = 0;
                #pragma unroll 1
                for (int scan_g = current_group_idx; scan_g < group_bound; scan_g++) {
                    unsigned int group_m = (unsigned int)masked_m[scan_g];
                    m_blocks_g = (group_m + 32 - 1) / 32;
                    unsigned int next_m_cumsum = current_m_cumsum + m_blocks_g;
                    if (next_block_idx < next_m_cumsum * grid_n) {
                        current_group_idx = scan_g;
                        has_block = 1;
                        break;
                    }
                    current_m_cumsum = next_m_cumsum;
                }
                if (has_block == 0) {
                    break;
                }
                unsigned int group_m_1 = (unsigned int)masked_m[current_group_idx];
                unsigned int block_idx = next_block_idx - current_m_cumsum * grid_n;
                unsigned int blocks_per_l2_group = m_blocks_g;
                unsigned int l2_group = block_idx / blocks_per_l2_group;
                unsigned int first_n_block = l2_group;
                unsigned int in_l2_group = block_idx % blocks_per_l2_group;
                unsigned int remaining_n_blocks = grid_n - first_n_block;
                unsigned int n_l2_limit = (unsigned int)1;
                unsigned int n_blocks_in_group = ((remaining_n_blocks > n_l2_limit) ? n_l2_limit : remaining_n_blocks);
                unsigned int m_block = in_l2_group / n_blocks_in_group;
                unsigned int n_block = first_n_block + in_l2_group % n_blocks_in_group;
                unsigned int off_m = m_block * 32;
                unsigned int off_n = n_block * 128;
                unsigned int remaining_m = group_m_1 - off_m;
                unsigned int runtime_mma_n = (remaining_m + 15) / 16 * 16;
                if (runtime_mma_n > 32) {
                    runtime_mma_n = 32;
                }
                unsigned int load_block_m = runtime_mma_n / 2;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < num_k_blocks; iter_k++) {
                    mbarrier_wait(tma_free_addr + (load_stage) * 8, _phase_tma_free);
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_b_addr + load_stage * 37888), "l"((&B)), "r"(0), "r"(off_n), "r"((unsigned int)iter_k * 2), "r"(current_group_idx),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(smem_a_addr + load_stage * 37888), "l"((&A)), "r"(0), "r"(off_m + cta_rank_0 * load_block_m), "r"((unsigned int)iter_k * 2), "r"(current_group_idx),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x1000000000000000ULL) : "memory");
                        if (iter_k % 2 == 0) {
                            unsigned int sf_outer = current_group_idx * sf_packed_cols + (unsigned int)iter_k / 2;
                            tma_2d_gmem2smem(smem_sfa_addr + load_stage * 37888, (&SFA_packed), off_m, sf_outer, tma_full_addr + (load_stage) * 8);
                            tma_2d_gmem2smem(smem_sfb_addr + load_stage * 37888, (&SFB_packed), off_n, sf_outer, tma_full_addr + (load_stage) * 8);
                        }
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 36864 + ((iter_k % 2 == 0) ? 640 : 0));
                    }
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_tma_free ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int tma_stage = 0;
            unsigned int epi_stage = 0;
            unsigned int grid_n_1 = ((1) ? (unsigned int)32 : N / 128);
            unsigned int num_k_blocks_1 = ((1) ? (unsigned int)28 : K / 256);
            unsigned int num_bids_u_1 = (unsigned int)num_bids;
            unsigned int bid_u_1 = (unsigned int)bid;
            unsigned int current_group_idx_1 = 0;
            unsigned int current_m_cumsum_1 = 0;
            unsigned int group_bound_1 = ((1) ? (unsigned int)64 : num_groups);
            unsigned int max_m_blocks_1 = (shape_m + 32 - 1) / 32;
            unsigned int max_total_tiles_1 = group_bound_1 * max_m_blocks_1 * grid_n_1;
            unsigned int max_scheduler_iters_1 = (max_total_tiles_1 + num_bids_u_1 - 1) / num_bids_u_1;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_scale_ready = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (int current_iter_1 = 0; current_iter_1 < max_scheduler_iters_1; current_iter_1++) {
                    unsigned int next_block_idx_1 = (unsigned int)current_iter_1 * num_bids_u_1 + bid_u_1;
                    int has_block_1 = 0;
                    unsigned int m_blocks_g_1 = 0;
                    #pragma unroll 1
                    for (int scan_g_1 = current_group_idx_1; scan_g_1 < group_bound_1; scan_g_1++) {
                        unsigned int group_m_2 = (unsigned int)masked_m[scan_g_1];
                        m_blocks_g_1 = (group_m_2 + 32 - 1) / 32;
                        unsigned int next_m_cumsum_1 = current_m_cumsum_1 + m_blocks_g_1;
                        if (next_block_idx_1 < next_m_cumsum_1 * grid_n_1) {
                            current_group_idx_1 = scan_g_1;
                            has_block_1 = 1;
                            break;
                        }
                        current_m_cumsum_1 = next_m_cumsum_1;
                    }
                    if (has_block_1 == 0) {
                        break;
                    }
                    unsigned int group_m_3 = (unsigned int)masked_m[current_group_idx_1];
                    unsigned int block_idx_1 = next_block_idx_1 - current_m_cumsum_1 * grid_n_1;
                    unsigned int blocks_per_l2_group_1 = m_blocks_g_1;
                    unsigned int l2_group_1 = block_idx_1 / blocks_per_l2_group_1;
                    unsigned int first_n_block_1 = l2_group_1;
                    unsigned int in_l2_group_1 = block_idx_1 % blocks_per_l2_group_1;
                    unsigned int remaining_n_blocks_1 = grid_n_1 - first_n_block_1;
                    unsigned int n_l2_limit_1 = (unsigned int)1;
                    unsigned int n_blocks_in_group_1 = ((remaining_n_blocks_1 > n_l2_limit_1) ? n_l2_limit_1 : remaining_n_blocks_1);
                    unsigned int m_block_1 = in_l2_group_1 / n_blocks_in_group_1;
                    unsigned int remaining_m_1 = group_m_3 - m_block_1 * 32;
                    unsigned int runtime_mma_n_1 = (remaining_m_1 + 15) / 16 * 16;
                    if (runtime_mma_n_1 > 32) {
                        runtime_mma_n_1 = 32;
                    }
                    mbarrier_wait(epilogue_done_addr + (epi_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 4
                    for (int iter_k_1 = 0; iter_k_1 < num_k_blocks_1; iter_k_1++) {
                        mbarrier_wait(scale_ready_addr + (tma_stage) * 8, _phase_scale_ready);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                        if (elect_sync()) {
                            if (iter_k_1 % 2 == 0) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (tma_stage) * 2368)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (tma_stage) * 2368)));
                            }
                            int _mma_a_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (tma_stage) * 2368;
                            int _mma_b_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (tma_stage) * 2368;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (epi_stage * 32)), a_desc + 0, b_desc + 0,
                                    (((0x10880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n_1) >> 3) << 17)) | ((((uint32_t)(iter_k_1 * 2) % 4U)) << 29) | ((((uint32_t)(iter_k_1 * 2) % 4U)) << 4)), tmem_tmem_sfb, tmem_tmem_sfa, ((init_flag) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (epi_stage * 32)), a_desc + 2, b_desc + 2,
                                    (((0x10880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n_1) >> 3) << 17)) | ((((uint32_t)(iter_k_1 * 2) % 4U)) << 29) | ((((uint32_t)(iter_k_1 * 2) % 4U)) << 4)), tmem_tmem_sfb, tmem_tmem_sfa, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (epi_stage * 32)), a_desc + 4, b_desc + 4,
                                    (((0x10880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n_1) >> 3) << 17)) | ((((uint32_t)(iter_k_1 * 2) % 4U)) << 29) | ((((uint32_t)(iter_k_1 * 2) % 4U)) << 4)), tmem_tmem_sfb, tmem_tmem_sfa, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (epi_stage * 32)), a_desc + 6, b_desc + 6,
                                    (((0x10880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n_1) >> 3) << 17)) | ((((uint32_t)(iter_k_1 * 2) % 4U)) << 29) | ((((uint32_t)(iter_k_1 * 2) % 4U)) << 4)), tmem_tmem_sfb, tmem_tmem_sfa, 1);
                            }
                            int _mma_a_lo_1 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (tma_stage) * 2368;
                            int _mma_b_lo_1 = (((smem_a_addr + 2048) >> 4) & 0x3FFF) + (tma_stage) * 2368;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (epi_stage * 32)), a_desc + 0, b_desc + 0,
                                    (((0x10880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n_1) >> 3) << 17)) | ((((uint32_t)(iter_k_1 * 2 + 1) % 4U)) << 29) | ((((uint32_t)(iter_k_1 * 2 + 1) % 4U)) << 4)), tmem_tmem_sfb, tmem_tmem_sfa, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (epi_stage * 32)), a_desc + 2, b_desc + 2,
                                    (((0x10880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n_1) >> 3) << 17)) | ((((uint32_t)(iter_k_1 * 2 + 1) % 4U)) << 29) | ((((uint32_t)(iter_k_1 * 2 + 1) % 4U)) << 4)), tmem_tmem_sfb, tmem_tmem_sfa, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (epi_stage * 32)), a_desc + 4, b_desc + 4,
                                    (((0x10880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n_1) >> 3) << 17)) | ((((uint32_t)(iter_k_1 * 2 + 1) % 4U)) << 29) | ((((uint32_t)(iter_k_1 * 2 + 1) % 4U)) << 4)), tmem_tmem_sfb, tmem_tmem_sfa, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (epi_stage * 32)), a_desc + 6, b_desc + 6,
                                    (((0x10880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n_1) >> 3) << 17)) | ((((uint32_t)(iter_k_1 * 2 + 1) % 4U)) << 29) | ((((uint32_t)(iter_k_1 * 2 + 1) % 4U)) << 4)), tmem_tmem_sfb, tmem_tmem_sfa, 1);
                            }
                        }
                        elect_commit_cg2_multicast(tma_free_addr + (tma_stage) * 8, (uint16_t)(3));
                        tma_stage += 1;
                        if (tma_stage == 6) { tma_stage = 0; _phase_scale_ready ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (epi_stage) * 8, (uint16_t)(3));
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_epilogue_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: transpose ----
    if (warp == 2) {
        { // transpose_main
            unsigned int transpose_stage = 0;
            unsigned int grid_n_2 = ((1) ? (unsigned int)32 : N / 128);
            unsigned int num_k_blocks_2 = ((1) ? (unsigned int)28 : K / 256);
            unsigned int num_bids_u_2 = (unsigned int)num_bids;
            unsigned int bid_u_2 = (unsigned int)bid;
            unsigned int current_group_idx_2 = 0;
            unsigned int current_m_cumsum_2 = 0;
            unsigned int group_bound_2 = ((1) ? (unsigned int)64 : num_groups);
            unsigned int max_m_blocks_2 = (shape_m + 32 - 1) / 32;
            unsigned int max_total_tiles_2 = group_bound_2 * max_m_blocks_2 * grid_n_2;
            unsigned int max_scheduler_iters_2 = (max_total_tiles_2 + num_bids_u_2 - 1) / num_bids_u_2;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (int current_iter_2 = 0; current_iter_2 < max_scheduler_iters_2; current_iter_2++) {
                unsigned int next_block_idx_2 = (unsigned int)current_iter_2 * num_bids_u_2 + bid_u_2;
                int has_block_2 = 0;
                #pragma unroll 1
                for (int scan_g_2 = current_group_idx_2; scan_g_2 < group_bound_2; scan_g_2++) {
                    unsigned int group_m_4 = (unsigned int)masked_m[scan_g_2];
                    unsigned int m_blocks_g_2 = (group_m_4 + 32 - 1) / 32;
                    unsigned int next_m_cumsum_2 = current_m_cumsum_2 + m_blocks_g_2;
                    if (next_block_idx_2 < next_m_cumsum_2 * grid_n_2) {
                        current_group_idx_2 = scan_g_2;
                        has_block_2 = 1;
                        break;
                    }
                    current_m_cumsum_2 = next_m_cumsum_2;
                }
                if (has_block_2 == 0) {
                    break;
                }
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < num_k_blocks_2; iter_k_2++) {
                    mbarrier_wait(tma_full_addr + (transpose_stage) * 8, _phase_tma_full);
                    if (iter_k_2 % 2 == 0) {
                        unsigned int _sf_v[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[0])) : "r"(smem_sfa_addr + transpose_stage * 37888 + (unsigned int)(((lane >> 3 ^ 0) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[1])) : "r"(smem_sfa_addr + transpose_stage * 37888 + (unsigned int)(((lane >> 3 ^ 1) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[2])) : "r"(smem_sfa_addr + transpose_stage * 37888 + (unsigned int)(((lane >> 3 ^ 2) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[3])) : "r"(smem_sfa_addr + transpose_stage * 37888 + (unsigned int)(((lane >> 3 ^ 3) * 32 + lane) * 4)));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfa_addr + transpose_stage * 37888 + (unsigned int)((lane * 4 + (lane >> 3 ^ 0)) * 4)), "r"((_sf_v[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfa_addr + transpose_stage * 37888 + (unsigned int)((lane * 4 + (lane >> 3 ^ 1)) * 4)), "r"((_sf_v[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfa_addr + transpose_stage * 37888 + (unsigned int)((lane * 4 + (lane >> 3 ^ 2)) * 4)), "r"((_sf_v[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfa_addr + transpose_stage * 37888 + (unsigned int)((lane * 4 + (lane >> 3 ^ 3)) * 4)), "r"((_sf_v[3])));
                        unsigned int _sf_v_0[4];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[0])) : "r"(smem_sfb_addr + transpose_stage * 37888 + (unsigned int)(((lane >> 3 ^ 0) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[1])) : "r"(smem_sfb_addr + transpose_stage * 37888 + (unsigned int)(((lane >> 3 ^ 1) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[2])) : "r"(smem_sfb_addr + transpose_stage * 37888 + (unsigned int)(((lane >> 3 ^ 2) * 32 + lane) * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[3])) : "r"(smem_sfb_addr + transpose_stage * 37888 + (unsigned int)(((lane >> 3 ^ 3) * 32 + lane) * 4)));
                        __syncwarp();
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfb_addr + transpose_stage * 37888 + (unsigned int)((lane * 4 + (lane >> 3 ^ 0)) * 4)), "r"((_sf_v_0[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfb_addr + transpose_stage * 37888 + (unsigned int)((lane * 4 + (lane >> 3 ^ 1)) * 4)), "r"((_sf_v_0[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfb_addr + transpose_stage * 37888 + (unsigned int)((lane * 4 + (lane >> 3 ^ 2)) * 4)), "r"((_sf_v_0[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfb_addr + transpose_stage * 37888 + (unsigned int)((lane * 4 + (lane >> 3 ^ 3)) * 4)), "r"((_sf_v_0[3])));
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((scale_ready_addr + (transpose_stage) * 8) & 0xFEFFFFFF) : "memory");
                    transpose_stage += 1;
                    if (transpose_stage == 6) { transpose_stage = 0; _phase_tma_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            unsigned int epi_stage_1 = 0;
            unsigned int tma_stage_1 = 0;
            unsigned int grid_n_3 = ((1) ? (unsigned int)32 : N / 128);
            unsigned int num_bids_u_3 = (unsigned int)num_bids;
            unsigned int bid_u_3 = (unsigned int)bid;
            unsigned int current_group_idx_3 = 0;
            unsigned int current_m_cumsum_3 = 0;
            unsigned int group_bound_3 = ((1) ? (unsigned int)64 : num_groups);
            unsigned int max_m_blocks_3 = (shape_m + 32 - 1) / 32;
            unsigned int max_total_tiles_3 = group_bound_3 * max_m_blocks_3 * grid_n_3;
            unsigned int max_scheduler_iters_3 = (max_total_tiles_3 + num_bids_u_3 - 1) / num_bids_u_3;
            const int epi_warp = warp % 4;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (int current_iter_3 = 0; current_iter_3 < max_scheduler_iters_3; current_iter_3++) {
                unsigned int next_block_idx_3 = (unsigned int)current_iter_3 * num_bids_u_3 + bid_u_3;
                int has_block_3 = 0;
                unsigned int m_blocks_g_3 = 0;
                #pragma unroll 1
                for (int scan_g_3 = current_group_idx_3; scan_g_3 < group_bound_3; scan_g_3++) {
                    unsigned int group_m_5 = (unsigned int)masked_m[scan_g_3];
                    m_blocks_g_3 = (group_m_5 + 32 - 1) / 32;
                    unsigned int next_m_cumsum_3 = current_m_cumsum_3 + m_blocks_g_3;
                    if (next_block_idx_3 < next_m_cumsum_3 * grid_n_3) {
                        current_group_idx_3 = scan_g_3;
                        has_block_3 = 1;
                        break;
                    }
                    current_m_cumsum_3 = next_m_cumsum_3;
                }
                if (has_block_3 == 0) {
                    break;
                }
                unsigned int block_idx_2 = next_block_idx_3 - current_m_cumsum_3 * grid_n_3;
                unsigned int blocks_per_l2_group_2 = m_blocks_g_3;
                unsigned int l2_group_2 = block_idx_2 / blocks_per_l2_group_2;
                unsigned int first_n_block_2 = l2_group_2;
                unsigned int in_l2_group_2 = block_idx_2 % blocks_per_l2_group_2;
                unsigned int remaining_n_blocks_2 = grid_n_3 - first_n_block_2;
                unsigned int n_l2_limit_2 = (unsigned int)1;
                unsigned int n_blocks_in_group_2 = ((remaining_n_blocks_2 > n_l2_limit_2) ? n_l2_limit_2 : remaining_n_blocks_2);
                unsigned int m_block_2 = in_l2_group_2 / n_blocks_in_group_2;
                unsigned int n_block_1 = first_n_block_2 + in_l2_group_2 % n_blocks_in_group_2;
                mbarrier_wait(mainloop_done_addr + (epi_stage_1) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int off_n_1 = n_block_1 * 128;
                unsigned int flat_row = current_group_idx_3 * shape_m + m_block_2 * 32;
                int tmem_base = taddr + (unsigned int)((int)epi_stage_1 * 32);
                #pragma unroll
                for (int strip = 0; strip < 2; strip++) {
                    int col_strip = strip * 16;
                    if (epi_warp == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("barrier.sync 15, 128;" ::: "memory");
                    int stage_base = epi_staging_addr + tma_stage_1 * 4096;
                    #pragma unroll
                    for (int atom = 0; atom < 2; atom++) {
                        int tmem_addr = tmem_base + col_strip + atom * 8;
                        float _tmem_load_0[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                            : "r"(tmem_addr));
                        float _tmem_load_1[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                            : "r"(tmem_addr | 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                        __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                        __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_tmem_load_1[0], _tmem_load_1[1]));
                        __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                        int outer_atom = epi_warp / 2 * 2048;
                        int inner_atom = atom * 1024;
                        int lane_row = lane % 8;
                        int lane_col = epi_warp % 2 * 4 + lane / 8;
                        int smem_ptr = stage_base + outer_atom + inner_atom + lane_row * 128 + (lane_col ^ lane_row) * 16;
                        uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(smem_ptr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_0)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_1)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_2)), "r"(*reinterpret_cast<const uint32_t*>(&_bf16x2_3))
                            : "memory");
                    }
                    if (strip == 1) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((epilogue_done_addr + (epi_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 15, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            if (m_block_2 * 32 + (unsigned int)col_strip < shape_m) {
                                tma_store_2d((&C_tma), off_n_1, flat_row + (unsigned int)col_strip, stage_base);
                                tma_store_2d((&C_tma), off_n_1 + 64, flat_row + (unsigned int)col_strip, stage_base + 2048);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    }
                    __syncwarp();
                }
                epi_stage_1 += 1;
                if (epi_stage_1 == 2) { epi_stage_1 = 0; _phase_mainloop_done ^= 1; }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 2) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
