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
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAINLOOP_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 49152
#define SMEM_SMEM_B0_OFF 17408
#define SMEM_SMEM_B0_STAGE_BYTES 16384
#define SMEM_SMEM_B0_STRIDE 49152
#define SMEM_SMEM_B1_OFF 33792
#define SMEM_SMEM_B1_STAGE_BYTES 16384
#define SMEM_SMEM_B1_STRIDE 49152
#define SMEM_TOTAL 197632
#define BLOCK_M 128
#define BLOCK_N 256
#define BLOCK_K 64
#define MMA_K 16
#define CTA_GROUP 2
#define NUM_STAGES 4
#define TILE_K 512
#define NUM_EPILOGUE_WARPS 4
#define CLUSTER_M 256
#define EPI_CHUNK 16

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






__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
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


extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_cake_moe_grouped_gemm_d77aff0f30a2661aecc8(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, float* __restrict__ C, int* __restrict__ offs, uint8_t* __restrict__ tensormap_workspace, float* __restrict__ partials, int tail_splits, int raster_rows, int num_groups, int sum_m, int N, int K, int ldc, int stride_e)
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
    #define mma_done_addr (mbar_base + 32)
    #define mainloop_done_addr (mbar_base + 64)
    #define epilogue_done_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b0_addr = smem + 17408;
    __nv_bfloat16* smem_b1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_b1_addr = smem + 33792;

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 4 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            // mma_done: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // epilogue_done: 1 barriers, init_count=8
            mbarrier_init(smem + 72, 8);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 0) {
        int _tmem_hold = smem + 80;
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
            unsigned int grid_n_u = (unsigned int)(N / BLOCK_N);
            unsigned int grid_k_u = (unsigned int)(K / TILE_K);
            unsigned int tiles_per_group = grid_n_u * grid_k_u;
            unsigned int total_tiles = (unsigned int)num_groups * tiles_per_group;
            unsigned int num_clusters_u = (unsigned int)num_clusters;
            unsigned int cluster_u = (unsigned int)cluster_id;
            int cta_linear = (int)cluster_u * CTA_GROUP + cta_rank;
            int slot_group = -1;
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&A));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_linear * 256)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&B));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_linear * 256 + 128)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256))) : "memory");
                asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + 128))) : "memory");
                unsigned int reg_end = total_tiles;
                unsigned int tail_base = total_tiles / num_clusters_u * num_clusters_u;
                unsigned int full_waves = tail_base / num_clusters_u;
                if (tail_splits > 1) {
                    reg_end = tail_base;
                }
                #pragma unroll 1
                for (unsigned int t = cluster_u; t < reg_end; t += num_clusters_u) {
                    unsigned int w = t / num_clusters_u;
                    unsigned int p = t - w * num_clusters_u;
                    if (w % 2 == 1) {
                        if (w < full_waves) {
                            p = num_clusters_u - 1 - p;
                        }
                    }
                    unsigned int ts = w * num_clusters_u + p;
                    unsigned int e_u = ts / tiles_per_group;
                    unsigned int rem = ts - e_u * tiles_per_group;
                    unsigned int n_block = 0;
                    unsigned int k_block = 0;
                    {
                        unsigned int rr_n_block = (unsigned int)raster_rows;
                        unsigned int nc_n_block = rem / grid_k_u;
                        unsigned int kr_n_block = rem / grid_n_u;
                        n_block = nc_n_block * (1 - rr_n_block) + (rem - kr_n_block * grid_n_u) * rr_n_block;
                        k_block = (rem - nc_n_block * grid_k_u) * (1 - rr_n_block) + kr_n_block * rr_n_block;
                    }
                    int e_i = (int)e_u;
                    int end_e = offs[e_i];
                    int prev_i = e_i - 1;
                    if (prev_i < 0) {
                        prev_i = 0;
                    }
                    int start_e = offs[prev_i];
                    if (e_i == 0) {
                        start_e = 0;
                    }
                    int k_iters = (end_e - start_e + (BLOCK_K - 1)) / BLOCK_K;
                    int k_iters_eff = k_iters;
                    int m_base = start_e;
                    if (k_iters == 0) {
                        k_iters_eff = 1;
                        m_base = sum_m;
                    }
                    if (k_iters > 0) {
                        if (e_i != slot_group) {
                            asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256))), "r"((uint32_t)(end_e)) : "memory");
                            asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + 128))), "r"((uint32_t)(end_e)) : "memory");
                            asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256))) : "memory");
                            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + 128))) : "memory");
                            slot_group = e_i;
                        }
                    }
                    int n_chunk0 = (int)n_block * (BLOCK_N / 64) + cta_rank * 2;
                    int k_chunk0 = (int)k_block * (TILE_K / 64) + cta_rank * 2;
                    #pragma unroll 1
                    for (int iter_m = 0; iter_m < k_iters_eff; iter_m++) {
                        int m_coord = m_base + iter_m * BLOCK_K;
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152, tensormap_workspace + (cta_linear * 256), 0, m_coord, n_chunk0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152 + 8192, tensormap_workspace + (cta_linear * 256), 0, m_coord, n_chunk0 + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b0_addr + load_stage * 49152, tensormap_workspace + (cta_linear * 256 + 128), 0, m_coord, k_chunk0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b0_addr + load_stage * 49152 + 8192, tensormap_workspace + (cta_linear * 256 + 128), 0, m_coord, k_chunk0 + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b1_addr + load_stage * 49152, tensormap_workspace + (cta_linear * 256 + 128), 0, m_coord, k_chunk0 + 4, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b1_addr + load_stage * 49152 + 8192, tensormap_workspace + (cta_linear * 256 + 128), 0, m_coord, k_chunk0 + 5, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(49152)) : "memory");
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                if (tail_splits > 1) {
                    unsigned int num_tail_tl = total_tiles - tail_base;
                    unsigned int units_tl = num_tail_tl * (unsigned int)tail_splits;
                    #pragma unroll 1
                    for (unsigned int u_tl = cluster_u; u_tl < units_tl; u_tl += num_clusters_u) {
                        unsigned int s_u_tl = u_tl / num_tail_tl;
                        unsigned int j_u_tl = u_tl - s_u_tl * num_tail_tl;
                        unsigned int tt = tail_base + j_u_tl;
                        unsigned int e_ut = tt / tiles_per_group;
                        unsigned int remt = tt - e_ut * tiles_per_group;
                        unsigned int n_blockt = 0;
                        unsigned int k_blockt = 0;
                        {
                            unsigned int rr_n_blockt = (unsigned int)raster_rows;
                            unsigned int nc_n_blockt = remt / grid_k_u;
                            unsigned int kr_n_blockt = remt / grid_n_u;
                            n_blockt = nc_n_blockt * (1 - rr_n_blockt) + (remt - kr_n_blockt * grid_n_u) * rr_n_blockt;
                            k_blockt = (remt - nc_n_blockt * grid_k_u) * (1 - rr_n_blockt) + kr_n_blockt * rr_n_blockt;
                        }
                        int e_it = (int)e_ut;
                        int end_et = offs[e_it];
                        int prev_it = e_it - 1;
                        if (prev_it < 0) {
                            prev_it = 0;
                        }
                        int start_et = offs[prev_it];
                        if (e_it == 0) {
                            start_et = 0;
                        }
                        int k_iterst = (end_et - start_et + (BLOCK_K - 1)) / BLOCK_K;
                        int k_iters_efft = k_iterst;
                        int m_baset = start_et;
                        if (k_iterst == 0) {
                            k_iters_efft = 1;
                            m_baset = sum_m;
                        }
                        int s_i_tl = (int)s_u_tl;
                        int lo_tl = s_i_tl * k_iters_efft / tail_splits;
                        int hi_tl = (s_i_tl + 1) * k_iters_efft / tail_splits;
                        int n_steps_tl = hi_tl - lo_tl;
                        if (n_steps_tl > 0) {
                            if (k_iterst > 0) {
                                if (e_it != slot_group) {
                                    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256))), "r"((uint32_t)(end_et)) : "memory");
                                    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + 128))), "r"((uint32_t)(end_et)) : "memory");
                                    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                                    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256))) : "memory");
                                    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + 128))) : "memory");
                                    slot_group = e_it;
                                }
                            }
                            int n_chunk0t = (int)n_blockt * (BLOCK_N / 64) + cta_rank * 2;
                            int k_chunk0t = (int)k_blockt * (TILE_K / 64) + cta_rank * 2;
                            #pragma unroll 1
                            for (int iter_mt = 0; iter_mt < n_steps_tl; iter_mt++) {
                                int m_coordt = m_baset + (lo_tl + iter_mt) * BLOCK_K;
                                mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                                tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152, tensormap_workspace + (cta_linear * 256), 0, m_coordt, n_chunk0t, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                                tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152 + 8192, tensormap_workspace + (cta_linear * 256), 0, m_coordt, n_chunk0t + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                                tma_3d_gmem2smem_cta2(smem_b0_addr + load_stage * 49152, tensormap_workspace + (cta_linear * 256 + 128), 0, m_coordt, k_chunk0t, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                                tma_3d_gmem2smem_cta2(smem_b0_addr + load_stage * 49152 + 8192, tensormap_workspace + (cta_linear * 256 + 128), 0, m_coordt, k_chunk0t + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                                tma_3d_gmem2smem_cta2(smem_b1_addr + load_stage * 49152, tensormap_workspace + (cta_linear * 256 + 128), 0, m_coordt, k_chunk0t + 4, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                                tma_3d_gmem2smem_cta2(smem_b1_addr + load_stage * 49152 + 8192, tensormap_workspace + (cta_linear * 256 + 128), 0, m_coordt, k_chunk0t + 5, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(49152)) : "memory");
                                load_stage += 1;
                                if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int grid_n_m = (unsigned int)(N / BLOCK_N);
            unsigned int grid_k_m = (unsigned int)(K / TILE_K);
            unsigned int tiles_per_group_m = grid_n_m * grid_k_m;
            unsigned int total_tiles_m = (unsigned int)num_groups * tiles_per_group_m;
            unsigned int num_clusters_m = (unsigned int)num_clusters;
            unsigned int cluster_m = (unsigned int)cluster_id;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                unsigned int reg_end_m = total_tiles_m;
                unsigned int tail_base_m = total_tiles_m / num_clusters_m * num_clusters_m;
                unsigned int full_waves_m = tail_base_m / num_clusters_m;
                if (tail_splits > 1) {
                    reg_end_m = tail_base_m;
                }
                #pragma unroll 1
                for (unsigned int t_m = cluster_m; t_m < reg_end_m; t_m += num_clusters_m) {
                    unsigned int w_m = t_m / num_clusters_m;
                    unsigned int p_m = t_m - w_m * num_clusters_m;
                    if (w_m % 2 == 1) {
                        if (w_m < full_waves_m) {
                            p_m = num_clusters_m - 1 - p_m;
                        }
                    }
                    unsigned int ts_m = w_m * num_clusters_m + p_m;
                    unsigned int e_um = ts_m / tiles_per_group_m;
                    int e_im = (int)e_um;
                    int end_m = offs[e_im];
                    int prev_m = e_im - 1;
                    if (prev_m < 0) {
                        prev_m = 0;
                    }
                    int start_m = offs[prev_m];
                    if (e_im == 0) {
                        start_m = 0;
                    }
                    int k_iters_m = (end_m - start_m + (BLOCK_K - 1)) / BLOCK_K;
                    if (k_iters_m == 0) {
                        k_iters_m = 1;
                    }
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_m = 0; iter_k_m < k_iters_m; iter_k_m++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((iter_k_m == 0) ? 1 : 0);
                        int _mma_a_lo_0 = ((((smem_a_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _mma_b_lo_0 = ((((smem_b0_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272729232;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)));
                        int _mma_a_lo_1 = ((((smem_a_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _mma_b_lo_1 = ((((smem_b1_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272729232;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accum + (256))), "r"(((init_flag) ? 0 : 1)));
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                    _phase_epilogue_done ^= 1;
                }
                if (tail_splits > 1) {
                    unsigned int num_tail_tm = total_tiles_m - tail_base_m;
                    unsigned int units_tm = num_tail_tm * (unsigned int)tail_splits;
                    #pragma unroll 1
                    for (unsigned int u_tm = cluster_m; u_tm < units_tm; u_tm += num_clusters_m) {
                        unsigned int s_u_tm = u_tm / num_tail_tm;
                        unsigned int j_u_tm = u_tm - s_u_tm * num_tail_tm;
                        unsigned int tt_m = tail_base_m + j_u_tm;
                        unsigned int e_umt = tt_m / tiles_per_group_m;
                        int e_imt = (int)e_umt;
                        int end_mt = offs[e_imt];
                        int prev_mt = e_imt - 1;
                        if (prev_mt < 0) {
                            prev_mt = 0;
                        }
                        int start_mt = offs[prev_mt];
                        if (e_imt == 0) {
                            start_mt = 0;
                        }
                        int k_iters_mt = (end_mt - start_mt + (BLOCK_K - 1)) / BLOCK_K;
                        if (k_iters_mt == 0) {
                            k_iters_mt = 1;
                        }
                        int s_i_tm = (int)s_u_tm;
                        int lo_tm = s_i_tm * k_iters_mt / tail_splits;
                        int hi_tm = (s_i_tm + 1) * k_iters_mt / tail_splits;
                        int n_steps_tm = hi_tm - lo_tm;
                        if (n_steps_tm > 0) {
                            mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                            #pragma unroll 1
                            for (int iter_k_mt = 0; iter_k_mt < n_steps_tm; iter_k_mt++) {
                                mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int init_flagt = ((iter_k_mt == 0) ? 1 : 0);
                                int _mma_a_lo_2 = ((((smem_a_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                                int _mma_b_lo_2 = ((((smem_b0_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272729232;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_accum), "r"(((init_flagt) ? 0 : 1)));
                                int _mma_a_lo_3 = ((((smem_a_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                                int _mma_b_lo_3 = ((((smem_b1_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272729232;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_accum + (256))), "r"(((init_flagt) ? 0 : 1)));
                                elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                                mma_tma_stage += 1;
                                if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            }
                            elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                            _phase_epilogue_done ^= 1;
                        }
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            unsigned int grid_n_e = (unsigned int)(N / BLOCK_N);
            unsigned int grid_k_e = (unsigned int)(K / TILE_K);
            unsigned int tiles_per_group_e = grid_n_e * grid_k_e;
            unsigned int total_tiles_e = (unsigned int)num_groups * tiles_per_group_e;
            unsigned int num_clusters_e = (unsigned int)num_clusters;
            unsigned int cluster_e = (unsigned int)cluster_id;
            unsigned int reg_end_e = total_tiles_e;
            unsigned int tail_base_e = total_tiles_e / num_clusters_e * num_clusters_e;
            unsigned int full_waves_e = tail_base_e / num_clusters_e;
            if (tail_splits > 1) {
                reg_end_e = tail_base_e;
            }
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int t_e = cluster_e; t_e < reg_end_e; t_e += num_clusters_e) {
                unsigned int w_e = t_e / num_clusters_e;
                unsigned int p_e = t_e - w_e * num_clusters_e;
                if (w_e % 2 == 1) {
                    if (w_e < full_waves_e) {
                        p_e = num_clusters_e - 1 - p_e;
                    }
                }
                unsigned int ts_e = w_e * num_clusters_e + p_e;
                unsigned int e_ue = ts_e / tiles_per_group_e;
                unsigned int rem_e = ts_e - e_ue * tiles_per_group_e;
                unsigned int n_block_e = 0;
                unsigned int k_block_e = 0;
                {
                    unsigned int rr_n_block_e = (unsigned int)raster_rows;
                    unsigned int nc_n_block_e = rem_e / grid_k_e;
                    unsigned int kr_n_block_e = rem_e / grid_n_e;
                    n_block_e = nc_n_block_e * (1 - rr_n_block_e) + (rem_e - kr_n_block_e * grid_n_e) * rr_n_block_e;
                    k_block_e = (rem_e - nc_n_block_e * grid_k_e) * (1 - rr_n_block_e) + kr_n_block_e * rr_n_block_e;
                }
                int row_n = (int)n_block_e * BLOCK_N + cta_rank * BLOCK_M + epi_tid;
                int col0 = (int)k_block_e * TILE_K;
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int n_chunk = 0; n_chunk < TILE_K / EPI_CHUNK; n_chunk++) {
                    int row = cta_rank * 128 + epi_warp * 32;
                    int col = n_chunk * EPI_CHUNK;
                    int tmem_addr = taddr + (unsigned int)(row << 16) + (unsigned int)col;
                    float _tmem_load_0[16];
                    tmem_ld_x16(&_tmem_load_0[0], tmem_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    {
                        {
                            float4 _v4 = make_float4(_tmem_load_0[0 + 0], _tmem_load_0[0 + 1], _tmem_load_0[0 + 2], _tmem_load_0[0 + 3]);
                            *reinterpret_cast<float4*>(C + ((int)e_ue * stride_e + row_n * ldc + col0 + n_chunk * EPI_CHUNK) + 0) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(_tmem_load_0[4 + 0], _tmem_load_0[4 + 1], _tmem_load_0[4 + 2], _tmem_load_0[4 + 3]);
                            *reinterpret_cast<float4*>(C + ((int)e_ue * stride_e + row_n * ldc + col0 + n_chunk * EPI_CHUNK + 4) + 0) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(_tmem_load_0[8 + 0], _tmem_load_0[8 + 1], _tmem_load_0[8 + 2], _tmem_load_0[8 + 3]);
                            *reinterpret_cast<float4*>(C + ((int)e_ue * stride_e + row_n * ldc + col0 + n_chunk * EPI_CHUNK + 8) + 0) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(_tmem_load_0[12 + 0], _tmem_load_0[12 + 1], _tmem_load_0[12 + 2], _tmem_load_0[12 + 3]);
                            *reinterpret_cast<float4*>(C + ((int)e_ue * stride_e + row_n * ldc + col0 + n_chunk * EPI_CHUNK + 12) + 0) = _v4;
                        }
                    }
                }
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
            }
            if (tail_splits > 1) {
                unsigned int num_tail_te = total_tiles_e - tail_base_e;
                unsigned int units_te = num_tail_te * (unsigned int)tail_splits;
                #pragma unroll 1
                for (unsigned int u_te = cluster_e; u_te < units_te; u_te += num_clusters_e) {
                    unsigned int s_u_te = u_te / num_tail_te;
                    unsigned int j_u_te = u_te - s_u_te * num_tail_te;
                    unsigned int tt_e = tail_base_e + j_u_te;
                    unsigned int e_uet = tt_e / tiles_per_group_e;
                    unsigned int rem_et = tt_e - e_uet * tiles_per_group_e;
                    unsigned int n_block_et = 0;
                    unsigned int k_block_et = 0;
                    {
                        unsigned int rr_n_block_et = (unsigned int)raster_rows;
                        unsigned int nc_n_block_et = rem_et / grid_k_e;
                        unsigned int kr_n_block_et = rem_et / grid_n_e;
                        n_block_et = nc_n_block_et * (1 - rr_n_block_et) + (rem_et - kr_n_block_et * grid_n_e) * rr_n_block_et;
                        k_block_et = (rem_et - nc_n_block_et * grid_k_e) * (1 - rr_n_block_et) + kr_n_block_et * rr_n_block_et;
                    }
                    int row_nt = (int)n_block_et * BLOCK_N + cta_rank * BLOCK_M + epi_tid;
                    int col0t = (int)k_block_et * TILE_K;
                    int e_kb_te = (int)e_uet;
                    int end_te = offs[e_kb_te];
                    int prev_te = e_kb_te - 1;
                    if (prev_te < 0) {
                        prev_te = 0;
                    }
                    int start_te = offs[prev_te];
                    if (e_kb_te == 0) {
                        start_te = 0;
                    }
                    int kb_te = (end_te - start_te + (BLOCK_K - 1)) / BLOCK_K;
                    if (kb_te == 0) {
                        kb_te = 1;
                    }
                    int s_i_te = (int)s_u_te;
                    int lo_te = s_i_te * kb_te / tail_splits;
                    int hi_te = (s_i_te + 1) * kb_te / tail_splits;
                    int n_steps_te = hi_te - lo_te;
                    int whole_te = 0;
                    if (lo_te == 0) {
                        if (hi_te == kb_te) {
                            whole_te = 1;
                        }
                    }
                    int slot_te = (int)j_u_te * tail_splits + s_i_te;
                    if (n_steps_te > 0) {
                        mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int n_chunkt = 0; n_chunkt < TILE_K / EPI_CHUNK; n_chunkt++) {
                            int rowt = cta_rank * 128 + epi_warp * 32;
                            int colt = n_chunkt * EPI_CHUNK;
                            int tmem_addrt = taddr + (unsigned int)(rowt << 16) + (unsigned int)colt;
                            float _tmem_load_1[16];
                            tmem_ld_x16(&_tmem_load_1[0], tmem_addrt);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            if (whole_te == 1) {
                                {
                                    {
                                        float4 _v4 = make_float4(_tmem_load_1[0 + 0], _tmem_load_1[0 + 1], _tmem_load_1[0 + 2], _tmem_load_1[0 + 3]);
                                        *reinterpret_cast<float4*>(C + ((int)e_uet * stride_e + row_nt * ldc + col0t + n_chunkt * EPI_CHUNK) + 0) = _v4;
                                    }
                                    {
                                        float4 _v4 = make_float4(_tmem_load_1[4 + 0], _tmem_load_1[4 + 1], _tmem_load_1[4 + 2], _tmem_load_1[4 + 3]);
                                        *reinterpret_cast<float4*>(C + ((int)e_uet * stride_e + row_nt * ldc + col0t + n_chunkt * EPI_CHUNK + 4) + 0) = _v4;
                                    }
                                    {
                                        float4 _v4 = make_float4(_tmem_load_1[8 + 0], _tmem_load_1[8 + 1], _tmem_load_1[8 + 2], _tmem_load_1[8 + 3]);
                                        *reinterpret_cast<float4*>(C + ((int)e_uet * stride_e + row_nt * ldc + col0t + n_chunkt * EPI_CHUNK + 8) + 0) = _v4;
                                    }
                                    {
                                        float4 _v4 = make_float4(_tmem_load_1[12 + 0], _tmem_load_1[12 + 1], _tmem_load_1[12 + 2], _tmem_load_1[12 + 3]);
                                        *reinterpret_cast<float4*>(C + ((int)e_uet * stride_e + row_nt * ldc + col0t + n_chunkt * EPI_CHUNK + 12) + 0) = _v4;
                                    }
                                }
                            }
                            if (whole_te == 0) {
                                {
                                    float4 _v4 = make_float4(_tmem_load_1[0 + 0], _tmem_load_1[0 + 1], _tmem_load_1[0 + 2], _tmem_load_1[0 + 3]);
                                    *reinterpret_cast<float4*>(partials + (slot_te * (CLUSTER_M * TILE_K) + (cta_rank * BLOCK_M + epi_tid) * TILE_K + n_chunkt * EPI_CHUNK) + 0) = _v4;
                                }
                                {
                                    float4 _v4 = make_float4(_tmem_load_1[4 + 0], _tmem_load_1[4 + 1], _tmem_load_1[4 + 2], _tmem_load_1[4 + 3]);
                                    *reinterpret_cast<float4*>(partials + (slot_te * (CLUSTER_M * TILE_K) + (cta_rank * BLOCK_M + epi_tid) * TILE_K + n_chunkt * EPI_CHUNK + 4) + 0) = _v4;
                                }
                                {
                                    float4 _v4 = make_float4(_tmem_load_1[8 + 0], _tmem_load_1[8 + 1], _tmem_load_1[8 + 2], _tmem_load_1[8 + 3]);
                                    *reinterpret_cast<float4*>(partials + (slot_te * (CLUSTER_M * TILE_K) + (cta_rank * BLOCK_M + epi_tid) * TILE_K + n_chunkt * EPI_CHUNK + 8) + 0) = _v4;
                                }
                                {
                                    float4 _v4 = make_float4(_tmem_load_1[12 + 0], _tmem_load_1[12 + 1], _tmem_load_1[12 + 2], _tmem_load_1[12 + 3]);
                                    *reinterpret_cast<float4*>(partials + (slot_te * (CLUSTER_M * TILE_K) + (cta_rank * BLOCK_M + epi_tid) * TILE_K + n_chunkt * EPI_CHUNK + 12) + 0) = _v4;
                                }
                            }
                        }
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        _phase_mainloop_done ^= 1;
                    }
                }
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
