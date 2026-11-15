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

#define BLOCK_M 128
#define BLOCK_N 256
#define BLOCK_K 64
#define MMA_K 16
#define CTA_GROUP 2
#define NUM_STAGES 4
#define TILE_K 512
#define TMEM_NCOLS 512
#define NUM_EPILOGUE_WARPS 8
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


extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_cake_moe_grouped_det_gemm_412177509e084ace65e7_wgrad_out32(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, float* __restrict__ C, int* __restrict__ offs, uint8_t* __restrict__ tensormap_workspace, float* __restrict__ partials, int tail_splits, int raster_rows, int num_groups, int sum_m, int N, int K, int ldc, int stride_e)
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
    #define epi_free0_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_b0_addr = smem + 17408;
    __nv_bfloat16* smem_b1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_b1_addr = smem + 33792;

    // Mbarrier init (11 barriers at smem_raw[0..88))

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
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 72, 16);
            // epi_free0: 1 barriers, init_count=16
            mbarrier_init(smem + 80, 16);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 88);
    if (warp == 0) {
        int _tmem_hold = smem + 88;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_accum = taddr;

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
            int pair_stride = (int)num_bids * 256;
            int slot_pair = 0;
            int slot_group = -1;
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                #pragma unroll
                for (int pair = 0; pair < 9; pair++) {
                    {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&A));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_linear * 256 + pair * pair_stride)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                    {
                const uint64_t* __tm_src = reinterpret_cast<const uint64_t*>((&B));
                uint64_t* __tm_dst = reinterpret_cast<uint64_t*>((uint64_t)(tensormap_workspace + (cta_linear * 256 + pair * pair_stride + 128)));
                #pragma unroll
                for (int __tm_i = 0; __tm_i < 16; ++__tm_i) {
                    __tm_dst[__tm_i] = __tm_src[__tm_i];
                }
            }
                }
                asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                #pragma unroll
                for (int pair_1 = 0; pair_1 < 9; pair_1++) {
                    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + pair_1 * pair_stride))) : "memory");
                    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + pair_1 * pair_stride + 128))) : "memory");
                }
                unsigned int reg_end = total_tiles;
                unsigned int tail_base = total_tiles / num_clusters_u * num_clusters_u;
                unsigned int full_waves = tail_base / num_clusters_u;
                reg_end = tail_base;
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
                    unsigned int rr_n_block = 0;
                    unsigned int nc_n_block = rem / grid_k_u;
                    unsigned int kr_n_block = rem / grid_n_u;
                    n_block = nc_n_block * (1 - rr_n_block) + (rem - kr_n_block * grid_n_u) * rr_n_block;
                    k_block = (rem - nc_n_block * grid_k_u) * (1 - rr_n_block) + kr_n_block * rr_n_block;
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
                            slot_pair = slot_pair + 1;
                            if (slot_pair == 9) {
                                slot_pair = 0;
                            }
                            int next_pair = slot_pair + 1;
                            if (next_pair == 9) {
                                next_pair = 0;
                            }
                            asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(tensormap_workspace + (cta_linear * 256 + next_pair * pair_stride))), "r"(static_cast<unsigned int>(0)) : "memory");
                            asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(tensormap_workspace + (cta_linear * 256 + next_pair * pair_stride + 128))), "r"(static_cast<unsigned int>(0)) : "memory");
                            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + next_pair * pair_stride))));
                            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + next_pair * pair_stride + 128))));
                            asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride))), "r"((uint32_t)(end_e)) : "memory");
                            asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride + 128))), "r"((uint32_t)(end_e)) : "memory");
                            asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride))) : "memory");
                            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride + 128))) : "memory");
                            slot_group = e_i;
                        }
                    }
                    int n_chunk0 = (int)n_block * (BLOCK_N / 64) + cta_rank * 2;
                    int k_chunk0 = (int)k_block * (TILE_K / 64) + cta_rank * 2;
                    #pragma unroll 1
                    for (int iter_m = 0; iter_m < k_iters_eff; iter_m++) {
                        int m_coord = m_base + iter_m * BLOCK_K;
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152, tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride), 0, m_coord, n_chunk0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152 + 8192, tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride), 0, m_coord, n_chunk0 + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b0_addr + load_stage * 49152, tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride + 128), 0, m_coord, k_chunk0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b0_addr + load_stage * 49152 + 8192, tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride + 128), 0, m_coord, k_chunk0 + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b1_addr + load_stage * 49152, tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride + 128), 0, m_coord, k_chunk0 + 4, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b1_addr + load_stage * 49152 + 8192, tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride + 128), 0, m_coord, k_chunk0 + 5, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(49152)) : "memory");
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                unsigned int total_h = total_tiles * 2;
                unsigned int tail_base_h = total_h / num_clusters_u * num_clusters_u;
                unsigned int num_tail_h = total_h - tail_base_h;
                unsigned int whole_end_h = total_h;
                unsigned int units_h = 0;
                if (tail_splits > 1) {
                    whole_end_h = tail_base_h;
                    units_h = num_tail_h * (unsigned int)tail_splits;
                }
                unsigned int num_whole_h = whole_end_h - tail_base * 2;
                #pragma unroll 1
                for (unsigned int v_h = cluster_u; v_h < num_whole_h + units_h; v_h += num_clusters_u) {
                    unsigned int hs_h = tail_base * 2 + v_h;
                    int s_h = 0;
                    int splits_h = 1;
                    if (v_h >= num_whole_h) {
                        unsigned int u_h = v_h - num_whole_h;
                        unsigned int s_u_h = u_h / num_tail_h;
                        hs_h = tail_base_h + (u_h - s_u_h * num_tail_h);
                        s_h = (int)s_u_h;
                        splits_h = tail_splits;
                    }
                    unsigned int t_h = hs_h / 2;
                    unsigned int e_uh = t_h / tiles_per_group;
                    unsigned int rem_h = t_h - e_uh * tiles_per_group;
                    unsigned int n_block_h = rem_h / grid_k_u;
                    unsigned int k_block_h = (rem_h - n_block_h * grid_k_u) * 2 + (hs_h - t_h * 2);
                    int e_ih = (int)e_uh;
                    int end_eh = offs[e_ih];
                    int prev_ih = e_ih - 1;
                    if (prev_ih < 0) {
                        prev_ih = 0;
                    }
                    int start_eh = offs[prev_ih];
                    if (e_ih == 0) {
                        start_eh = 0;
                    }
                    int k_iters_h = (end_eh - start_eh + (BLOCK_K - 1)) / BLOCK_K;
                    int k_iters_effh = k_iters_h;
                    int m_baseh = start_eh;
                    if (k_iters_h == 0) {
                        k_iters_effh = 1;
                        m_baseh = sum_m;
                    }
                    int lo_h = s_h * k_iters_effh / splits_h;
                    int hi_h = (s_h + 1) * k_iters_effh / splits_h;
                    int n_steps_h = hi_h - lo_h;
                    if (n_steps_h > 0) {
                        if (k_iters_h > 0) {
                            if (e_ih != slot_group) {
                                slot_pair = slot_pair + 1;
                                if (slot_pair == 9) {
                                    slot_pair = 0;
                                }
                                int next_pair_h = slot_pair + 1;
                                if (next_pair_h == 9) {
                                    next_pair_h = 0;
                                }
                                asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(tensormap_workspace + (cta_linear * 256 + next_pair_h * pair_stride))), "r"(static_cast<unsigned int>(0)) : "memory");
                                asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(tensormap_workspace + (cta_linear * 256 + next_pair_h * pair_stride + 128))), "r"(static_cast<unsigned int>(0)) : "memory");
                                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + next_pair_h * pair_stride))));
                                asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + next_pair_h * pair_stride + 128))));
                                asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride))), "r"((uint32_t)(end_eh)) : "memory");
                                asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride + 128))), "r"((uint32_t)(end_eh)) : "memory");
                                asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
                                asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride))) : "memory");
                                asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"((uint64_t)(tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride + 128))) : "memory");
                                slot_group = e_ih;
                            }
                        }
                        int n_chunk0h = (int)n_block_h * (BLOCK_N / 64) + cta_rank * 2;
                        int k_chunk0h = (int)k_block_h * (BLOCK_N / 64) + cta_rank * 2;
                        #pragma unroll 1
                        for (int iter_mh = 0; iter_mh < n_steps_h; iter_mh++) {
                            int m_coordh = m_baseh + (lo_h + iter_mh) * BLOCK_K;
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152, tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride), 0, m_coordh, n_chunk0h, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152 + 8192, tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride), 0, m_coordh, n_chunk0h + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b0_addr + load_stage * 49152, tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride + 128), 0, m_coordh, k_chunk0h, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b0_addr + load_stage * 49152 + 8192, tensormap_workspace + (cta_linear * 256 + slot_pair * pair_stride + 128), 0, m_coordh, k_chunk0h + 1, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            load_stage += 1;
                            if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
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
            unsigned int grid_n_m = (unsigned int)(N / BLOCK_N);
            unsigned int grid_k_m = (unsigned int)(K / TILE_K);
            unsigned int tiles_per_group_m = grid_n_m * grid_k_m;
            unsigned int total_tiles_m = (unsigned int)num_groups * tiles_per_group_m;
            unsigned int num_clusters_m = (unsigned int)num_clusters;
            unsigned int cluster_m = (unsigned int)cluster_id;
            unsigned int _phase_epi_free0 = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                unsigned int reg_end_m = total_tiles_m;
                unsigned int tail_base_m = total_tiles_m / num_clusters_m * num_clusters_m;
                unsigned int full_waves_m = tail_base_m / num_clusters_m;
                reg_end_m = tail_base_m;
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
                    mbarrier_wait(epi_free0_addr, _phase_epi_free0);
                    if (0 < k_iters_m) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_0 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_1 = ((((smem_b0_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_accum), "r"(0) : "memory");
                    }
                    if (1 < k_iters_m) {
                        mbarrier_wait(tma_full_addr + (((mma_tma_stage + 1 < 4) ? mma_tma_stage + 1 : (unsigned int)0)) * 8, ((mma_tma_stage + 1 < 4) ? _phase_tma_full : _phase_tma_full ^ 1));
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_2 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (((mma_tma_stage + 1 < 4) ? mma_tma_stage + 1 : (unsigned int)0)) * 3072;
                        int _smem_desc_lo_3 = ((((smem_b0_addr) >> 4) & 0x7FFF) | 0x2000000) + (((mma_tma_stage + 1 < 4) ? mma_tma_stage + 1 : (unsigned int)0)) * 3072;
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_accum), "r"(1) : "memory");
                    }
                    if (2 < k_iters_m) {
                        mbarrier_wait(tma_full_addr + (((mma_tma_stage + 2 >= 4) ? mma_tma_stage + 2 - 4 : mma_tma_stage + 2)) * 8, _phase_tma_full ^ (unsigned int)(mma_tma_stage + 2 >= 4));
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_4 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (((mma_tma_stage + 2 >= 4) ? mma_tma_stage + 2 - 4 : mma_tma_stage + 2)) * 3072;
                        int _smem_desc_lo_5 = ((((smem_b0_addr) >> 4) & 0x7FFF) | 0x2000000) + (((mma_tma_stage + 2 >= 4) ? mma_tma_stage + 2 - 4 : mma_tma_stage + 2)) * 3072;
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
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_accum), "r"(1) : "memory");
                    }
                    if (3 < k_iters_m) {
                        mbarrier_wait(tma_full_addr + (((mma_tma_stage + 3 >= 4) ? mma_tma_stage + 3 - 4 : mma_tma_stage + 3)) * 8, _phase_tma_full ^ (unsigned int)(mma_tma_stage + 3 >= 4));
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_6 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (((mma_tma_stage + 3 >= 4) ? mma_tma_stage + 3 - 4 : mma_tma_stage + 3)) * 3072;
                        int _smem_desc_lo_7 = ((((smem_b0_addr) >> 4) & 0x7FFF) | 0x2000000) + (((mma_tma_stage + 3 >= 4) ? mma_tma_stage + 3 - 4 : mma_tma_stage + 3)) * 3072;
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
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_accum), "r"(1) : "memory");
                    }
                    mbarrier_wait(epilogue_done_addr, _phase_epi_free0);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (0 < k_iters_m) {
                        int _smem_desc_lo_8 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_9 = ((((smem_b1_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"((tmem_accum + (256))), "r"(0) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    if (1 < k_iters_m) {
                        int _smem_desc_lo_10 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_11 = ((((smem_b1_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"((tmem_accum + (256))), "r"(1) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    if (2 < k_iters_m) {
                        int _smem_desc_lo_12 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_13 = ((((smem_b1_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    :: "r"(_smem_desc_lo_12), "r"(_smem_desc_lo_13), "r"((tmem_accum + (256))), "r"(1) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    if (3 < k_iters_m) {
                        int _smem_desc_lo_14 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_15 = ((((smem_b1_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    :: "r"(_smem_desc_lo_14), "r"(_smem_desc_lo_15), "r"((tmem_accum + (256))), "r"(1) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    #pragma unroll 1
                    for (int iter_k_m = 4; iter_k_m < k_iters_m; iter_k_m++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_16 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_17 = ((((smem_b0_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::fill [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_16), "r"(_smem_desc_lo_17), "r"(tmem_accum), "r"(1) : "memory");
                        int _smem_desc_lo_18 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_19 = ((((smem_b1_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::lastuse [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_18), "r"(_smem_desc_lo_19), "r"((tmem_accum + (256))), "r"(1) : "memory");
                        int _smem_desc_lo_20 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_21 = ((((smem_b0_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    "add.u32 alo, %0, 128;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::fill [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_20), "r"(_smem_desc_lo_21), "r"(tmem_accum), "r"(1) : "memory");
                        int _smem_desc_lo_22 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_23 = ((((smem_b1_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    "add.u32 alo, %0, 128;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::lastuse [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_22), "r"(_smem_desc_lo_23), "r"((tmem_accum + (256))), "r"(1) : "memory");
                        int _smem_desc_lo_24 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_25 = ((((smem_b0_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    "add.u32 alo, %0, 256;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::fill [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_24), "r"(_smem_desc_lo_25), "r"(tmem_accum), "r"(1) : "memory");
                        int _smem_desc_lo_26 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_27 = ((((smem_b1_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    "add.u32 alo, %0, 256;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::lastuse [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_26), "r"(_smem_desc_lo_27), "r"((tmem_accum + (256))), "r"(1) : "memory");
                        int _smem_desc_lo_28 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_29 = ((((smem_b0_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    "add.u32 alo, %0, 384;\n\t"
                    "add.u32 blo, %1, 384;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::fill [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_28), "r"(_smem_desc_lo_29), "r"(tmem_accum), "r"(1) : "memory");
                        int _smem_desc_lo_30 = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _smem_desc_lo_31 = ((((smem_b1_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    "add.u32 alo, %0, 384;\n\t"
                    "add.u32 blo, %1, 384;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16.collector::a::lastuse [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_30), "r"(_smem_desc_lo_31), "r"((tmem_accum + (256))), "r"(1) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                    _phase_epi_free0 ^= 1;
                }
                unsigned int total_mh = total_tiles_m * 2;
                unsigned int tail_base_mh = total_mh / num_clusters_m * num_clusters_m;
                unsigned int num_tail_mh = total_mh - tail_base_mh;
                unsigned int whole_end_mh = total_mh;
                unsigned int units_mh = 0;
                if (tail_splits > 1) {
                    whole_end_mh = tail_base_mh;
                    units_mh = num_tail_mh * (unsigned int)tail_splits;
                }
                unsigned int num_whole_mh = whole_end_mh - tail_base_m * 2;
                #pragma unroll 1
                for (unsigned int v_m = cluster_m; v_m < num_whole_mh + units_mh; v_m += num_clusters_m) {
                    unsigned int hs_m = tail_base_m * 2 + v_m;
                    int s_mh = 0;
                    int splits_mh = 1;
                    if (v_m >= num_whole_mh) {
                        unsigned int u_mh = v_m - num_whole_mh;
                        unsigned int s_u_mh = u_mh / num_tail_mh;
                        hs_m = tail_base_mh + (u_mh - s_u_mh * num_tail_mh);
                        s_mh = (int)s_u_mh;
                        splits_mh = tail_splits;
                    }
                    int e_imh = (int)(hs_m / 2 / tiles_per_group_m);
                    int end_mh = offs[e_imh];
                    int prev_mh = e_imh - 1;
                    if (prev_mh < 0) {
                        prev_mh = 0;
                    }
                    int start_mh = offs[prev_mh];
                    if (e_imh == 0) {
                        start_mh = 0;
                    }
                    int k_iters_mh = (end_mh - start_mh + (BLOCK_K - 1)) / BLOCK_K;
                    if (k_iters_mh == 0) {
                        k_iters_mh = 1;
                    }
                    int lo_mh = s_mh * k_iters_mh / splits_mh;
                    int hi_mh = (s_mh + 1) * k_iters_mh / splits_mh;
                    int n_steps_mh = hi_mh - lo_mh;
                    if (n_steps_mh > 0) {
                        mbarrier_wait(epi_free0_addr, _phase_epi_free0);
                        #pragma unroll 1
                        for (int iter_k_mh = 0; iter_k_mh < n_steps_mh; iter_k_mh++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flagh = ((iter_k_mh == 0) ? 1 : 0);
                            int _mma_a_lo_h = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                            int _mma_b_lo_h = ((((smem_b0_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_tma_stage) * 3072;
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
                    :: "r"(_mma_a_lo_h), "r"(_mma_b_lo_h), "r"(tmem_accum), "r"(((init_flagh) ? 0 : 1)) : "memory");
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        mbarrier_wait(epilogue_done_addr, _phase_epi_free0);
                        elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                        _phase_epi_free0 ^= 1;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            const int epi_half = (warp - 2) / 4;
            unsigned int grid_n_e = (unsigned int)(N / BLOCK_N);
            unsigned int grid_k_e = (unsigned int)(K / TILE_K);
            unsigned int tiles_per_group_e = grid_n_e * grid_k_e;
            unsigned int total_tiles_e = (unsigned int)num_groups * tiles_per_group_e;
            unsigned int num_clusters_e = (unsigned int)num_clusters;
            unsigned int cluster_e = (unsigned int)cluster_id;
            unsigned int reg_end_e = total_tiles_e;
            unsigned int tail_base_e = total_tiles_e / num_clusters_e * num_clusters_e;
            unsigned int full_waves_e = tail_base_e / num_clusters_e;
            reg_end_e = tail_base_e;
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
                unsigned int rr_n_block_e = 0;
                unsigned int nc_n_block_e = rem_e / grid_k_e;
                unsigned int kr_n_block_e = rem_e / grid_n_e;
                n_block_e = nc_n_block_e * (1 - rr_n_block_e) + (rem_e - kr_n_block_e * grid_n_e) * rr_n_block_e;
                k_block_e = (rem_e - nc_n_block_e * grid_k_e) * (1 - rr_n_block_e) + kr_n_block_e * rr_n_block_e;
                int row_n = (int)n_block_e * BLOCK_N + cta_rank * BLOCK_M + epi_tid;
                int col0 = (int)k_block_e * TILE_K;
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _er_f[128];
                int er_rowf = taddr + (unsigned int)(cta_rank * 128 + epi_warp * 32 << 16) + (unsigned int)(epi_half * 128);
                int er_outf = (int)e_ue * stride_e + row_n * ldc + col0 + epi_half * 128;
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_er_f[0]), "=f"(_er_f[1]), "=f"(_er_f[2]), "=f"(_er_f[3]), "=f"(_er_f[4]), "=f"(_er_f[5]), "=f"(_er_f[6]), "=f"(_er_f[7]), "=f"(_er_f[8]), "=f"(_er_f[9]), "=f"(_er_f[10]), "=f"(_er_f[11]), "=f"(_er_f[12]), "=f"(_er_f[13]), "=f"(_er_f[14]), "=f"(_er_f[15]), "=f"(_er_f[16]), "=f"(_er_f[17]), "=f"(_er_f[18]), "=f"(_er_f[19]), "=f"(_er_f[20]), "=f"(_er_f[21]), "=f"(_er_f[22]), "=f"(_er_f[23]), "=f"(_er_f[24]), "=f"(_er_f[25]), "=f"(_er_f[26]), "=f"(_er_f[27]), "=f"(_er_f[28]), "=f"(_er_f[29]), "=f"(_er_f[30]), "=f"(_er_f[31])
                    : "r"(er_rowf));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_er_f[32]), "=f"(_er_f[33]), "=f"(_er_f[34]), "=f"(_er_f[35]), "=f"(_er_f[36]), "=f"(_er_f[37]), "=f"(_er_f[38]), "=f"(_er_f[39]), "=f"(_er_f[40]), "=f"(_er_f[41]), "=f"(_er_f[42]), "=f"(_er_f[43]), "=f"(_er_f[44]), "=f"(_er_f[45]), "=f"(_er_f[46]), "=f"(_er_f[47]), "=f"(_er_f[48]), "=f"(_er_f[49]), "=f"(_er_f[50]), "=f"(_er_f[51]), "=f"(_er_f[52]), "=f"(_er_f[53]), "=f"(_er_f[54]), "=f"(_er_f[55]), "=f"(_er_f[56]), "=f"(_er_f[57]), "=f"(_er_f[58]), "=f"(_er_f[59]), "=f"(_er_f[60]), "=f"(_er_f[61]), "=f"(_er_f[62]), "=f"(_er_f[63])
                    : "r"(er_rowf + 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_er_f[64]), "=f"(_er_f[65]), "=f"(_er_f[66]), "=f"(_er_f[67]), "=f"(_er_f[68]), "=f"(_er_f[69]), "=f"(_er_f[70]), "=f"(_er_f[71]), "=f"(_er_f[72]), "=f"(_er_f[73]), "=f"(_er_f[74]), "=f"(_er_f[75]), "=f"(_er_f[76]), "=f"(_er_f[77]), "=f"(_er_f[78]), "=f"(_er_f[79]), "=f"(_er_f[80]), "=f"(_er_f[81]), "=f"(_er_f[82]), "=f"(_er_f[83]), "=f"(_er_f[84]), "=f"(_er_f[85]), "=f"(_er_f[86]), "=f"(_er_f[87]), "=f"(_er_f[88]), "=f"(_er_f[89]), "=f"(_er_f[90]), "=f"(_er_f[91]), "=f"(_er_f[92]), "=f"(_er_f[93]), "=f"(_er_f[94]), "=f"(_er_f[95])
                    : "r"(er_rowf + 64));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_er_f[96]), "=f"(_er_f[97]), "=f"(_er_f[98]), "=f"(_er_f[99]), "=f"(_er_f[100]), "=f"(_er_f[101]), "=f"(_er_f[102]), "=f"(_er_f[103]), "=f"(_er_f[104]), "=f"(_er_f[105]), "=f"(_er_f[106]), "=f"(_er_f[107]), "=f"(_er_f[108]), "=f"(_er_f[109]), "=f"(_er_f[110]), "=f"(_er_f[111]), "=f"(_er_f[112]), "=f"(_er_f[113]), "=f"(_er_f[114]), "=f"(_er_f[115]), "=f"(_er_f[116]), "=f"(_er_f[117]), "=f"(_er_f[118]), "=f"(_er_f[119]), "=f"(_er_f[120]), "=f"(_er_f[121]), "=f"(_er_f[122]), "=f"(_er_f[123]), "=f"(_er_f[124]), "=f"(_er_f[125]), "=f"(_er_f[126]), "=f"(_er_f[127])
                    : "r"(er_rowf + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epi_free0_addr) & 0xFEFFFFFF) : "memory");
                }
                int ac_live = 1;
                int er_at2 = er_outf + (lane & 7) * (4 - ldc);
                if (ac_live == 1) {
                    int ar_bit_0 = lane >> 2 & 1;
                    float ar_x0_0 = _er_f[0];
                    float ar_x1_0 = _er_f[16];
                    float ar_send_0 = ((ar_bit_0 == 1) ? ar_x0_0 : ar_x1_0);
                    float ar_got_0 = __shfl_xor_sync(0xFFFFFFFF, ar_send_0, 4);
                    _er_f[0] = ((ar_bit_0 == 1) ? ar_got_0 : ar_x0_0);
                    _er_f[16] = ((ar_bit_0 == 1) ? ar_x1_0 : ar_got_0);
                    float ar_x0_1 = _er_f[4];
                    float ar_x1_1 = _er_f[20];
                    float ar_send_1 = ((ar_bit_0 == 1) ? ar_x0_1 : ar_x1_1);
                    float ar_got_1 = __shfl_xor_sync(0xFFFFFFFF, ar_send_1, 4);
                    _er_f[4] = ((ar_bit_0 == 1) ? ar_got_1 : ar_x0_1);
                    _er_f[20] = ((ar_bit_0 == 1) ? ar_x1_1 : ar_got_1);
                    float ar_x0_2 = _er_f[8];
                    float ar_x1_2 = _er_f[24];
                    float ar_send_2 = ((ar_bit_0 == 1) ? ar_x0_2 : ar_x1_2);
                    float ar_got_2 = __shfl_xor_sync(0xFFFFFFFF, ar_send_2, 4);
                    _er_f[8] = ((ar_bit_0 == 1) ? ar_got_2 : ar_x0_2);
                    _er_f[24] = ((ar_bit_0 == 1) ? ar_x1_2 : ar_got_2);
                    float ar_x0_3 = _er_f[12];
                    float ar_x1_3 = _er_f[28];
                    float ar_send_3 = ((ar_bit_0 == 1) ? ar_x0_3 : ar_x1_3);
                    float ar_got_3 = __shfl_xor_sync(0xFFFFFFFF, ar_send_3, 4);
                    _er_f[12] = ((ar_bit_0 == 1) ? ar_got_3 : ar_x0_3);
                    _er_f[28] = ((ar_bit_0 == 1) ? ar_x1_3 : ar_got_3);
                    int ar_bit_1 = lane >> 1 & 1;
                    float ar_x0_4 = _er_f[0];
                    float ar_x1_4 = _er_f[8];
                    float ar_send_4 = ((ar_bit_1 == 1) ? ar_x0_4 : ar_x1_4);
                    float ar_got_4 = __shfl_xor_sync(0xFFFFFFFF, ar_send_4, 2);
                    _er_f[0] = ((ar_bit_1 == 1) ? ar_got_4 : ar_x0_4);
                    _er_f[8] = ((ar_bit_1 == 1) ? ar_x1_4 : ar_got_4);
                    float ar_x0_5 = _er_f[4];
                    float ar_x1_5 = _er_f[12];
                    float ar_send_5 = ((ar_bit_1 == 1) ? ar_x0_5 : ar_x1_5);
                    float ar_got_5 = __shfl_xor_sync(0xFFFFFFFF, ar_send_5, 2);
                    _er_f[4] = ((ar_bit_1 == 1) ? ar_got_5 : ar_x0_5);
                    _er_f[12] = ((ar_bit_1 == 1) ? ar_x1_5 : ar_got_5);
                    float ar_x0_6 = _er_f[16];
                    float ar_x1_6 = _er_f[24];
                    float ar_send_6 = ((ar_bit_1 == 1) ? ar_x0_6 : ar_x1_6);
                    float ar_got_6 = __shfl_xor_sync(0xFFFFFFFF, ar_send_6, 2);
                    _er_f[16] = ((ar_bit_1 == 1) ? ar_got_6 : ar_x0_6);
                    _er_f[24] = ((ar_bit_1 == 1) ? ar_x1_6 : ar_got_6);
                    float ar_x0_7 = _er_f[20];
                    float ar_x1_7 = _er_f[28];
                    float ar_send_7 = ((ar_bit_1 == 1) ? ar_x0_7 : ar_x1_7);
                    float ar_got_7 = __shfl_xor_sync(0xFFFFFFFF, ar_send_7, 2);
                    _er_f[20] = ((ar_bit_1 == 1) ? ar_got_7 : ar_x0_7);
                    _er_f[28] = ((ar_bit_1 == 1) ? ar_x1_7 : ar_got_7);
                    int ar_bit_2 = lane & 1;
                    float ar_x0_8 = _er_f[0];
                    float ar_x1_8 = _er_f[4];
                    float ar_send_8 = ((ar_bit_2 == 1) ? ar_x0_8 : ar_x1_8);
                    float ar_got_8 = __shfl_xor_sync(0xFFFFFFFF, ar_send_8, 1);
                    _er_f[0] = ((ar_bit_2 == 1) ? ar_got_8 : ar_x0_8);
                    _er_f[4] = ((ar_bit_2 == 1) ? ar_x1_8 : ar_got_8);
                    float ar_x0_9 = _er_f[8];
                    float ar_x1_9 = _er_f[12];
                    float ar_send_9 = ((ar_bit_2 == 1) ? ar_x0_9 : ar_x1_9);
                    float ar_got_9 = __shfl_xor_sync(0xFFFFFFFF, ar_send_9, 1);
                    _er_f[8] = ((ar_bit_2 == 1) ? ar_got_9 : ar_x0_9);
                    _er_f[12] = ((ar_bit_2 == 1) ? ar_x1_9 : ar_got_9);
                    float ar_x0_10 = _er_f[16];
                    float ar_x1_10 = _er_f[20];
                    float ar_send_10 = ((ar_bit_2 == 1) ? ar_x0_10 : ar_x1_10);
                    float ar_got_10 = __shfl_xor_sync(0xFFFFFFFF, ar_send_10, 1);
                    _er_f[16] = ((ar_bit_2 == 1) ? ar_got_10 : ar_x0_10);
                    _er_f[20] = ((ar_bit_2 == 1) ? ar_x1_10 : ar_got_10);
                    float ar_x0_11 = _er_f[24];
                    float ar_x1_11 = _er_f[28];
                    float ar_send_11 = ((ar_bit_2 == 1) ? ar_x0_11 : ar_x1_11);
                    float ar_got_11 = __shfl_xor_sync(0xFFFFFFFF, ar_send_11, 1);
                    _er_f[24] = ((ar_bit_2 == 1) ? ar_got_11 : ar_x0_11);
                    _er_f[28] = ((ar_bit_2 == 1) ? ar_x1_11 : ar_got_11);
                    int ar_bit_3 = lane >> 2 & 1;
                    float ar_x0_12 = _er_f[1];
                    float ar_x1_12 = _er_f[17];
                    float ar_send_12 = ((ar_bit_3 == 1) ? ar_x0_12 : ar_x1_12);
                    float ar_got_12 = __shfl_xor_sync(0xFFFFFFFF, ar_send_12, 4);
                    _er_f[1] = ((ar_bit_3 == 1) ? ar_got_12 : ar_x0_12);
                    _er_f[17] = ((ar_bit_3 == 1) ? ar_x1_12 : ar_got_12);
                    float ar_x0_13 = _er_f[5];
                    float ar_x1_13 = _er_f[21];
                    float ar_send_13 = ((ar_bit_3 == 1) ? ar_x0_13 : ar_x1_13);
                    float ar_got_13 = __shfl_xor_sync(0xFFFFFFFF, ar_send_13, 4);
                    _er_f[5] = ((ar_bit_3 == 1) ? ar_got_13 : ar_x0_13);
                    _er_f[21] = ((ar_bit_3 == 1) ? ar_x1_13 : ar_got_13);
                    float ar_x0_14 = _er_f[9];
                    float ar_x1_14 = _er_f[25];
                    float ar_send_14 = ((ar_bit_3 == 1) ? ar_x0_14 : ar_x1_14);
                    float ar_got_14 = __shfl_xor_sync(0xFFFFFFFF, ar_send_14, 4);
                    _er_f[9] = ((ar_bit_3 == 1) ? ar_got_14 : ar_x0_14);
                    _er_f[25] = ((ar_bit_3 == 1) ? ar_x1_14 : ar_got_14);
                    float ar_x0_15 = _er_f[13];
                    float ar_x1_15 = _er_f[29];
                    float ar_send_15 = ((ar_bit_3 == 1) ? ar_x0_15 : ar_x1_15);
                    float ar_got_15 = __shfl_xor_sync(0xFFFFFFFF, ar_send_15, 4);
                    _er_f[13] = ((ar_bit_3 == 1) ? ar_got_15 : ar_x0_15);
                    _er_f[29] = ((ar_bit_3 == 1) ? ar_x1_15 : ar_got_15);
                    int ar_bit_4 = lane >> 1 & 1;
                    float ar_x0_16 = _er_f[1];
                    float ar_x1_16 = _er_f[9];
                    float ar_send_16 = ((ar_bit_4 == 1) ? ar_x0_16 : ar_x1_16);
                    float ar_got_16 = __shfl_xor_sync(0xFFFFFFFF, ar_send_16, 2);
                    _er_f[1] = ((ar_bit_4 == 1) ? ar_got_16 : ar_x0_16);
                    _er_f[9] = ((ar_bit_4 == 1) ? ar_x1_16 : ar_got_16);
                    float ar_x0_17 = _er_f[5];
                    float ar_x1_17 = _er_f[13];
                    float ar_send_17 = ((ar_bit_4 == 1) ? ar_x0_17 : ar_x1_17);
                    float ar_got_17 = __shfl_xor_sync(0xFFFFFFFF, ar_send_17, 2);
                    _er_f[5] = ((ar_bit_4 == 1) ? ar_got_17 : ar_x0_17);
                    _er_f[13] = ((ar_bit_4 == 1) ? ar_x1_17 : ar_got_17);
                    float ar_x0_18 = _er_f[17];
                    float ar_x1_18 = _er_f[25];
                    float ar_send_18 = ((ar_bit_4 == 1) ? ar_x0_18 : ar_x1_18);
                    float ar_got_18 = __shfl_xor_sync(0xFFFFFFFF, ar_send_18, 2);
                    _er_f[17] = ((ar_bit_4 == 1) ? ar_got_18 : ar_x0_18);
                    _er_f[25] = ((ar_bit_4 == 1) ? ar_x1_18 : ar_got_18);
                    float ar_x0_19 = _er_f[21];
                    float ar_x1_19 = _er_f[29];
                    float ar_send_19 = ((ar_bit_4 == 1) ? ar_x0_19 : ar_x1_19);
                    float ar_got_19 = __shfl_xor_sync(0xFFFFFFFF, ar_send_19, 2);
                    _er_f[21] = ((ar_bit_4 == 1) ? ar_got_19 : ar_x0_19);
                    _er_f[29] = ((ar_bit_4 == 1) ? ar_x1_19 : ar_got_19);
                    int ar_bit_5 = lane & 1;
                    float ar_x0_20 = _er_f[1];
                    float ar_x1_20 = _er_f[5];
                    float ar_send_20 = ((ar_bit_5 == 1) ? ar_x0_20 : ar_x1_20);
                    float ar_got_20 = __shfl_xor_sync(0xFFFFFFFF, ar_send_20, 1);
                    _er_f[1] = ((ar_bit_5 == 1) ? ar_got_20 : ar_x0_20);
                    _er_f[5] = ((ar_bit_5 == 1) ? ar_x1_20 : ar_got_20);
                    float ar_x0_21 = _er_f[9];
                    float ar_x1_21 = _er_f[13];
                    float ar_send_21 = ((ar_bit_5 == 1) ? ar_x0_21 : ar_x1_21);
                    float ar_got_21 = __shfl_xor_sync(0xFFFFFFFF, ar_send_21, 1);
                    _er_f[9] = ((ar_bit_5 == 1) ? ar_got_21 : ar_x0_21);
                    _er_f[13] = ((ar_bit_5 == 1) ? ar_x1_21 : ar_got_21);
                    float ar_x0_22 = _er_f[17];
                    float ar_x1_22 = _er_f[21];
                    float ar_send_22 = ((ar_bit_5 == 1) ? ar_x0_22 : ar_x1_22);
                    float ar_got_22 = __shfl_xor_sync(0xFFFFFFFF, ar_send_22, 1);
                    _er_f[17] = ((ar_bit_5 == 1) ? ar_got_22 : ar_x0_22);
                    _er_f[21] = ((ar_bit_5 == 1) ? ar_x1_22 : ar_got_22);
                    float ar_x0_23 = _er_f[25];
                    float ar_x1_23 = _er_f[29];
                    float ar_send_23 = ((ar_bit_5 == 1) ? ar_x0_23 : ar_x1_23);
                    float ar_got_23 = __shfl_xor_sync(0xFFFFFFFF, ar_send_23, 1);
                    _er_f[25] = ((ar_bit_5 == 1) ? ar_got_23 : ar_x0_23);
                    _er_f[29] = ((ar_bit_5 == 1) ? ar_x1_23 : ar_got_23);
                    int ar_bit_6 = lane >> 2 & 1;
                    float ar_x0_24 = _er_f[2];
                    float ar_x1_24 = _er_f[18];
                    float ar_send_24 = ((ar_bit_6 == 1) ? ar_x0_24 : ar_x1_24);
                    float ar_got_24 = __shfl_xor_sync(0xFFFFFFFF, ar_send_24, 4);
                    _er_f[2] = ((ar_bit_6 == 1) ? ar_got_24 : ar_x0_24);
                    _er_f[18] = ((ar_bit_6 == 1) ? ar_x1_24 : ar_got_24);
                    float ar_x0_25 = _er_f[6];
                    float ar_x1_25 = _er_f[22];
                    float ar_send_25 = ((ar_bit_6 == 1) ? ar_x0_25 : ar_x1_25);
                    float ar_got_25 = __shfl_xor_sync(0xFFFFFFFF, ar_send_25, 4);
                    _er_f[6] = ((ar_bit_6 == 1) ? ar_got_25 : ar_x0_25);
                    _er_f[22] = ((ar_bit_6 == 1) ? ar_x1_25 : ar_got_25);
                    float ar_x0_26 = _er_f[10];
                    float ar_x1_26 = _er_f[26];
                    float ar_send_26 = ((ar_bit_6 == 1) ? ar_x0_26 : ar_x1_26);
                    float ar_got_26 = __shfl_xor_sync(0xFFFFFFFF, ar_send_26, 4);
                    _er_f[10] = ((ar_bit_6 == 1) ? ar_got_26 : ar_x0_26);
                    _er_f[26] = ((ar_bit_6 == 1) ? ar_x1_26 : ar_got_26);
                    float ar_x0_27 = _er_f[14];
                    float ar_x1_27 = _er_f[30];
                    float ar_send_27 = ((ar_bit_6 == 1) ? ar_x0_27 : ar_x1_27);
                    float ar_got_27 = __shfl_xor_sync(0xFFFFFFFF, ar_send_27, 4);
                    _er_f[14] = ((ar_bit_6 == 1) ? ar_got_27 : ar_x0_27);
                    _er_f[30] = ((ar_bit_6 == 1) ? ar_x1_27 : ar_got_27);
                    int ar_bit_7 = lane >> 1 & 1;
                    float ar_x0_28 = _er_f[2];
                    float ar_x1_28 = _er_f[10];
                    float ar_send_28 = ((ar_bit_7 == 1) ? ar_x0_28 : ar_x1_28);
                    float ar_got_28 = __shfl_xor_sync(0xFFFFFFFF, ar_send_28, 2);
                    _er_f[2] = ((ar_bit_7 == 1) ? ar_got_28 : ar_x0_28);
                    _er_f[10] = ((ar_bit_7 == 1) ? ar_x1_28 : ar_got_28);
                    float ar_x0_29 = _er_f[6];
                    float ar_x1_29 = _er_f[14];
                    float ar_send_29 = ((ar_bit_7 == 1) ? ar_x0_29 : ar_x1_29);
                    float ar_got_29 = __shfl_xor_sync(0xFFFFFFFF, ar_send_29, 2);
                    _er_f[6] = ((ar_bit_7 == 1) ? ar_got_29 : ar_x0_29);
                    _er_f[14] = ((ar_bit_7 == 1) ? ar_x1_29 : ar_got_29);
                    float ar_x0_30 = _er_f[18];
                    float ar_x1_30 = _er_f[26];
                    float ar_send_30 = ((ar_bit_7 == 1) ? ar_x0_30 : ar_x1_30);
                    float ar_got_30 = __shfl_xor_sync(0xFFFFFFFF, ar_send_30, 2);
                    _er_f[18] = ((ar_bit_7 == 1) ? ar_got_30 : ar_x0_30);
                    _er_f[26] = ((ar_bit_7 == 1) ? ar_x1_30 : ar_got_30);
                    float ar_x0_31 = _er_f[22];
                    float ar_x1_31 = _er_f[30];
                    float ar_send_31 = ((ar_bit_7 == 1) ? ar_x0_31 : ar_x1_31);
                    float ar_got_31 = __shfl_xor_sync(0xFFFFFFFF, ar_send_31, 2);
                    _er_f[22] = ((ar_bit_7 == 1) ? ar_got_31 : ar_x0_31);
                    _er_f[30] = ((ar_bit_7 == 1) ? ar_x1_31 : ar_got_31);
                    int ar_bit_8 = lane & 1;
                    float ar_x0_32 = _er_f[2];
                    float ar_x1_32 = _er_f[6];
                    float ar_send_32 = ((ar_bit_8 == 1) ? ar_x0_32 : ar_x1_32);
                    float ar_got_32 = __shfl_xor_sync(0xFFFFFFFF, ar_send_32, 1);
                    _er_f[2] = ((ar_bit_8 == 1) ? ar_got_32 : ar_x0_32);
                    _er_f[6] = ((ar_bit_8 == 1) ? ar_x1_32 : ar_got_32);
                    float ar_x0_33 = _er_f[10];
                    float ar_x1_33 = _er_f[14];
                    float ar_send_33 = ((ar_bit_8 == 1) ? ar_x0_33 : ar_x1_33);
                    float ar_got_33 = __shfl_xor_sync(0xFFFFFFFF, ar_send_33, 1);
                    _er_f[10] = ((ar_bit_8 == 1) ? ar_got_33 : ar_x0_33);
                    _er_f[14] = ((ar_bit_8 == 1) ? ar_x1_33 : ar_got_33);
                    float ar_x0_34 = _er_f[18];
                    float ar_x1_34 = _er_f[22];
                    float ar_send_34 = ((ar_bit_8 == 1) ? ar_x0_34 : ar_x1_34);
                    float ar_got_34 = __shfl_xor_sync(0xFFFFFFFF, ar_send_34, 1);
                    _er_f[18] = ((ar_bit_8 == 1) ? ar_got_34 : ar_x0_34);
                    _er_f[22] = ((ar_bit_8 == 1) ? ar_x1_34 : ar_got_34);
                    float ar_x0_35 = _er_f[26];
                    float ar_x1_35 = _er_f[30];
                    float ar_send_35 = ((ar_bit_8 == 1) ? ar_x0_35 : ar_x1_35);
                    float ar_got_35 = __shfl_xor_sync(0xFFFFFFFF, ar_send_35, 1);
                    _er_f[26] = ((ar_bit_8 == 1) ? ar_got_35 : ar_x0_35);
                    _er_f[30] = ((ar_bit_8 == 1) ? ar_x1_35 : ar_got_35);
                    int ar_bit_9 = lane >> 2 & 1;
                    float ar_x0_36 = _er_f[3];
                    float ar_x1_36 = _er_f[19];
                    float ar_send_36 = ((ar_bit_9 == 1) ? ar_x0_36 : ar_x1_36);
                    float ar_got_36 = __shfl_xor_sync(0xFFFFFFFF, ar_send_36, 4);
                    _er_f[3] = ((ar_bit_9 == 1) ? ar_got_36 : ar_x0_36);
                    _er_f[19] = ((ar_bit_9 == 1) ? ar_x1_36 : ar_got_36);
                    float ar_x0_37 = _er_f[7];
                    float ar_x1_37 = _er_f[23];
                    float ar_send_37 = ((ar_bit_9 == 1) ? ar_x0_37 : ar_x1_37);
                    float ar_got_37 = __shfl_xor_sync(0xFFFFFFFF, ar_send_37, 4);
                    _er_f[7] = ((ar_bit_9 == 1) ? ar_got_37 : ar_x0_37);
                    _er_f[23] = ((ar_bit_9 == 1) ? ar_x1_37 : ar_got_37);
                    float ar_x0_38 = _er_f[11];
                    float ar_x1_38 = _er_f[27];
                    float ar_send_38 = ((ar_bit_9 == 1) ? ar_x0_38 : ar_x1_38);
                    float ar_got_38 = __shfl_xor_sync(0xFFFFFFFF, ar_send_38, 4);
                    _er_f[11] = ((ar_bit_9 == 1) ? ar_got_38 : ar_x0_38);
                    _er_f[27] = ((ar_bit_9 == 1) ? ar_x1_38 : ar_got_38);
                    float ar_x0_39 = _er_f[15];
                    float ar_x1_39 = _er_f[31];
                    float ar_send_39 = ((ar_bit_9 == 1) ? ar_x0_39 : ar_x1_39);
                    float ar_got_39 = __shfl_xor_sync(0xFFFFFFFF, ar_send_39, 4);
                    _er_f[15] = ((ar_bit_9 == 1) ? ar_got_39 : ar_x0_39);
                    _er_f[31] = ((ar_bit_9 == 1) ? ar_x1_39 : ar_got_39);
                    int ar_bit_10 = lane >> 1 & 1;
                    float ar_x0_40 = _er_f[3];
                    float ar_x1_40 = _er_f[11];
                    float ar_send_40 = ((ar_bit_10 == 1) ? ar_x0_40 : ar_x1_40);
                    float ar_got_40 = __shfl_xor_sync(0xFFFFFFFF, ar_send_40, 2);
                    _er_f[3] = ((ar_bit_10 == 1) ? ar_got_40 : ar_x0_40);
                    _er_f[11] = ((ar_bit_10 == 1) ? ar_x1_40 : ar_got_40);
                    float ar_x0_41 = _er_f[7];
                    float ar_x1_41 = _er_f[15];
                    float ar_send_41 = ((ar_bit_10 == 1) ? ar_x0_41 : ar_x1_41);
                    float ar_got_41 = __shfl_xor_sync(0xFFFFFFFF, ar_send_41, 2);
                    _er_f[7] = ((ar_bit_10 == 1) ? ar_got_41 : ar_x0_41);
                    _er_f[15] = ((ar_bit_10 == 1) ? ar_x1_41 : ar_got_41);
                    float ar_x0_42 = _er_f[19];
                    float ar_x1_42 = _er_f[27];
                    float ar_send_42 = ((ar_bit_10 == 1) ? ar_x0_42 : ar_x1_42);
                    float ar_got_42 = __shfl_xor_sync(0xFFFFFFFF, ar_send_42, 2);
                    _er_f[19] = ((ar_bit_10 == 1) ? ar_got_42 : ar_x0_42);
                    _er_f[27] = ((ar_bit_10 == 1) ? ar_x1_42 : ar_got_42);
                    float ar_x0_43 = _er_f[23];
                    float ar_x1_43 = _er_f[31];
                    float ar_send_43 = ((ar_bit_10 == 1) ? ar_x0_43 : ar_x1_43);
                    float ar_got_43 = __shfl_xor_sync(0xFFFFFFFF, ar_send_43, 2);
                    _er_f[23] = ((ar_bit_10 == 1) ? ar_got_43 : ar_x0_43);
                    _er_f[31] = ((ar_bit_10 == 1) ? ar_x1_43 : ar_got_43);
                    int ar_bit_11 = lane & 1;
                    float ar_x0_44 = _er_f[3];
                    float ar_x1_44 = _er_f[7];
                    float ar_send_44 = ((ar_bit_11 == 1) ? ar_x0_44 : ar_x1_44);
                    float ar_got_44 = __shfl_xor_sync(0xFFFFFFFF, ar_send_44, 1);
                    _er_f[3] = ((ar_bit_11 == 1) ? ar_got_44 : ar_x0_44);
                    _er_f[7] = ((ar_bit_11 == 1) ? ar_x1_44 : ar_got_44);
                    float ar_x0_45 = _er_f[11];
                    float ar_x1_45 = _er_f[15];
                    float ar_send_45 = ((ar_bit_11 == 1) ? ar_x0_45 : ar_x1_45);
                    float ar_got_45 = __shfl_xor_sync(0xFFFFFFFF, ar_send_45, 1);
                    _er_f[11] = ((ar_bit_11 == 1) ? ar_got_45 : ar_x0_45);
                    _er_f[15] = ((ar_bit_11 == 1) ? ar_x1_45 : ar_got_45);
                    float ar_x0_46 = _er_f[19];
                    float ar_x1_46 = _er_f[23];
                    float ar_send_46 = ((ar_bit_11 == 1) ? ar_x0_46 : ar_x1_46);
                    float ar_got_46 = __shfl_xor_sync(0xFFFFFFFF, ar_send_46, 1);
                    _er_f[19] = ((ar_bit_11 == 1) ? ar_got_46 : ar_x0_46);
                    _er_f[23] = ((ar_bit_11 == 1) ? ar_x1_46 : ar_got_46);
                    float ar_x0_47 = _er_f[27];
                    float ar_x1_47 = _er_f[31];
                    float ar_send_47 = ((ar_bit_11 == 1) ? ar_x0_47 : ar_x1_47);
                    float ar_got_47 = __shfl_xor_sync(0xFFFFFFFF, ar_send_47, 1);
                    _er_f[27] = ((ar_bit_11 == 1) ? ar_got_47 : ar_x0_47);
                    _er_f[31] = ((ar_bit_11 == 1) ? ar_x1_47 : ar_got_47);
                    int ar_ri_0 = er_at2;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_0])), "f"(_er_f[0]), "f"(_er_f[1]), "f"(_er_f[2]), "f"(_er_f[3]) : "memory");
                    ar_ri_0 = ar_ri_0 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_0])), "f"(_er_f[4]), "f"(_er_f[5]), "f"(_er_f[6]), "f"(_er_f[7]) : "memory");
                    ar_ri_0 = ar_ri_0 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_0])), "f"(_er_f[8]), "f"(_er_f[9]), "f"(_er_f[10]), "f"(_er_f[11]) : "memory");
                    ar_ri_0 = ar_ri_0 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_0])), "f"(_er_f[12]), "f"(_er_f[13]), "f"(_er_f[14]), "f"(_er_f[15]) : "memory");
                    ar_ri_0 = ar_ri_0 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_0])), "f"(_er_f[16]), "f"(_er_f[17]), "f"(_er_f[18]), "f"(_er_f[19]) : "memory");
                    ar_ri_0 = ar_ri_0 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_0])), "f"(_er_f[20]), "f"(_er_f[21]), "f"(_er_f[22]), "f"(_er_f[23]) : "memory");
                    ar_ri_0 = ar_ri_0 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_0])), "f"(_er_f[24]), "f"(_er_f[25]), "f"(_er_f[26]), "f"(_er_f[27]) : "memory");
                    ar_ri_0 = ar_ri_0 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_0])), "f"(_er_f[28]), "f"(_er_f[29]), "f"(_er_f[30]), "f"(_er_f[31]) : "memory");
                    int ar_bit_12 = lane >> 2 & 1;
                    float ar_x0_48 = _er_f[32];
                    float ar_x1_48 = _er_f[48];
                    float ar_send_48 = ((ar_bit_12 == 1) ? ar_x0_48 : ar_x1_48);
                    float ar_got_48 = __shfl_xor_sync(0xFFFFFFFF, ar_send_48, 4);
                    _er_f[32] = ((ar_bit_12 == 1) ? ar_got_48 : ar_x0_48);
                    _er_f[48] = ((ar_bit_12 == 1) ? ar_x1_48 : ar_got_48);
                    float ar_x0_49 = _er_f[36];
                    float ar_x1_49 = _er_f[52];
                    float ar_send_49 = ((ar_bit_12 == 1) ? ar_x0_49 : ar_x1_49);
                    float ar_got_49 = __shfl_xor_sync(0xFFFFFFFF, ar_send_49, 4);
                    _er_f[36] = ((ar_bit_12 == 1) ? ar_got_49 : ar_x0_49);
                    _er_f[52] = ((ar_bit_12 == 1) ? ar_x1_49 : ar_got_49);
                    float ar_x0_50 = _er_f[40];
                    float ar_x1_50 = _er_f[56];
                    float ar_send_50 = ((ar_bit_12 == 1) ? ar_x0_50 : ar_x1_50);
                    float ar_got_50 = __shfl_xor_sync(0xFFFFFFFF, ar_send_50, 4);
                    _er_f[40] = ((ar_bit_12 == 1) ? ar_got_50 : ar_x0_50);
                    _er_f[56] = ((ar_bit_12 == 1) ? ar_x1_50 : ar_got_50);
                    float ar_x0_51 = _er_f[44];
                    float ar_x1_51 = _er_f[60];
                    float ar_send_51 = ((ar_bit_12 == 1) ? ar_x0_51 : ar_x1_51);
                    float ar_got_51 = __shfl_xor_sync(0xFFFFFFFF, ar_send_51, 4);
                    _er_f[44] = ((ar_bit_12 == 1) ? ar_got_51 : ar_x0_51);
                    _er_f[60] = ((ar_bit_12 == 1) ? ar_x1_51 : ar_got_51);
                    int ar_bit_13 = lane >> 1 & 1;
                    float ar_x0_52 = _er_f[32];
                    float ar_x1_52 = _er_f[40];
                    float ar_send_52 = ((ar_bit_13 == 1) ? ar_x0_52 : ar_x1_52);
                    float ar_got_52 = __shfl_xor_sync(0xFFFFFFFF, ar_send_52, 2);
                    _er_f[32] = ((ar_bit_13 == 1) ? ar_got_52 : ar_x0_52);
                    _er_f[40] = ((ar_bit_13 == 1) ? ar_x1_52 : ar_got_52);
                    float ar_x0_53 = _er_f[36];
                    float ar_x1_53 = _er_f[44];
                    float ar_send_53 = ((ar_bit_13 == 1) ? ar_x0_53 : ar_x1_53);
                    float ar_got_53 = __shfl_xor_sync(0xFFFFFFFF, ar_send_53, 2);
                    _er_f[36] = ((ar_bit_13 == 1) ? ar_got_53 : ar_x0_53);
                    _er_f[44] = ((ar_bit_13 == 1) ? ar_x1_53 : ar_got_53);
                    float ar_x0_54 = _er_f[48];
                    float ar_x1_54 = _er_f[56];
                    float ar_send_54 = ((ar_bit_13 == 1) ? ar_x0_54 : ar_x1_54);
                    float ar_got_54 = __shfl_xor_sync(0xFFFFFFFF, ar_send_54, 2);
                    _er_f[48] = ((ar_bit_13 == 1) ? ar_got_54 : ar_x0_54);
                    _er_f[56] = ((ar_bit_13 == 1) ? ar_x1_54 : ar_got_54);
                    float ar_x0_55 = _er_f[52];
                    float ar_x1_55 = _er_f[60];
                    float ar_send_55 = ((ar_bit_13 == 1) ? ar_x0_55 : ar_x1_55);
                    float ar_got_55 = __shfl_xor_sync(0xFFFFFFFF, ar_send_55, 2);
                    _er_f[52] = ((ar_bit_13 == 1) ? ar_got_55 : ar_x0_55);
                    _er_f[60] = ((ar_bit_13 == 1) ? ar_x1_55 : ar_got_55);
                    int ar_bit_14 = lane & 1;
                    float ar_x0_56 = _er_f[32];
                    float ar_x1_56 = _er_f[36];
                    float ar_send_56 = ((ar_bit_14 == 1) ? ar_x0_56 : ar_x1_56);
                    float ar_got_56 = __shfl_xor_sync(0xFFFFFFFF, ar_send_56, 1);
                    _er_f[32] = ((ar_bit_14 == 1) ? ar_got_56 : ar_x0_56);
                    _er_f[36] = ((ar_bit_14 == 1) ? ar_x1_56 : ar_got_56);
                    float ar_x0_57 = _er_f[40];
                    float ar_x1_57 = _er_f[44];
                    float ar_send_57 = ((ar_bit_14 == 1) ? ar_x0_57 : ar_x1_57);
                    float ar_got_57 = __shfl_xor_sync(0xFFFFFFFF, ar_send_57, 1);
                    _er_f[40] = ((ar_bit_14 == 1) ? ar_got_57 : ar_x0_57);
                    _er_f[44] = ((ar_bit_14 == 1) ? ar_x1_57 : ar_got_57);
                    float ar_x0_58 = _er_f[48];
                    float ar_x1_58 = _er_f[52];
                    float ar_send_58 = ((ar_bit_14 == 1) ? ar_x0_58 : ar_x1_58);
                    float ar_got_58 = __shfl_xor_sync(0xFFFFFFFF, ar_send_58, 1);
                    _er_f[48] = ((ar_bit_14 == 1) ? ar_got_58 : ar_x0_58);
                    _er_f[52] = ((ar_bit_14 == 1) ? ar_x1_58 : ar_got_58);
                    float ar_x0_59 = _er_f[56];
                    float ar_x1_59 = _er_f[60];
                    float ar_send_59 = ((ar_bit_14 == 1) ? ar_x0_59 : ar_x1_59);
                    float ar_got_59 = __shfl_xor_sync(0xFFFFFFFF, ar_send_59, 1);
                    _er_f[56] = ((ar_bit_14 == 1) ? ar_got_59 : ar_x0_59);
                    _er_f[60] = ((ar_bit_14 == 1) ? ar_x1_59 : ar_got_59);
                    int ar_bit_15 = lane >> 2 & 1;
                    float ar_x0_60 = _er_f[33];
                    float ar_x1_60 = _er_f[49];
                    float ar_send_60 = ((ar_bit_15 == 1) ? ar_x0_60 : ar_x1_60);
                    float ar_got_60 = __shfl_xor_sync(0xFFFFFFFF, ar_send_60, 4);
                    _er_f[33] = ((ar_bit_15 == 1) ? ar_got_60 : ar_x0_60);
                    _er_f[49] = ((ar_bit_15 == 1) ? ar_x1_60 : ar_got_60);
                    float ar_x0_61 = _er_f[37];
                    float ar_x1_61 = _er_f[53];
                    float ar_send_61 = ((ar_bit_15 == 1) ? ar_x0_61 : ar_x1_61);
                    float ar_got_61 = __shfl_xor_sync(0xFFFFFFFF, ar_send_61, 4);
                    _er_f[37] = ((ar_bit_15 == 1) ? ar_got_61 : ar_x0_61);
                    _er_f[53] = ((ar_bit_15 == 1) ? ar_x1_61 : ar_got_61);
                    float ar_x0_62 = _er_f[41];
                    float ar_x1_62 = _er_f[57];
                    float ar_send_62 = ((ar_bit_15 == 1) ? ar_x0_62 : ar_x1_62);
                    float ar_got_62 = __shfl_xor_sync(0xFFFFFFFF, ar_send_62, 4);
                    _er_f[41] = ((ar_bit_15 == 1) ? ar_got_62 : ar_x0_62);
                    _er_f[57] = ((ar_bit_15 == 1) ? ar_x1_62 : ar_got_62);
                    float ar_x0_63 = _er_f[45];
                    float ar_x1_63 = _er_f[61];
                    float ar_send_63 = ((ar_bit_15 == 1) ? ar_x0_63 : ar_x1_63);
                    float ar_got_63 = __shfl_xor_sync(0xFFFFFFFF, ar_send_63, 4);
                    _er_f[45] = ((ar_bit_15 == 1) ? ar_got_63 : ar_x0_63);
                    _er_f[61] = ((ar_bit_15 == 1) ? ar_x1_63 : ar_got_63);
                    int ar_bit_16 = lane >> 1 & 1;
                    float ar_x0_64 = _er_f[33];
                    float ar_x1_64 = _er_f[41];
                    float ar_send_64 = ((ar_bit_16 == 1) ? ar_x0_64 : ar_x1_64);
                    float ar_got_64 = __shfl_xor_sync(0xFFFFFFFF, ar_send_64, 2);
                    _er_f[33] = ((ar_bit_16 == 1) ? ar_got_64 : ar_x0_64);
                    _er_f[41] = ((ar_bit_16 == 1) ? ar_x1_64 : ar_got_64);
                    float ar_x0_65 = _er_f[37];
                    float ar_x1_65 = _er_f[45];
                    float ar_send_65 = ((ar_bit_16 == 1) ? ar_x0_65 : ar_x1_65);
                    float ar_got_65 = __shfl_xor_sync(0xFFFFFFFF, ar_send_65, 2);
                    _er_f[37] = ((ar_bit_16 == 1) ? ar_got_65 : ar_x0_65);
                    _er_f[45] = ((ar_bit_16 == 1) ? ar_x1_65 : ar_got_65);
                    float ar_x0_66 = _er_f[49];
                    float ar_x1_66 = _er_f[57];
                    float ar_send_66 = ((ar_bit_16 == 1) ? ar_x0_66 : ar_x1_66);
                    float ar_got_66 = __shfl_xor_sync(0xFFFFFFFF, ar_send_66, 2);
                    _er_f[49] = ((ar_bit_16 == 1) ? ar_got_66 : ar_x0_66);
                    _er_f[57] = ((ar_bit_16 == 1) ? ar_x1_66 : ar_got_66);
                    float ar_x0_67 = _er_f[53];
                    float ar_x1_67 = _er_f[61];
                    float ar_send_67 = ((ar_bit_16 == 1) ? ar_x0_67 : ar_x1_67);
                    float ar_got_67 = __shfl_xor_sync(0xFFFFFFFF, ar_send_67, 2);
                    _er_f[53] = ((ar_bit_16 == 1) ? ar_got_67 : ar_x0_67);
                    _er_f[61] = ((ar_bit_16 == 1) ? ar_x1_67 : ar_got_67);
                    int ar_bit_17 = lane & 1;
                    float ar_x0_68 = _er_f[33];
                    float ar_x1_68 = _er_f[37];
                    float ar_send_68 = ((ar_bit_17 == 1) ? ar_x0_68 : ar_x1_68);
                    float ar_got_68 = __shfl_xor_sync(0xFFFFFFFF, ar_send_68, 1);
                    _er_f[33] = ((ar_bit_17 == 1) ? ar_got_68 : ar_x0_68);
                    _er_f[37] = ((ar_bit_17 == 1) ? ar_x1_68 : ar_got_68);
                    float ar_x0_69 = _er_f[41];
                    float ar_x1_69 = _er_f[45];
                    float ar_send_69 = ((ar_bit_17 == 1) ? ar_x0_69 : ar_x1_69);
                    float ar_got_69 = __shfl_xor_sync(0xFFFFFFFF, ar_send_69, 1);
                    _er_f[41] = ((ar_bit_17 == 1) ? ar_got_69 : ar_x0_69);
                    _er_f[45] = ((ar_bit_17 == 1) ? ar_x1_69 : ar_got_69);
                    float ar_x0_70 = _er_f[49];
                    float ar_x1_70 = _er_f[53];
                    float ar_send_70 = ((ar_bit_17 == 1) ? ar_x0_70 : ar_x1_70);
                    float ar_got_70 = __shfl_xor_sync(0xFFFFFFFF, ar_send_70, 1);
                    _er_f[49] = ((ar_bit_17 == 1) ? ar_got_70 : ar_x0_70);
                    _er_f[53] = ((ar_bit_17 == 1) ? ar_x1_70 : ar_got_70);
                    float ar_x0_71 = _er_f[57];
                    float ar_x1_71 = _er_f[61];
                    float ar_send_71 = ((ar_bit_17 == 1) ? ar_x0_71 : ar_x1_71);
                    float ar_got_71 = __shfl_xor_sync(0xFFFFFFFF, ar_send_71, 1);
                    _er_f[57] = ((ar_bit_17 == 1) ? ar_got_71 : ar_x0_71);
                    _er_f[61] = ((ar_bit_17 == 1) ? ar_x1_71 : ar_got_71);
                    int ar_bit_18 = lane >> 2 & 1;
                    float ar_x0_72 = _er_f[34];
                    float ar_x1_72 = _er_f[50];
                    float ar_send_72 = ((ar_bit_18 == 1) ? ar_x0_72 : ar_x1_72);
                    float ar_got_72 = __shfl_xor_sync(0xFFFFFFFF, ar_send_72, 4);
                    _er_f[34] = ((ar_bit_18 == 1) ? ar_got_72 : ar_x0_72);
                    _er_f[50] = ((ar_bit_18 == 1) ? ar_x1_72 : ar_got_72);
                    float ar_x0_73 = _er_f[38];
                    float ar_x1_73 = _er_f[54];
                    float ar_send_73 = ((ar_bit_18 == 1) ? ar_x0_73 : ar_x1_73);
                    float ar_got_73 = __shfl_xor_sync(0xFFFFFFFF, ar_send_73, 4);
                    _er_f[38] = ((ar_bit_18 == 1) ? ar_got_73 : ar_x0_73);
                    _er_f[54] = ((ar_bit_18 == 1) ? ar_x1_73 : ar_got_73);
                    float ar_x0_74 = _er_f[42];
                    float ar_x1_74 = _er_f[58];
                    float ar_send_74 = ((ar_bit_18 == 1) ? ar_x0_74 : ar_x1_74);
                    float ar_got_74 = __shfl_xor_sync(0xFFFFFFFF, ar_send_74, 4);
                    _er_f[42] = ((ar_bit_18 == 1) ? ar_got_74 : ar_x0_74);
                    _er_f[58] = ((ar_bit_18 == 1) ? ar_x1_74 : ar_got_74);
                    float ar_x0_75 = _er_f[46];
                    float ar_x1_75 = _er_f[62];
                    float ar_send_75 = ((ar_bit_18 == 1) ? ar_x0_75 : ar_x1_75);
                    float ar_got_75 = __shfl_xor_sync(0xFFFFFFFF, ar_send_75, 4);
                    _er_f[46] = ((ar_bit_18 == 1) ? ar_got_75 : ar_x0_75);
                    _er_f[62] = ((ar_bit_18 == 1) ? ar_x1_75 : ar_got_75);
                    int ar_bit_19 = lane >> 1 & 1;
                    float ar_x0_76 = _er_f[34];
                    float ar_x1_76 = _er_f[42];
                    float ar_send_76 = ((ar_bit_19 == 1) ? ar_x0_76 : ar_x1_76);
                    float ar_got_76 = __shfl_xor_sync(0xFFFFFFFF, ar_send_76, 2);
                    _er_f[34] = ((ar_bit_19 == 1) ? ar_got_76 : ar_x0_76);
                    _er_f[42] = ((ar_bit_19 == 1) ? ar_x1_76 : ar_got_76);
                    float ar_x0_77 = _er_f[38];
                    float ar_x1_77 = _er_f[46];
                    float ar_send_77 = ((ar_bit_19 == 1) ? ar_x0_77 : ar_x1_77);
                    float ar_got_77 = __shfl_xor_sync(0xFFFFFFFF, ar_send_77, 2);
                    _er_f[38] = ((ar_bit_19 == 1) ? ar_got_77 : ar_x0_77);
                    _er_f[46] = ((ar_bit_19 == 1) ? ar_x1_77 : ar_got_77);
                    float ar_x0_78 = _er_f[50];
                    float ar_x1_78 = _er_f[58];
                    float ar_send_78 = ((ar_bit_19 == 1) ? ar_x0_78 : ar_x1_78);
                    float ar_got_78 = __shfl_xor_sync(0xFFFFFFFF, ar_send_78, 2);
                    _er_f[50] = ((ar_bit_19 == 1) ? ar_got_78 : ar_x0_78);
                    _er_f[58] = ((ar_bit_19 == 1) ? ar_x1_78 : ar_got_78);
                    float ar_x0_79 = _er_f[54];
                    float ar_x1_79 = _er_f[62];
                    float ar_send_79 = ((ar_bit_19 == 1) ? ar_x0_79 : ar_x1_79);
                    float ar_got_79 = __shfl_xor_sync(0xFFFFFFFF, ar_send_79, 2);
                    _er_f[54] = ((ar_bit_19 == 1) ? ar_got_79 : ar_x0_79);
                    _er_f[62] = ((ar_bit_19 == 1) ? ar_x1_79 : ar_got_79);
                    int ar_bit_20 = lane & 1;
                    float ar_x0_80 = _er_f[34];
                    float ar_x1_80 = _er_f[38];
                    float ar_send_80 = ((ar_bit_20 == 1) ? ar_x0_80 : ar_x1_80);
                    float ar_got_80 = __shfl_xor_sync(0xFFFFFFFF, ar_send_80, 1);
                    _er_f[34] = ((ar_bit_20 == 1) ? ar_got_80 : ar_x0_80);
                    _er_f[38] = ((ar_bit_20 == 1) ? ar_x1_80 : ar_got_80);
                    float ar_x0_81 = _er_f[42];
                    float ar_x1_81 = _er_f[46];
                    float ar_send_81 = ((ar_bit_20 == 1) ? ar_x0_81 : ar_x1_81);
                    float ar_got_81 = __shfl_xor_sync(0xFFFFFFFF, ar_send_81, 1);
                    _er_f[42] = ((ar_bit_20 == 1) ? ar_got_81 : ar_x0_81);
                    _er_f[46] = ((ar_bit_20 == 1) ? ar_x1_81 : ar_got_81);
                    float ar_x0_82 = _er_f[50];
                    float ar_x1_82 = _er_f[54];
                    float ar_send_82 = ((ar_bit_20 == 1) ? ar_x0_82 : ar_x1_82);
                    float ar_got_82 = __shfl_xor_sync(0xFFFFFFFF, ar_send_82, 1);
                    _er_f[50] = ((ar_bit_20 == 1) ? ar_got_82 : ar_x0_82);
                    _er_f[54] = ((ar_bit_20 == 1) ? ar_x1_82 : ar_got_82);
                    float ar_x0_83 = _er_f[58];
                    float ar_x1_83 = _er_f[62];
                    float ar_send_83 = ((ar_bit_20 == 1) ? ar_x0_83 : ar_x1_83);
                    float ar_got_83 = __shfl_xor_sync(0xFFFFFFFF, ar_send_83, 1);
                    _er_f[58] = ((ar_bit_20 == 1) ? ar_got_83 : ar_x0_83);
                    _er_f[62] = ((ar_bit_20 == 1) ? ar_x1_83 : ar_got_83);
                    int ar_bit_21 = lane >> 2 & 1;
                    float ar_x0_84 = _er_f[35];
                    float ar_x1_84 = _er_f[51];
                    float ar_send_84 = ((ar_bit_21 == 1) ? ar_x0_84 : ar_x1_84);
                    float ar_got_84 = __shfl_xor_sync(0xFFFFFFFF, ar_send_84, 4);
                    _er_f[35] = ((ar_bit_21 == 1) ? ar_got_84 : ar_x0_84);
                    _er_f[51] = ((ar_bit_21 == 1) ? ar_x1_84 : ar_got_84);
                    float ar_x0_85 = _er_f[39];
                    float ar_x1_85 = _er_f[55];
                    float ar_send_85 = ((ar_bit_21 == 1) ? ar_x0_85 : ar_x1_85);
                    float ar_got_85 = __shfl_xor_sync(0xFFFFFFFF, ar_send_85, 4);
                    _er_f[39] = ((ar_bit_21 == 1) ? ar_got_85 : ar_x0_85);
                    _er_f[55] = ((ar_bit_21 == 1) ? ar_x1_85 : ar_got_85);
                    float ar_x0_86 = _er_f[43];
                    float ar_x1_86 = _er_f[59];
                    float ar_send_86 = ((ar_bit_21 == 1) ? ar_x0_86 : ar_x1_86);
                    float ar_got_86 = __shfl_xor_sync(0xFFFFFFFF, ar_send_86, 4);
                    _er_f[43] = ((ar_bit_21 == 1) ? ar_got_86 : ar_x0_86);
                    _er_f[59] = ((ar_bit_21 == 1) ? ar_x1_86 : ar_got_86);
                    float ar_x0_87 = _er_f[47];
                    float ar_x1_87 = _er_f[63];
                    float ar_send_87 = ((ar_bit_21 == 1) ? ar_x0_87 : ar_x1_87);
                    float ar_got_87 = __shfl_xor_sync(0xFFFFFFFF, ar_send_87, 4);
                    _er_f[47] = ((ar_bit_21 == 1) ? ar_got_87 : ar_x0_87);
                    _er_f[63] = ((ar_bit_21 == 1) ? ar_x1_87 : ar_got_87);
                    int ar_bit_22 = lane >> 1 & 1;
                    float ar_x0_88 = _er_f[35];
                    float ar_x1_88 = _er_f[43];
                    float ar_send_88 = ((ar_bit_22 == 1) ? ar_x0_88 : ar_x1_88);
                    float ar_got_88 = __shfl_xor_sync(0xFFFFFFFF, ar_send_88, 2);
                    _er_f[35] = ((ar_bit_22 == 1) ? ar_got_88 : ar_x0_88);
                    _er_f[43] = ((ar_bit_22 == 1) ? ar_x1_88 : ar_got_88);
                    float ar_x0_89 = _er_f[39];
                    float ar_x1_89 = _er_f[47];
                    float ar_send_89 = ((ar_bit_22 == 1) ? ar_x0_89 : ar_x1_89);
                    float ar_got_89 = __shfl_xor_sync(0xFFFFFFFF, ar_send_89, 2);
                    _er_f[39] = ((ar_bit_22 == 1) ? ar_got_89 : ar_x0_89);
                    _er_f[47] = ((ar_bit_22 == 1) ? ar_x1_89 : ar_got_89);
                    float ar_x0_90 = _er_f[51];
                    float ar_x1_90 = _er_f[59];
                    float ar_send_90 = ((ar_bit_22 == 1) ? ar_x0_90 : ar_x1_90);
                    float ar_got_90 = __shfl_xor_sync(0xFFFFFFFF, ar_send_90, 2);
                    _er_f[51] = ((ar_bit_22 == 1) ? ar_got_90 : ar_x0_90);
                    _er_f[59] = ((ar_bit_22 == 1) ? ar_x1_90 : ar_got_90);
                    float ar_x0_91 = _er_f[55];
                    float ar_x1_91 = _er_f[63];
                    float ar_send_91 = ((ar_bit_22 == 1) ? ar_x0_91 : ar_x1_91);
                    float ar_got_91 = __shfl_xor_sync(0xFFFFFFFF, ar_send_91, 2);
                    _er_f[55] = ((ar_bit_22 == 1) ? ar_got_91 : ar_x0_91);
                    _er_f[63] = ((ar_bit_22 == 1) ? ar_x1_91 : ar_got_91);
                    int ar_bit_23 = lane & 1;
                    float ar_x0_92 = _er_f[35];
                    float ar_x1_92 = _er_f[39];
                    float ar_send_92 = ((ar_bit_23 == 1) ? ar_x0_92 : ar_x1_92);
                    float ar_got_92 = __shfl_xor_sync(0xFFFFFFFF, ar_send_92, 1);
                    _er_f[35] = ((ar_bit_23 == 1) ? ar_got_92 : ar_x0_92);
                    _er_f[39] = ((ar_bit_23 == 1) ? ar_x1_92 : ar_got_92);
                    float ar_x0_93 = _er_f[43];
                    float ar_x1_93 = _er_f[47];
                    float ar_send_93 = ((ar_bit_23 == 1) ? ar_x0_93 : ar_x1_93);
                    float ar_got_93 = __shfl_xor_sync(0xFFFFFFFF, ar_send_93, 1);
                    _er_f[43] = ((ar_bit_23 == 1) ? ar_got_93 : ar_x0_93);
                    _er_f[47] = ((ar_bit_23 == 1) ? ar_x1_93 : ar_got_93);
                    float ar_x0_94 = _er_f[51];
                    float ar_x1_94 = _er_f[55];
                    float ar_send_94 = ((ar_bit_23 == 1) ? ar_x0_94 : ar_x1_94);
                    float ar_got_94 = __shfl_xor_sync(0xFFFFFFFF, ar_send_94, 1);
                    _er_f[51] = ((ar_bit_23 == 1) ? ar_got_94 : ar_x0_94);
                    _er_f[55] = ((ar_bit_23 == 1) ? ar_x1_94 : ar_got_94);
                    float ar_x0_95 = _er_f[59];
                    float ar_x1_95 = _er_f[63];
                    float ar_send_95 = ((ar_bit_23 == 1) ? ar_x0_95 : ar_x1_95);
                    float ar_got_95 = __shfl_xor_sync(0xFFFFFFFF, ar_send_95, 1);
                    _er_f[59] = ((ar_bit_23 == 1) ? ar_got_95 : ar_x0_95);
                    _er_f[63] = ((ar_bit_23 == 1) ? ar_x1_95 : ar_got_95);
                    int ar_ri_1 = er_at2 + 32;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_1])), "f"(_er_f[32]), "f"(_er_f[33]), "f"(_er_f[34]), "f"(_er_f[35]) : "memory");
                    ar_ri_1 = ar_ri_1 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_1])), "f"(_er_f[36]), "f"(_er_f[37]), "f"(_er_f[38]), "f"(_er_f[39]) : "memory");
                    ar_ri_1 = ar_ri_1 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_1])), "f"(_er_f[40]), "f"(_er_f[41]), "f"(_er_f[42]), "f"(_er_f[43]) : "memory");
                    ar_ri_1 = ar_ri_1 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_1])), "f"(_er_f[44]), "f"(_er_f[45]), "f"(_er_f[46]), "f"(_er_f[47]) : "memory");
                    ar_ri_1 = ar_ri_1 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_1])), "f"(_er_f[48]), "f"(_er_f[49]), "f"(_er_f[50]), "f"(_er_f[51]) : "memory");
                    ar_ri_1 = ar_ri_1 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_1])), "f"(_er_f[52]), "f"(_er_f[53]), "f"(_er_f[54]), "f"(_er_f[55]) : "memory");
                    ar_ri_1 = ar_ri_1 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_1])), "f"(_er_f[56]), "f"(_er_f[57]), "f"(_er_f[58]), "f"(_er_f[59]) : "memory");
                    ar_ri_1 = ar_ri_1 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_1])), "f"(_er_f[60]), "f"(_er_f[61]), "f"(_er_f[62]), "f"(_er_f[63]) : "memory");
                    int ar_bit_24 = lane >> 2 & 1;
                    float ar_x0_96 = _er_f[64];
                    float ar_x1_96 = _er_f[80];
                    float ar_send_96 = ((ar_bit_24 == 1) ? ar_x0_96 : ar_x1_96);
                    float ar_got_96 = __shfl_xor_sync(0xFFFFFFFF, ar_send_96, 4);
                    _er_f[64] = ((ar_bit_24 == 1) ? ar_got_96 : ar_x0_96);
                    _er_f[80] = ((ar_bit_24 == 1) ? ar_x1_96 : ar_got_96);
                    float ar_x0_97 = _er_f[68];
                    float ar_x1_97 = _er_f[84];
                    float ar_send_97 = ((ar_bit_24 == 1) ? ar_x0_97 : ar_x1_97);
                    float ar_got_97 = __shfl_xor_sync(0xFFFFFFFF, ar_send_97, 4);
                    _er_f[68] = ((ar_bit_24 == 1) ? ar_got_97 : ar_x0_97);
                    _er_f[84] = ((ar_bit_24 == 1) ? ar_x1_97 : ar_got_97);
                    float ar_x0_98 = _er_f[72];
                    float ar_x1_98 = _er_f[88];
                    float ar_send_98 = ((ar_bit_24 == 1) ? ar_x0_98 : ar_x1_98);
                    float ar_got_98 = __shfl_xor_sync(0xFFFFFFFF, ar_send_98, 4);
                    _er_f[72] = ((ar_bit_24 == 1) ? ar_got_98 : ar_x0_98);
                    _er_f[88] = ((ar_bit_24 == 1) ? ar_x1_98 : ar_got_98);
                    float ar_x0_99 = _er_f[76];
                    float ar_x1_99 = _er_f[92];
                    float ar_send_99 = ((ar_bit_24 == 1) ? ar_x0_99 : ar_x1_99);
                    float ar_got_99 = __shfl_xor_sync(0xFFFFFFFF, ar_send_99, 4);
                    _er_f[76] = ((ar_bit_24 == 1) ? ar_got_99 : ar_x0_99);
                    _er_f[92] = ((ar_bit_24 == 1) ? ar_x1_99 : ar_got_99);
                    int ar_bit_25 = lane >> 1 & 1;
                    float ar_x0_100 = _er_f[64];
                    float ar_x1_100 = _er_f[72];
                    float ar_send_100 = ((ar_bit_25 == 1) ? ar_x0_100 : ar_x1_100);
                    float ar_got_100 = __shfl_xor_sync(0xFFFFFFFF, ar_send_100, 2);
                    _er_f[64] = ((ar_bit_25 == 1) ? ar_got_100 : ar_x0_100);
                    _er_f[72] = ((ar_bit_25 == 1) ? ar_x1_100 : ar_got_100);
                    float ar_x0_101 = _er_f[68];
                    float ar_x1_101 = _er_f[76];
                    float ar_send_101 = ((ar_bit_25 == 1) ? ar_x0_101 : ar_x1_101);
                    float ar_got_101 = __shfl_xor_sync(0xFFFFFFFF, ar_send_101, 2);
                    _er_f[68] = ((ar_bit_25 == 1) ? ar_got_101 : ar_x0_101);
                    _er_f[76] = ((ar_bit_25 == 1) ? ar_x1_101 : ar_got_101);
                    float ar_x0_102 = _er_f[80];
                    float ar_x1_102 = _er_f[88];
                    float ar_send_102 = ((ar_bit_25 == 1) ? ar_x0_102 : ar_x1_102);
                    float ar_got_102 = __shfl_xor_sync(0xFFFFFFFF, ar_send_102, 2);
                    _er_f[80] = ((ar_bit_25 == 1) ? ar_got_102 : ar_x0_102);
                    _er_f[88] = ((ar_bit_25 == 1) ? ar_x1_102 : ar_got_102);
                    float ar_x0_103 = _er_f[84];
                    float ar_x1_103 = _er_f[92];
                    float ar_send_103 = ((ar_bit_25 == 1) ? ar_x0_103 : ar_x1_103);
                    float ar_got_103 = __shfl_xor_sync(0xFFFFFFFF, ar_send_103, 2);
                    _er_f[84] = ((ar_bit_25 == 1) ? ar_got_103 : ar_x0_103);
                    _er_f[92] = ((ar_bit_25 == 1) ? ar_x1_103 : ar_got_103);
                    int ar_bit_26 = lane & 1;
                    float ar_x0_104 = _er_f[64];
                    float ar_x1_104 = _er_f[68];
                    float ar_send_104 = ((ar_bit_26 == 1) ? ar_x0_104 : ar_x1_104);
                    float ar_got_104 = __shfl_xor_sync(0xFFFFFFFF, ar_send_104, 1);
                    _er_f[64] = ((ar_bit_26 == 1) ? ar_got_104 : ar_x0_104);
                    _er_f[68] = ((ar_bit_26 == 1) ? ar_x1_104 : ar_got_104);
                    float ar_x0_105 = _er_f[72];
                    float ar_x1_105 = _er_f[76];
                    float ar_send_105 = ((ar_bit_26 == 1) ? ar_x0_105 : ar_x1_105);
                    float ar_got_105 = __shfl_xor_sync(0xFFFFFFFF, ar_send_105, 1);
                    _er_f[72] = ((ar_bit_26 == 1) ? ar_got_105 : ar_x0_105);
                    _er_f[76] = ((ar_bit_26 == 1) ? ar_x1_105 : ar_got_105);
                    float ar_x0_106 = _er_f[80];
                    float ar_x1_106 = _er_f[84];
                    float ar_send_106 = ((ar_bit_26 == 1) ? ar_x0_106 : ar_x1_106);
                    float ar_got_106 = __shfl_xor_sync(0xFFFFFFFF, ar_send_106, 1);
                    _er_f[80] = ((ar_bit_26 == 1) ? ar_got_106 : ar_x0_106);
                    _er_f[84] = ((ar_bit_26 == 1) ? ar_x1_106 : ar_got_106);
                    float ar_x0_107 = _er_f[88];
                    float ar_x1_107 = _er_f[92];
                    float ar_send_107 = ((ar_bit_26 == 1) ? ar_x0_107 : ar_x1_107);
                    float ar_got_107 = __shfl_xor_sync(0xFFFFFFFF, ar_send_107, 1);
                    _er_f[88] = ((ar_bit_26 == 1) ? ar_got_107 : ar_x0_107);
                    _er_f[92] = ((ar_bit_26 == 1) ? ar_x1_107 : ar_got_107);
                    int ar_bit_27 = lane >> 2 & 1;
                    float ar_x0_108 = _er_f[65];
                    float ar_x1_108 = _er_f[81];
                    float ar_send_108 = ((ar_bit_27 == 1) ? ar_x0_108 : ar_x1_108);
                    float ar_got_108 = __shfl_xor_sync(0xFFFFFFFF, ar_send_108, 4);
                    _er_f[65] = ((ar_bit_27 == 1) ? ar_got_108 : ar_x0_108);
                    _er_f[81] = ((ar_bit_27 == 1) ? ar_x1_108 : ar_got_108);
                    float ar_x0_109 = _er_f[69];
                    float ar_x1_109 = _er_f[85];
                    float ar_send_109 = ((ar_bit_27 == 1) ? ar_x0_109 : ar_x1_109);
                    float ar_got_109 = __shfl_xor_sync(0xFFFFFFFF, ar_send_109, 4);
                    _er_f[69] = ((ar_bit_27 == 1) ? ar_got_109 : ar_x0_109);
                    _er_f[85] = ((ar_bit_27 == 1) ? ar_x1_109 : ar_got_109);
                    float ar_x0_110 = _er_f[73];
                    float ar_x1_110 = _er_f[89];
                    float ar_send_110 = ((ar_bit_27 == 1) ? ar_x0_110 : ar_x1_110);
                    float ar_got_110 = __shfl_xor_sync(0xFFFFFFFF, ar_send_110, 4);
                    _er_f[73] = ((ar_bit_27 == 1) ? ar_got_110 : ar_x0_110);
                    _er_f[89] = ((ar_bit_27 == 1) ? ar_x1_110 : ar_got_110);
                    float ar_x0_111 = _er_f[77];
                    float ar_x1_111 = _er_f[93];
                    float ar_send_111 = ((ar_bit_27 == 1) ? ar_x0_111 : ar_x1_111);
                    float ar_got_111 = __shfl_xor_sync(0xFFFFFFFF, ar_send_111, 4);
                    _er_f[77] = ((ar_bit_27 == 1) ? ar_got_111 : ar_x0_111);
                    _er_f[93] = ((ar_bit_27 == 1) ? ar_x1_111 : ar_got_111);
                    int ar_bit_28 = lane >> 1 & 1;
                    float ar_x0_112 = _er_f[65];
                    float ar_x1_112 = _er_f[73];
                    float ar_send_112 = ((ar_bit_28 == 1) ? ar_x0_112 : ar_x1_112);
                    float ar_got_112 = __shfl_xor_sync(0xFFFFFFFF, ar_send_112, 2);
                    _er_f[65] = ((ar_bit_28 == 1) ? ar_got_112 : ar_x0_112);
                    _er_f[73] = ((ar_bit_28 == 1) ? ar_x1_112 : ar_got_112);
                    float ar_x0_113 = _er_f[69];
                    float ar_x1_113 = _er_f[77];
                    float ar_send_113 = ((ar_bit_28 == 1) ? ar_x0_113 : ar_x1_113);
                    float ar_got_113 = __shfl_xor_sync(0xFFFFFFFF, ar_send_113, 2);
                    _er_f[69] = ((ar_bit_28 == 1) ? ar_got_113 : ar_x0_113);
                    _er_f[77] = ((ar_bit_28 == 1) ? ar_x1_113 : ar_got_113);
                    float ar_x0_114 = _er_f[81];
                    float ar_x1_114 = _er_f[89];
                    float ar_send_114 = ((ar_bit_28 == 1) ? ar_x0_114 : ar_x1_114);
                    float ar_got_114 = __shfl_xor_sync(0xFFFFFFFF, ar_send_114, 2);
                    _er_f[81] = ((ar_bit_28 == 1) ? ar_got_114 : ar_x0_114);
                    _er_f[89] = ((ar_bit_28 == 1) ? ar_x1_114 : ar_got_114);
                    float ar_x0_115 = _er_f[85];
                    float ar_x1_115 = _er_f[93];
                    float ar_send_115 = ((ar_bit_28 == 1) ? ar_x0_115 : ar_x1_115);
                    float ar_got_115 = __shfl_xor_sync(0xFFFFFFFF, ar_send_115, 2);
                    _er_f[85] = ((ar_bit_28 == 1) ? ar_got_115 : ar_x0_115);
                    _er_f[93] = ((ar_bit_28 == 1) ? ar_x1_115 : ar_got_115);
                    int ar_bit_29 = lane & 1;
                    float ar_x0_116 = _er_f[65];
                    float ar_x1_116 = _er_f[69];
                    float ar_send_116 = ((ar_bit_29 == 1) ? ar_x0_116 : ar_x1_116);
                    float ar_got_116 = __shfl_xor_sync(0xFFFFFFFF, ar_send_116, 1);
                    _er_f[65] = ((ar_bit_29 == 1) ? ar_got_116 : ar_x0_116);
                    _er_f[69] = ((ar_bit_29 == 1) ? ar_x1_116 : ar_got_116);
                    float ar_x0_117 = _er_f[73];
                    float ar_x1_117 = _er_f[77];
                    float ar_send_117 = ((ar_bit_29 == 1) ? ar_x0_117 : ar_x1_117);
                    float ar_got_117 = __shfl_xor_sync(0xFFFFFFFF, ar_send_117, 1);
                    _er_f[73] = ((ar_bit_29 == 1) ? ar_got_117 : ar_x0_117);
                    _er_f[77] = ((ar_bit_29 == 1) ? ar_x1_117 : ar_got_117);
                    float ar_x0_118 = _er_f[81];
                    float ar_x1_118 = _er_f[85];
                    float ar_send_118 = ((ar_bit_29 == 1) ? ar_x0_118 : ar_x1_118);
                    float ar_got_118 = __shfl_xor_sync(0xFFFFFFFF, ar_send_118, 1);
                    _er_f[81] = ((ar_bit_29 == 1) ? ar_got_118 : ar_x0_118);
                    _er_f[85] = ((ar_bit_29 == 1) ? ar_x1_118 : ar_got_118);
                    float ar_x0_119 = _er_f[89];
                    float ar_x1_119 = _er_f[93];
                    float ar_send_119 = ((ar_bit_29 == 1) ? ar_x0_119 : ar_x1_119);
                    float ar_got_119 = __shfl_xor_sync(0xFFFFFFFF, ar_send_119, 1);
                    _er_f[89] = ((ar_bit_29 == 1) ? ar_got_119 : ar_x0_119);
                    _er_f[93] = ((ar_bit_29 == 1) ? ar_x1_119 : ar_got_119);
                    int ar_bit_30 = lane >> 2 & 1;
                    float ar_x0_120 = _er_f[66];
                    float ar_x1_120 = _er_f[82];
                    float ar_send_120 = ((ar_bit_30 == 1) ? ar_x0_120 : ar_x1_120);
                    float ar_got_120 = __shfl_xor_sync(0xFFFFFFFF, ar_send_120, 4);
                    _er_f[66] = ((ar_bit_30 == 1) ? ar_got_120 : ar_x0_120);
                    _er_f[82] = ((ar_bit_30 == 1) ? ar_x1_120 : ar_got_120);
                    float ar_x0_121 = _er_f[70];
                    float ar_x1_121 = _er_f[86];
                    float ar_send_121 = ((ar_bit_30 == 1) ? ar_x0_121 : ar_x1_121);
                    float ar_got_121 = __shfl_xor_sync(0xFFFFFFFF, ar_send_121, 4);
                    _er_f[70] = ((ar_bit_30 == 1) ? ar_got_121 : ar_x0_121);
                    _er_f[86] = ((ar_bit_30 == 1) ? ar_x1_121 : ar_got_121);
                    float ar_x0_122 = _er_f[74];
                    float ar_x1_122 = _er_f[90];
                    float ar_send_122 = ((ar_bit_30 == 1) ? ar_x0_122 : ar_x1_122);
                    float ar_got_122 = __shfl_xor_sync(0xFFFFFFFF, ar_send_122, 4);
                    _er_f[74] = ((ar_bit_30 == 1) ? ar_got_122 : ar_x0_122);
                    _er_f[90] = ((ar_bit_30 == 1) ? ar_x1_122 : ar_got_122);
                    float ar_x0_123 = _er_f[78];
                    float ar_x1_123 = _er_f[94];
                    float ar_send_123 = ((ar_bit_30 == 1) ? ar_x0_123 : ar_x1_123);
                    float ar_got_123 = __shfl_xor_sync(0xFFFFFFFF, ar_send_123, 4);
                    _er_f[78] = ((ar_bit_30 == 1) ? ar_got_123 : ar_x0_123);
                    _er_f[94] = ((ar_bit_30 == 1) ? ar_x1_123 : ar_got_123);
                    int ar_bit_31 = lane >> 1 & 1;
                    float ar_x0_124 = _er_f[66];
                    float ar_x1_124 = _er_f[74];
                    float ar_send_124 = ((ar_bit_31 == 1) ? ar_x0_124 : ar_x1_124);
                    float ar_got_124 = __shfl_xor_sync(0xFFFFFFFF, ar_send_124, 2);
                    _er_f[66] = ((ar_bit_31 == 1) ? ar_got_124 : ar_x0_124);
                    _er_f[74] = ((ar_bit_31 == 1) ? ar_x1_124 : ar_got_124);
                    float ar_x0_125 = _er_f[70];
                    float ar_x1_125 = _er_f[78];
                    float ar_send_125 = ((ar_bit_31 == 1) ? ar_x0_125 : ar_x1_125);
                    float ar_got_125 = __shfl_xor_sync(0xFFFFFFFF, ar_send_125, 2);
                    _er_f[70] = ((ar_bit_31 == 1) ? ar_got_125 : ar_x0_125);
                    _er_f[78] = ((ar_bit_31 == 1) ? ar_x1_125 : ar_got_125);
                    float ar_x0_126 = _er_f[82];
                    float ar_x1_126 = _er_f[90];
                    float ar_send_126 = ((ar_bit_31 == 1) ? ar_x0_126 : ar_x1_126);
                    float ar_got_126 = __shfl_xor_sync(0xFFFFFFFF, ar_send_126, 2);
                    _er_f[82] = ((ar_bit_31 == 1) ? ar_got_126 : ar_x0_126);
                    _er_f[90] = ((ar_bit_31 == 1) ? ar_x1_126 : ar_got_126);
                    float ar_x0_127 = _er_f[86];
                    float ar_x1_127 = _er_f[94];
                    float ar_send_127 = ((ar_bit_31 == 1) ? ar_x0_127 : ar_x1_127);
                    float ar_got_127 = __shfl_xor_sync(0xFFFFFFFF, ar_send_127, 2);
                    _er_f[86] = ((ar_bit_31 == 1) ? ar_got_127 : ar_x0_127);
                    _er_f[94] = ((ar_bit_31 == 1) ? ar_x1_127 : ar_got_127);
                    int ar_bit_32 = lane & 1;
                    float ar_x0_128 = _er_f[66];
                    float ar_x1_128 = _er_f[70];
                    float ar_send_128 = ((ar_bit_32 == 1) ? ar_x0_128 : ar_x1_128);
                    float ar_got_128 = __shfl_xor_sync(0xFFFFFFFF, ar_send_128, 1);
                    _er_f[66] = ((ar_bit_32 == 1) ? ar_got_128 : ar_x0_128);
                    _er_f[70] = ((ar_bit_32 == 1) ? ar_x1_128 : ar_got_128);
                    float ar_x0_129 = _er_f[74];
                    float ar_x1_129 = _er_f[78];
                    float ar_send_129 = ((ar_bit_32 == 1) ? ar_x0_129 : ar_x1_129);
                    float ar_got_129 = __shfl_xor_sync(0xFFFFFFFF, ar_send_129, 1);
                    _er_f[74] = ((ar_bit_32 == 1) ? ar_got_129 : ar_x0_129);
                    _er_f[78] = ((ar_bit_32 == 1) ? ar_x1_129 : ar_got_129);
                    float ar_x0_130 = _er_f[82];
                    float ar_x1_130 = _er_f[86];
                    float ar_send_130 = ((ar_bit_32 == 1) ? ar_x0_130 : ar_x1_130);
                    float ar_got_130 = __shfl_xor_sync(0xFFFFFFFF, ar_send_130, 1);
                    _er_f[82] = ((ar_bit_32 == 1) ? ar_got_130 : ar_x0_130);
                    _er_f[86] = ((ar_bit_32 == 1) ? ar_x1_130 : ar_got_130);
                    float ar_x0_131 = _er_f[90];
                    float ar_x1_131 = _er_f[94];
                    float ar_send_131 = ((ar_bit_32 == 1) ? ar_x0_131 : ar_x1_131);
                    float ar_got_131 = __shfl_xor_sync(0xFFFFFFFF, ar_send_131, 1);
                    _er_f[90] = ((ar_bit_32 == 1) ? ar_got_131 : ar_x0_131);
                    _er_f[94] = ((ar_bit_32 == 1) ? ar_x1_131 : ar_got_131);
                    int ar_bit_33 = lane >> 2 & 1;
                    float ar_x0_132 = _er_f[67];
                    float ar_x1_132 = _er_f[83];
                    float ar_send_132 = ((ar_bit_33 == 1) ? ar_x0_132 : ar_x1_132);
                    float ar_got_132 = __shfl_xor_sync(0xFFFFFFFF, ar_send_132, 4);
                    _er_f[67] = ((ar_bit_33 == 1) ? ar_got_132 : ar_x0_132);
                    _er_f[83] = ((ar_bit_33 == 1) ? ar_x1_132 : ar_got_132);
                    float ar_x0_133 = _er_f[71];
                    float ar_x1_133 = _er_f[87];
                    float ar_send_133 = ((ar_bit_33 == 1) ? ar_x0_133 : ar_x1_133);
                    float ar_got_133 = __shfl_xor_sync(0xFFFFFFFF, ar_send_133, 4);
                    _er_f[71] = ((ar_bit_33 == 1) ? ar_got_133 : ar_x0_133);
                    _er_f[87] = ((ar_bit_33 == 1) ? ar_x1_133 : ar_got_133);
                    float ar_x0_134 = _er_f[75];
                    float ar_x1_134 = _er_f[91];
                    float ar_send_134 = ((ar_bit_33 == 1) ? ar_x0_134 : ar_x1_134);
                    float ar_got_134 = __shfl_xor_sync(0xFFFFFFFF, ar_send_134, 4);
                    _er_f[75] = ((ar_bit_33 == 1) ? ar_got_134 : ar_x0_134);
                    _er_f[91] = ((ar_bit_33 == 1) ? ar_x1_134 : ar_got_134);
                    float ar_x0_135 = _er_f[79];
                    float ar_x1_135 = _er_f[95];
                    float ar_send_135 = ((ar_bit_33 == 1) ? ar_x0_135 : ar_x1_135);
                    float ar_got_135 = __shfl_xor_sync(0xFFFFFFFF, ar_send_135, 4);
                    _er_f[79] = ((ar_bit_33 == 1) ? ar_got_135 : ar_x0_135);
                    _er_f[95] = ((ar_bit_33 == 1) ? ar_x1_135 : ar_got_135);
                    int ar_bit_34 = lane >> 1 & 1;
                    float ar_x0_136 = _er_f[67];
                    float ar_x1_136 = _er_f[75];
                    float ar_send_136 = ((ar_bit_34 == 1) ? ar_x0_136 : ar_x1_136);
                    float ar_got_136 = __shfl_xor_sync(0xFFFFFFFF, ar_send_136, 2);
                    _er_f[67] = ((ar_bit_34 == 1) ? ar_got_136 : ar_x0_136);
                    _er_f[75] = ((ar_bit_34 == 1) ? ar_x1_136 : ar_got_136);
                    float ar_x0_137 = _er_f[71];
                    float ar_x1_137 = _er_f[79];
                    float ar_send_137 = ((ar_bit_34 == 1) ? ar_x0_137 : ar_x1_137);
                    float ar_got_137 = __shfl_xor_sync(0xFFFFFFFF, ar_send_137, 2);
                    _er_f[71] = ((ar_bit_34 == 1) ? ar_got_137 : ar_x0_137);
                    _er_f[79] = ((ar_bit_34 == 1) ? ar_x1_137 : ar_got_137);
                    float ar_x0_138 = _er_f[83];
                    float ar_x1_138 = _er_f[91];
                    float ar_send_138 = ((ar_bit_34 == 1) ? ar_x0_138 : ar_x1_138);
                    float ar_got_138 = __shfl_xor_sync(0xFFFFFFFF, ar_send_138, 2);
                    _er_f[83] = ((ar_bit_34 == 1) ? ar_got_138 : ar_x0_138);
                    _er_f[91] = ((ar_bit_34 == 1) ? ar_x1_138 : ar_got_138);
                    float ar_x0_139 = _er_f[87];
                    float ar_x1_139 = _er_f[95];
                    float ar_send_139 = ((ar_bit_34 == 1) ? ar_x0_139 : ar_x1_139);
                    float ar_got_139 = __shfl_xor_sync(0xFFFFFFFF, ar_send_139, 2);
                    _er_f[87] = ((ar_bit_34 == 1) ? ar_got_139 : ar_x0_139);
                    _er_f[95] = ((ar_bit_34 == 1) ? ar_x1_139 : ar_got_139);
                    int ar_bit_35 = lane & 1;
                    float ar_x0_140 = _er_f[67];
                    float ar_x1_140 = _er_f[71];
                    float ar_send_140 = ((ar_bit_35 == 1) ? ar_x0_140 : ar_x1_140);
                    float ar_got_140 = __shfl_xor_sync(0xFFFFFFFF, ar_send_140, 1);
                    _er_f[67] = ((ar_bit_35 == 1) ? ar_got_140 : ar_x0_140);
                    _er_f[71] = ((ar_bit_35 == 1) ? ar_x1_140 : ar_got_140);
                    float ar_x0_141 = _er_f[75];
                    float ar_x1_141 = _er_f[79];
                    float ar_send_141 = ((ar_bit_35 == 1) ? ar_x0_141 : ar_x1_141);
                    float ar_got_141 = __shfl_xor_sync(0xFFFFFFFF, ar_send_141, 1);
                    _er_f[75] = ((ar_bit_35 == 1) ? ar_got_141 : ar_x0_141);
                    _er_f[79] = ((ar_bit_35 == 1) ? ar_x1_141 : ar_got_141);
                    float ar_x0_142 = _er_f[83];
                    float ar_x1_142 = _er_f[87];
                    float ar_send_142 = ((ar_bit_35 == 1) ? ar_x0_142 : ar_x1_142);
                    float ar_got_142 = __shfl_xor_sync(0xFFFFFFFF, ar_send_142, 1);
                    _er_f[83] = ((ar_bit_35 == 1) ? ar_got_142 : ar_x0_142);
                    _er_f[87] = ((ar_bit_35 == 1) ? ar_x1_142 : ar_got_142);
                    float ar_x0_143 = _er_f[91];
                    float ar_x1_143 = _er_f[95];
                    float ar_send_143 = ((ar_bit_35 == 1) ? ar_x0_143 : ar_x1_143);
                    float ar_got_143 = __shfl_xor_sync(0xFFFFFFFF, ar_send_143, 1);
                    _er_f[91] = ((ar_bit_35 == 1) ? ar_got_143 : ar_x0_143);
                    _er_f[95] = ((ar_bit_35 == 1) ? ar_x1_143 : ar_got_143);
                    int ar_ri_2 = er_at2 + 64;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_2])), "f"(_er_f[64]), "f"(_er_f[65]), "f"(_er_f[66]), "f"(_er_f[67]) : "memory");
                    ar_ri_2 = ar_ri_2 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_2])), "f"(_er_f[68]), "f"(_er_f[69]), "f"(_er_f[70]), "f"(_er_f[71]) : "memory");
                    ar_ri_2 = ar_ri_2 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_2])), "f"(_er_f[72]), "f"(_er_f[73]), "f"(_er_f[74]), "f"(_er_f[75]) : "memory");
                    ar_ri_2 = ar_ri_2 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_2])), "f"(_er_f[76]), "f"(_er_f[77]), "f"(_er_f[78]), "f"(_er_f[79]) : "memory");
                    ar_ri_2 = ar_ri_2 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_2])), "f"(_er_f[80]), "f"(_er_f[81]), "f"(_er_f[82]), "f"(_er_f[83]) : "memory");
                    ar_ri_2 = ar_ri_2 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_2])), "f"(_er_f[84]), "f"(_er_f[85]), "f"(_er_f[86]), "f"(_er_f[87]) : "memory");
                    ar_ri_2 = ar_ri_2 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_2])), "f"(_er_f[88]), "f"(_er_f[89]), "f"(_er_f[90]), "f"(_er_f[91]) : "memory");
                    ar_ri_2 = ar_ri_2 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_2])), "f"(_er_f[92]), "f"(_er_f[93]), "f"(_er_f[94]), "f"(_er_f[95]) : "memory");
                    int ar_bit_36 = lane >> 2 & 1;
                    float ar_x0_144 = _er_f[96];
                    float ar_x1_144 = _er_f[112];
                    float ar_send_144 = ((ar_bit_36 == 1) ? ar_x0_144 : ar_x1_144);
                    float ar_got_144 = __shfl_xor_sync(0xFFFFFFFF, ar_send_144, 4);
                    _er_f[96] = ((ar_bit_36 == 1) ? ar_got_144 : ar_x0_144);
                    _er_f[112] = ((ar_bit_36 == 1) ? ar_x1_144 : ar_got_144);
                    float ar_x0_145 = _er_f[100];
                    float ar_x1_145 = _er_f[116];
                    float ar_send_145 = ((ar_bit_36 == 1) ? ar_x0_145 : ar_x1_145);
                    float ar_got_145 = __shfl_xor_sync(0xFFFFFFFF, ar_send_145, 4);
                    _er_f[100] = ((ar_bit_36 == 1) ? ar_got_145 : ar_x0_145);
                    _er_f[116] = ((ar_bit_36 == 1) ? ar_x1_145 : ar_got_145);
                    float ar_x0_146 = _er_f[104];
                    float ar_x1_146 = _er_f[120];
                    float ar_send_146 = ((ar_bit_36 == 1) ? ar_x0_146 : ar_x1_146);
                    float ar_got_146 = __shfl_xor_sync(0xFFFFFFFF, ar_send_146, 4);
                    _er_f[104] = ((ar_bit_36 == 1) ? ar_got_146 : ar_x0_146);
                    _er_f[120] = ((ar_bit_36 == 1) ? ar_x1_146 : ar_got_146);
                    float ar_x0_147 = _er_f[108];
                    float ar_x1_147 = _er_f[124];
                    float ar_send_147 = ((ar_bit_36 == 1) ? ar_x0_147 : ar_x1_147);
                    float ar_got_147 = __shfl_xor_sync(0xFFFFFFFF, ar_send_147, 4);
                    _er_f[108] = ((ar_bit_36 == 1) ? ar_got_147 : ar_x0_147);
                    _er_f[124] = ((ar_bit_36 == 1) ? ar_x1_147 : ar_got_147);
                    int ar_bit_37 = lane >> 1 & 1;
                    float ar_x0_148 = _er_f[96];
                    float ar_x1_148 = _er_f[104];
                    float ar_send_148 = ((ar_bit_37 == 1) ? ar_x0_148 : ar_x1_148);
                    float ar_got_148 = __shfl_xor_sync(0xFFFFFFFF, ar_send_148, 2);
                    _er_f[96] = ((ar_bit_37 == 1) ? ar_got_148 : ar_x0_148);
                    _er_f[104] = ((ar_bit_37 == 1) ? ar_x1_148 : ar_got_148);
                    float ar_x0_149 = _er_f[100];
                    float ar_x1_149 = _er_f[108];
                    float ar_send_149 = ((ar_bit_37 == 1) ? ar_x0_149 : ar_x1_149);
                    float ar_got_149 = __shfl_xor_sync(0xFFFFFFFF, ar_send_149, 2);
                    _er_f[100] = ((ar_bit_37 == 1) ? ar_got_149 : ar_x0_149);
                    _er_f[108] = ((ar_bit_37 == 1) ? ar_x1_149 : ar_got_149);
                    float ar_x0_150 = _er_f[112];
                    float ar_x1_150 = _er_f[120];
                    float ar_send_150 = ((ar_bit_37 == 1) ? ar_x0_150 : ar_x1_150);
                    float ar_got_150 = __shfl_xor_sync(0xFFFFFFFF, ar_send_150, 2);
                    _er_f[112] = ((ar_bit_37 == 1) ? ar_got_150 : ar_x0_150);
                    _er_f[120] = ((ar_bit_37 == 1) ? ar_x1_150 : ar_got_150);
                    float ar_x0_151 = _er_f[116];
                    float ar_x1_151 = _er_f[124];
                    float ar_send_151 = ((ar_bit_37 == 1) ? ar_x0_151 : ar_x1_151);
                    float ar_got_151 = __shfl_xor_sync(0xFFFFFFFF, ar_send_151, 2);
                    _er_f[116] = ((ar_bit_37 == 1) ? ar_got_151 : ar_x0_151);
                    _er_f[124] = ((ar_bit_37 == 1) ? ar_x1_151 : ar_got_151);
                    int ar_bit_38 = lane & 1;
                    float ar_x0_152 = _er_f[96];
                    float ar_x1_152 = _er_f[100];
                    float ar_send_152 = ((ar_bit_38 == 1) ? ar_x0_152 : ar_x1_152);
                    float ar_got_152 = __shfl_xor_sync(0xFFFFFFFF, ar_send_152, 1);
                    _er_f[96] = ((ar_bit_38 == 1) ? ar_got_152 : ar_x0_152);
                    _er_f[100] = ((ar_bit_38 == 1) ? ar_x1_152 : ar_got_152);
                    float ar_x0_153 = _er_f[104];
                    float ar_x1_153 = _er_f[108];
                    float ar_send_153 = ((ar_bit_38 == 1) ? ar_x0_153 : ar_x1_153);
                    float ar_got_153 = __shfl_xor_sync(0xFFFFFFFF, ar_send_153, 1);
                    _er_f[104] = ((ar_bit_38 == 1) ? ar_got_153 : ar_x0_153);
                    _er_f[108] = ((ar_bit_38 == 1) ? ar_x1_153 : ar_got_153);
                    float ar_x0_154 = _er_f[112];
                    float ar_x1_154 = _er_f[116];
                    float ar_send_154 = ((ar_bit_38 == 1) ? ar_x0_154 : ar_x1_154);
                    float ar_got_154 = __shfl_xor_sync(0xFFFFFFFF, ar_send_154, 1);
                    _er_f[112] = ((ar_bit_38 == 1) ? ar_got_154 : ar_x0_154);
                    _er_f[116] = ((ar_bit_38 == 1) ? ar_x1_154 : ar_got_154);
                    float ar_x0_155 = _er_f[120];
                    float ar_x1_155 = _er_f[124];
                    float ar_send_155 = ((ar_bit_38 == 1) ? ar_x0_155 : ar_x1_155);
                    float ar_got_155 = __shfl_xor_sync(0xFFFFFFFF, ar_send_155, 1);
                    _er_f[120] = ((ar_bit_38 == 1) ? ar_got_155 : ar_x0_155);
                    _er_f[124] = ((ar_bit_38 == 1) ? ar_x1_155 : ar_got_155);
                    int ar_bit_39 = lane >> 2 & 1;
                    float ar_x0_156 = _er_f[97];
                    float ar_x1_156 = _er_f[113];
                    float ar_send_156 = ((ar_bit_39 == 1) ? ar_x0_156 : ar_x1_156);
                    float ar_got_156 = __shfl_xor_sync(0xFFFFFFFF, ar_send_156, 4);
                    _er_f[97] = ((ar_bit_39 == 1) ? ar_got_156 : ar_x0_156);
                    _er_f[113] = ((ar_bit_39 == 1) ? ar_x1_156 : ar_got_156);
                    float ar_x0_157 = _er_f[101];
                    float ar_x1_157 = _er_f[117];
                    float ar_send_157 = ((ar_bit_39 == 1) ? ar_x0_157 : ar_x1_157);
                    float ar_got_157 = __shfl_xor_sync(0xFFFFFFFF, ar_send_157, 4);
                    _er_f[101] = ((ar_bit_39 == 1) ? ar_got_157 : ar_x0_157);
                    _er_f[117] = ((ar_bit_39 == 1) ? ar_x1_157 : ar_got_157);
                    float ar_x0_158 = _er_f[105];
                    float ar_x1_158 = _er_f[121];
                    float ar_send_158 = ((ar_bit_39 == 1) ? ar_x0_158 : ar_x1_158);
                    float ar_got_158 = __shfl_xor_sync(0xFFFFFFFF, ar_send_158, 4);
                    _er_f[105] = ((ar_bit_39 == 1) ? ar_got_158 : ar_x0_158);
                    _er_f[121] = ((ar_bit_39 == 1) ? ar_x1_158 : ar_got_158);
                    float ar_x0_159 = _er_f[109];
                    float ar_x1_159 = _er_f[125];
                    float ar_send_159 = ((ar_bit_39 == 1) ? ar_x0_159 : ar_x1_159);
                    float ar_got_159 = __shfl_xor_sync(0xFFFFFFFF, ar_send_159, 4);
                    _er_f[109] = ((ar_bit_39 == 1) ? ar_got_159 : ar_x0_159);
                    _er_f[125] = ((ar_bit_39 == 1) ? ar_x1_159 : ar_got_159);
                    int ar_bit_40 = lane >> 1 & 1;
                    float ar_x0_160 = _er_f[97];
                    float ar_x1_160 = _er_f[105];
                    float ar_send_160 = ((ar_bit_40 == 1) ? ar_x0_160 : ar_x1_160);
                    float ar_got_160 = __shfl_xor_sync(0xFFFFFFFF, ar_send_160, 2);
                    _er_f[97] = ((ar_bit_40 == 1) ? ar_got_160 : ar_x0_160);
                    _er_f[105] = ((ar_bit_40 == 1) ? ar_x1_160 : ar_got_160);
                    float ar_x0_161 = _er_f[101];
                    float ar_x1_161 = _er_f[109];
                    float ar_send_161 = ((ar_bit_40 == 1) ? ar_x0_161 : ar_x1_161);
                    float ar_got_161 = __shfl_xor_sync(0xFFFFFFFF, ar_send_161, 2);
                    _er_f[101] = ((ar_bit_40 == 1) ? ar_got_161 : ar_x0_161);
                    _er_f[109] = ((ar_bit_40 == 1) ? ar_x1_161 : ar_got_161);
                    float ar_x0_162 = _er_f[113];
                    float ar_x1_162 = _er_f[121];
                    float ar_send_162 = ((ar_bit_40 == 1) ? ar_x0_162 : ar_x1_162);
                    float ar_got_162 = __shfl_xor_sync(0xFFFFFFFF, ar_send_162, 2);
                    _er_f[113] = ((ar_bit_40 == 1) ? ar_got_162 : ar_x0_162);
                    _er_f[121] = ((ar_bit_40 == 1) ? ar_x1_162 : ar_got_162);
                    float ar_x0_163 = _er_f[117];
                    float ar_x1_163 = _er_f[125];
                    float ar_send_163 = ((ar_bit_40 == 1) ? ar_x0_163 : ar_x1_163);
                    float ar_got_163 = __shfl_xor_sync(0xFFFFFFFF, ar_send_163, 2);
                    _er_f[117] = ((ar_bit_40 == 1) ? ar_got_163 : ar_x0_163);
                    _er_f[125] = ((ar_bit_40 == 1) ? ar_x1_163 : ar_got_163);
                    int ar_bit_41 = lane & 1;
                    float ar_x0_164 = _er_f[97];
                    float ar_x1_164 = _er_f[101];
                    float ar_send_164 = ((ar_bit_41 == 1) ? ar_x0_164 : ar_x1_164);
                    float ar_got_164 = __shfl_xor_sync(0xFFFFFFFF, ar_send_164, 1);
                    _er_f[97] = ((ar_bit_41 == 1) ? ar_got_164 : ar_x0_164);
                    _er_f[101] = ((ar_bit_41 == 1) ? ar_x1_164 : ar_got_164);
                    float ar_x0_165 = _er_f[105];
                    float ar_x1_165 = _er_f[109];
                    float ar_send_165 = ((ar_bit_41 == 1) ? ar_x0_165 : ar_x1_165);
                    float ar_got_165 = __shfl_xor_sync(0xFFFFFFFF, ar_send_165, 1);
                    _er_f[105] = ((ar_bit_41 == 1) ? ar_got_165 : ar_x0_165);
                    _er_f[109] = ((ar_bit_41 == 1) ? ar_x1_165 : ar_got_165);
                    float ar_x0_166 = _er_f[113];
                    float ar_x1_166 = _er_f[117];
                    float ar_send_166 = ((ar_bit_41 == 1) ? ar_x0_166 : ar_x1_166);
                    float ar_got_166 = __shfl_xor_sync(0xFFFFFFFF, ar_send_166, 1);
                    _er_f[113] = ((ar_bit_41 == 1) ? ar_got_166 : ar_x0_166);
                    _er_f[117] = ((ar_bit_41 == 1) ? ar_x1_166 : ar_got_166);
                    float ar_x0_167 = _er_f[121];
                    float ar_x1_167 = _er_f[125];
                    float ar_send_167 = ((ar_bit_41 == 1) ? ar_x0_167 : ar_x1_167);
                    float ar_got_167 = __shfl_xor_sync(0xFFFFFFFF, ar_send_167, 1);
                    _er_f[121] = ((ar_bit_41 == 1) ? ar_got_167 : ar_x0_167);
                    _er_f[125] = ((ar_bit_41 == 1) ? ar_x1_167 : ar_got_167);
                    int ar_bit_42 = lane >> 2 & 1;
                    float ar_x0_168 = _er_f[98];
                    float ar_x1_168 = _er_f[114];
                    float ar_send_168 = ((ar_bit_42 == 1) ? ar_x0_168 : ar_x1_168);
                    float ar_got_168 = __shfl_xor_sync(0xFFFFFFFF, ar_send_168, 4);
                    _er_f[98] = ((ar_bit_42 == 1) ? ar_got_168 : ar_x0_168);
                    _er_f[114] = ((ar_bit_42 == 1) ? ar_x1_168 : ar_got_168);
                    float ar_x0_169 = _er_f[102];
                    float ar_x1_169 = _er_f[118];
                    float ar_send_169 = ((ar_bit_42 == 1) ? ar_x0_169 : ar_x1_169);
                    float ar_got_169 = __shfl_xor_sync(0xFFFFFFFF, ar_send_169, 4);
                    _er_f[102] = ((ar_bit_42 == 1) ? ar_got_169 : ar_x0_169);
                    _er_f[118] = ((ar_bit_42 == 1) ? ar_x1_169 : ar_got_169);
                    float ar_x0_170 = _er_f[106];
                    float ar_x1_170 = _er_f[122];
                    float ar_send_170 = ((ar_bit_42 == 1) ? ar_x0_170 : ar_x1_170);
                    float ar_got_170 = __shfl_xor_sync(0xFFFFFFFF, ar_send_170, 4);
                    _er_f[106] = ((ar_bit_42 == 1) ? ar_got_170 : ar_x0_170);
                    _er_f[122] = ((ar_bit_42 == 1) ? ar_x1_170 : ar_got_170);
                    float ar_x0_171 = _er_f[110];
                    float ar_x1_171 = _er_f[126];
                    float ar_send_171 = ((ar_bit_42 == 1) ? ar_x0_171 : ar_x1_171);
                    float ar_got_171 = __shfl_xor_sync(0xFFFFFFFF, ar_send_171, 4);
                    _er_f[110] = ((ar_bit_42 == 1) ? ar_got_171 : ar_x0_171);
                    _er_f[126] = ((ar_bit_42 == 1) ? ar_x1_171 : ar_got_171);
                    int ar_bit_43 = lane >> 1 & 1;
                    float ar_x0_172 = _er_f[98];
                    float ar_x1_172 = _er_f[106];
                    float ar_send_172 = ((ar_bit_43 == 1) ? ar_x0_172 : ar_x1_172);
                    float ar_got_172 = __shfl_xor_sync(0xFFFFFFFF, ar_send_172, 2);
                    _er_f[98] = ((ar_bit_43 == 1) ? ar_got_172 : ar_x0_172);
                    _er_f[106] = ((ar_bit_43 == 1) ? ar_x1_172 : ar_got_172);
                    float ar_x0_173 = _er_f[102];
                    float ar_x1_173 = _er_f[110];
                    float ar_send_173 = ((ar_bit_43 == 1) ? ar_x0_173 : ar_x1_173);
                    float ar_got_173 = __shfl_xor_sync(0xFFFFFFFF, ar_send_173, 2);
                    _er_f[102] = ((ar_bit_43 == 1) ? ar_got_173 : ar_x0_173);
                    _er_f[110] = ((ar_bit_43 == 1) ? ar_x1_173 : ar_got_173);
                    float ar_x0_174 = _er_f[114];
                    float ar_x1_174 = _er_f[122];
                    float ar_send_174 = ((ar_bit_43 == 1) ? ar_x0_174 : ar_x1_174);
                    float ar_got_174 = __shfl_xor_sync(0xFFFFFFFF, ar_send_174, 2);
                    _er_f[114] = ((ar_bit_43 == 1) ? ar_got_174 : ar_x0_174);
                    _er_f[122] = ((ar_bit_43 == 1) ? ar_x1_174 : ar_got_174);
                    float ar_x0_175 = _er_f[118];
                    float ar_x1_175 = _er_f[126];
                    float ar_send_175 = ((ar_bit_43 == 1) ? ar_x0_175 : ar_x1_175);
                    float ar_got_175 = __shfl_xor_sync(0xFFFFFFFF, ar_send_175, 2);
                    _er_f[118] = ((ar_bit_43 == 1) ? ar_got_175 : ar_x0_175);
                    _er_f[126] = ((ar_bit_43 == 1) ? ar_x1_175 : ar_got_175);
                    int ar_bit_44 = lane & 1;
                    float ar_x0_176 = _er_f[98];
                    float ar_x1_176 = _er_f[102];
                    float ar_send_176 = ((ar_bit_44 == 1) ? ar_x0_176 : ar_x1_176);
                    float ar_got_176 = __shfl_xor_sync(0xFFFFFFFF, ar_send_176, 1);
                    _er_f[98] = ((ar_bit_44 == 1) ? ar_got_176 : ar_x0_176);
                    _er_f[102] = ((ar_bit_44 == 1) ? ar_x1_176 : ar_got_176);
                    float ar_x0_177 = _er_f[106];
                    float ar_x1_177 = _er_f[110];
                    float ar_send_177 = ((ar_bit_44 == 1) ? ar_x0_177 : ar_x1_177);
                    float ar_got_177 = __shfl_xor_sync(0xFFFFFFFF, ar_send_177, 1);
                    _er_f[106] = ((ar_bit_44 == 1) ? ar_got_177 : ar_x0_177);
                    _er_f[110] = ((ar_bit_44 == 1) ? ar_x1_177 : ar_got_177);
                    float ar_x0_178 = _er_f[114];
                    float ar_x1_178 = _er_f[118];
                    float ar_send_178 = ((ar_bit_44 == 1) ? ar_x0_178 : ar_x1_178);
                    float ar_got_178 = __shfl_xor_sync(0xFFFFFFFF, ar_send_178, 1);
                    _er_f[114] = ((ar_bit_44 == 1) ? ar_got_178 : ar_x0_178);
                    _er_f[118] = ((ar_bit_44 == 1) ? ar_x1_178 : ar_got_178);
                    float ar_x0_179 = _er_f[122];
                    float ar_x1_179 = _er_f[126];
                    float ar_send_179 = ((ar_bit_44 == 1) ? ar_x0_179 : ar_x1_179);
                    float ar_got_179 = __shfl_xor_sync(0xFFFFFFFF, ar_send_179, 1);
                    _er_f[122] = ((ar_bit_44 == 1) ? ar_got_179 : ar_x0_179);
                    _er_f[126] = ((ar_bit_44 == 1) ? ar_x1_179 : ar_got_179);
                    int ar_bit_45 = lane >> 2 & 1;
                    float ar_x0_180 = _er_f[99];
                    float ar_x1_180 = _er_f[115];
                    float ar_send_180 = ((ar_bit_45 == 1) ? ar_x0_180 : ar_x1_180);
                    float ar_got_180 = __shfl_xor_sync(0xFFFFFFFF, ar_send_180, 4);
                    _er_f[99] = ((ar_bit_45 == 1) ? ar_got_180 : ar_x0_180);
                    _er_f[115] = ((ar_bit_45 == 1) ? ar_x1_180 : ar_got_180);
                    float ar_x0_181 = _er_f[103];
                    float ar_x1_181 = _er_f[119];
                    float ar_send_181 = ((ar_bit_45 == 1) ? ar_x0_181 : ar_x1_181);
                    float ar_got_181 = __shfl_xor_sync(0xFFFFFFFF, ar_send_181, 4);
                    _er_f[103] = ((ar_bit_45 == 1) ? ar_got_181 : ar_x0_181);
                    _er_f[119] = ((ar_bit_45 == 1) ? ar_x1_181 : ar_got_181);
                    float ar_x0_182 = _er_f[107];
                    float ar_x1_182 = _er_f[123];
                    float ar_send_182 = ((ar_bit_45 == 1) ? ar_x0_182 : ar_x1_182);
                    float ar_got_182 = __shfl_xor_sync(0xFFFFFFFF, ar_send_182, 4);
                    _er_f[107] = ((ar_bit_45 == 1) ? ar_got_182 : ar_x0_182);
                    _er_f[123] = ((ar_bit_45 == 1) ? ar_x1_182 : ar_got_182);
                    float ar_x0_183 = _er_f[111];
                    float ar_x1_183 = _er_f[127];
                    float ar_send_183 = ((ar_bit_45 == 1) ? ar_x0_183 : ar_x1_183);
                    float ar_got_183 = __shfl_xor_sync(0xFFFFFFFF, ar_send_183, 4);
                    _er_f[111] = ((ar_bit_45 == 1) ? ar_got_183 : ar_x0_183);
                    _er_f[127] = ((ar_bit_45 == 1) ? ar_x1_183 : ar_got_183);
                    int ar_bit_46 = lane >> 1 & 1;
                    float ar_x0_184 = _er_f[99];
                    float ar_x1_184 = _er_f[107];
                    float ar_send_184 = ((ar_bit_46 == 1) ? ar_x0_184 : ar_x1_184);
                    float ar_got_184 = __shfl_xor_sync(0xFFFFFFFF, ar_send_184, 2);
                    _er_f[99] = ((ar_bit_46 == 1) ? ar_got_184 : ar_x0_184);
                    _er_f[107] = ((ar_bit_46 == 1) ? ar_x1_184 : ar_got_184);
                    float ar_x0_185 = _er_f[103];
                    float ar_x1_185 = _er_f[111];
                    float ar_send_185 = ((ar_bit_46 == 1) ? ar_x0_185 : ar_x1_185);
                    float ar_got_185 = __shfl_xor_sync(0xFFFFFFFF, ar_send_185, 2);
                    _er_f[103] = ((ar_bit_46 == 1) ? ar_got_185 : ar_x0_185);
                    _er_f[111] = ((ar_bit_46 == 1) ? ar_x1_185 : ar_got_185);
                    float ar_x0_186 = _er_f[115];
                    float ar_x1_186 = _er_f[123];
                    float ar_send_186 = ((ar_bit_46 == 1) ? ar_x0_186 : ar_x1_186);
                    float ar_got_186 = __shfl_xor_sync(0xFFFFFFFF, ar_send_186, 2);
                    _er_f[115] = ((ar_bit_46 == 1) ? ar_got_186 : ar_x0_186);
                    _er_f[123] = ((ar_bit_46 == 1) ? ar_x1_186 : ar_got_186);
                    float ar_x0_187 = _er_f[119];
                    float ar_x1_187 = _er_f[127];
                    float ar_send_187 = ((ar_bit_46 == 1) ? ar_x0_187 : ar_x1_187);
                    float ar_got_187 = __shfl_xor_sync(0xFFFFFFFF, ar_send_187, 2);
                    _er_f[119] = ((ar_bit_46 == 1) ? ar_got_187 : ar_x0_187);
                    _er_f[127] = ((ar_bit_46 == 1) ? ar_x1_187 : ar_got_187);
                    int ar_bit_47 = lane & 1;
                    float ar_x0_188 = _er_f[99];
                    float ar_x1_188 = _er_f[103];
                    float ar_send_188 = ((ar_bit_47 == 1) ? ar_x0_188 : ar_x1_188);
                    float ar_got_188 = __shfl_xor_sync(0xFFFFFFFF, ar_send_188, 1);
                    _er_f[99] = ((ar_bit_47 == 1) ? ar_got_188 : ar_x0_188);
                    _er_f[103] = ((ar_bit_47 == 1) ? ar_x1_188 : ar_got_188);
                    float ar_x0_189 = _er_f[107];
                    float ar_x1_189 = _er_f[111];
                    float ar_send_189 = ((ar_bit_47 == 1) ? ar_x0_189 : ar_x1_189);
                    float ar_got_189 = __shfl_xor_sync(0xFFFFFFFF, ar_send_189, 1);
                    _er_f[107] = ((ar_bit_47 == 1) ? ar_got_189 : ar_x0_189);
                    _er_f[111] = ((ar_bit_47 == 1) ? ar_x1_189 : ar_got_189);
                    float ar_x0_190 = _er_f[115];
                    float ar_x1_190 = _er_f[119];
                    float ar_send_190 = ((ar_bit_47 == 1) ? ar_x0_190 : ar_x1_190);
                    float ar_got_190 = __shfl_xor_sync(0xFFFFFFFF, ar_send_190, 1);
                    _er_f[115] = ((ar_bit_47 == 1) ? ar_got_190 : ar_x0_190);
                    _er_f[119] = ((ar_bit_47 == 1) ? ar_x1_190 : ar_got_190);
                    float ar_x0_191 = _er_f[123];
                    float ar_x1_191 = _er_f[127];
                    float ar_send_191 = ((ar_bit_47 == 1) ? ar_x0_191 : ar_x1_191);
                    float ar_got_191 = __shfl_xor_sync(0xFFFFFFFF, ar_send_191, 1);
                    _er_f[123] = ((ar_bit_47 == 1) ? ar_got_191 : ar_x0_191);
                    _er_f[127] = ((ar_bit_47 == 1) ? ar_x1_191 : ar_got_191);
                    int ar_ri_3 = er_at2 + 96;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_3])), "f"(_er_f[96]), "f"(_er_f[97]), "f"(_er_f[98]), "f"(_er_f[99]) : "memory");
                    ar_ri_3 = ar_ri_3 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_3])), "f"(_er_f[100]), "f"(_er_f[101]), "f"(_er_f[102]), "f"(_er_f[103]) : "memory");
                    ar_ri_3 = ar_ri_3 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_3])), "f"(_er_f[104]), "f"(_er_f[105]), "f"(_er_f[106]), "f"(_er_f[107]) : "memory");
                    ar_ri_3 = ar_ri_3 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_3])), "f"(_er_f[108]), "f"(_er_f[109]), "f"(_er_f[110]), "f"(_er_f[111]) : "memory");
                    ar_ri_3 = ar_ri_3 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_3])), "f"(_er_f[112]), "f"(_er_f[113]), "f"(_er_f[114]), "f"(_er_f[115]) : "memory");
                    ar_ri_3 = ar_ri_3 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_3])), "f"(_er_f[116]), "f"(_er_f[117]), "f"(_er_f[118]), "f"(_er_f[119]) : "memory");
                    ar_ri_3 = ar_ri_3 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_3])), "f"(_er_f[120]), "f"(_er_f[121]), "f"(_er_f[122]), "f"(_er_f[123]) : "memory");
                    ar_ri_3 = ar_ri_3 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_3])), "f"(_er_f[124]), "f"(_er_f[125]), "f"(_er_f[126]), "f"(_er_f[127]) : "memory");
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_er_f[0]), "=f"(_er_f[1]), "=f"(_er_f[2]), "=f"(_er_f[3]), "=f"(_er_f[4]), "=f"(_er_f[5]), "=f"(_er_f[6]), "=f"(_er_f[7]), "=f"(_er_f[8]), "=f"(_er_f[9]), "=f"(_er_f[10]), "=f"(_er_f[11]), "=f"(_er_f[12]), "=f"(_er_f[13]), "=f"(_er_f[14]), "=f"(_er_f[15]), "=f"(_er_f[16]), "=f"(_er_f[17]), "=f"(_er_f[18]), "=f"(_er_f[19]), "=f"(_er_f[20]), "=f"(_er_f[21]), "=f"(_er_f[22]), "=f"(_er_f[23]), "=f"(_er_f[24]), "=f"(_er_f[25]), "=f"(_er_f[26]), "=f"(_er_f[27]), "=f"(_er_f[28]), "=f"(_er_f[29]), "=f"(_er_f[30]), "=f"(_er_f[31])
                    : "r"(er_rowf + 256));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_er_f[32]), "=f"(_er_f[33]), "=f"(_er_f[34]), "=f"(_er_f[35]), "=f"(_er_f[36]), "=f"(_er_f[37]), "=f"(_er_f[38]), "=f"(_er_f[39]), "=f"(_er_f[40]), "=f"(_er_f[41]), "=f"(_er_f[42]), "=f"(_er_f[43]), "=f"(_er_f[44]), "=f"(_er_f[45]), "=f"(_er_f[46]), "=f"(_er_f[47]), "=f"(_er_f[48]), "=f"(_er_f[49]), "=f"(_er_f[50]), "=f"(_er_f[51]), "=f"(_er_f[52]), "=f"(_er_f[53]), "=f"(_er_f[54]), "=f"(_er_f[55]), "=f"(_er_f[56]), "=f"(_er_f[57]), "=f"(_er_f[58]), "=f"(_er_f[59]), "=f"(_er_f[60]), "=f"(_er_f[61]), "=f"(_er_f[62]), "=f"(_er_f[63])
                    : "r"(er_rowf + 288));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_er_f[64]), "=f"(_er_f[65]), "=f"(_er_f[66]), "=f"(_er_f[67]), "=f"(_er_f[68]), "=f"(_er_f[69]), "=f"(_er_f[70]), "=f"(_er_f[71]), "=f"(_er_f[72]), "=f"(_er_f[73]), "=f"(_er_f[74]), "=f"(_er_f[75]), "=f"(_er_f[76]), "=f"(_er_f[77]), "=f"(_er_f[78]), "=f"(_er_f[79]), "=f"(_er_f[80]), "=f"(_er_f[81]), "=f"(_er_f[82]), "=f"(_er_f[83]), "=f"(_er_f[84]), "=f"(_er_f[85]), "=f"(_er_f[86]), "=f"(_er_f[87]), "=f"(_er_f[88]), "=f"(_er_f[89]), "=f"(_er_f[90]), "=f"(_er_f[91]), "=f"(_er_f[92]), "=f"(_er_f[93]), "=f"(_er_f[94]), "=f"(_er_f[95])
                    : "r"(er_rowf + 320));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_er_f[96]), "=f"(_er_f[97]), "=f"(_er_f[98]), "=f"(_er_f[99]), "=f"(_er_f[100]), "=f"(_er_f[101]), "=f"(_er_f[102]), "=f"(_er_f[103]), "=f"(_er_f[104]), "=f"(_er_f[105]), "=f"(_er_f[106]), "=f"(_er_f[107]), "=f"(_er_f[108]), "=f"(_er_f[109]), "=f"(_er_f[110]), "=f"(_er_f[111]), "=f"(_er_f[112]), "=f"(_er_f[113]), "=f"(_er_f[114]), "=f"(_er_f[115]), "=f"(_er_f[116]), "=f"(_er_f[117]), "=f"(_er_f[118]), "=f"(_er_f[119]), "=f"(_er_f[120]), "=f"(_er_f[121]), "=f"(_er_f[122]), "=f"(_er_f[123]), "=f"(_er_f[124]), "=f"(_er_f[125]), "=f"(_er_f[126]), "=f"(_er_f[127])
                    : "r"(er_rowf + 352));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                if (ac_live == 1) {
                    int ar_bit_48 = lane >> 2 & 1;
                    float ar_x0_192 = _er_f[0];
                    float ar_x1_192 = _er_f[16];
                    float ar_send_192 = ((ar_bit_48 == 1) ? ar_x0_192 : ar_x1_192);
                    float ar_got_192 = __shfl_xor_sync(0xFFFFFFFF, ar_send_192, 4);
                    _er_f[0] = ((ar_bit_48 == 1) ? ar_got_192 : ar_x0_192);
                    _er_f[16] = ((ar_bit_48 == 1) ? ar_x1_192 : ar_got_192);
                    float ar_x0_193 = _er_f[4];
                    float ar_x1_193 = _er_f[20];
                    float ar_send_193 = ((ar_bit_48 == 1) ? ar_x0_193 : ar_x1_193);
                    float ar_got_193 = __shfl_xor_sync(0xFFFFFFFF, ar_send_193, 4);
                    _er_f[4] = ((ar_bit_48 == 1) ? ar_got_193 : ar_x0_193);
                    _er_f[20] = ((ar_bit_48 == 1) ? ar_x1_193 : ar_got_193);
                    float ar_x0_194 = _er_f[8];
                    float ar_x1_194 = _er_f[24];
                    float ar_send_194 = ((ar_bit_48 == 1) ? ar_x0_194 : ar_x1_194);
                    float ar_got_194 = __shfl_xor_sync(0xFFFFFFFF, ar_send_194, 4);
                    _er_f[8] = ((ar_bit_48 == 1) ? ar_got_194 : ar_x0_194);
                    _er_f[24] = ((ar_bit_48 == 1) ? ar_x1_194 : ar_got_194);
                    float ar_x0_195 = _er_f[12];
                    float ar_x1_195 = _er_f[28];
                    float ar_send_195 = ((ar_bit_48 == 1) ? ar_x0_195 : ar_x1_195);
                    float ar_got_195 = __shfl_xor_sync(0xFFFFFFFF, ar_send_195, 4);
                    _er_f[12] = ((ar_bit_48 == 1) ? ar_got_195 : ar_x0_195);
                    _er_f[28] = ((ar_bit_48 == 1) ? ar_x1_195 : ar_got_195);
                    int ar_bit_49 = lane >> 1 & 1;
                    float ar_x0_196 = _er_f[0];
                    float ar_x1_196 = _er_f[8];
                    float ar_send_196 = ((ar_bit_49 == 1) ? ar_x0_196 : ar_x1_196);
                    float ar_got_196 = __shfl_xor_sync(0xFFFFFFFF, ar_send_196, 2);
                    _er_f[0] = ((ar_bit_49 == 1) ? ar_got_196 : ar_x0_196);
                    _er_f[8] = ((ar_bit_49 == 1) ? ar_x1_196 : ar_got_196);
                    float ar_x0_197 = _er_f[4];
                    float ar_x1_197 = _er_f[12];
                    float ar_send_197 = ((ar_bit_49 == 1) ? ar_x0_197 : ar_x1_197);
                    float ar_got_197 = __shfl_xor_sync(0xFFFFFFFF, ar_send_197, 2);
                    _er_f[4] = ((ar_bit_49 == 1) ? ar_got_197 : ar_x0_197);
                    _er_f[12] = ((ar_bit_49 == 1) ? ar_x1_197 : ar_got_197);
                    float ar_x0_198 = _er_f[16];
                    float ar_x1_198 = _er_f[24];
                    float ar_send_198 = ((ar_bit_49 == 1) ? ar_x0_198 : ar_x1_198);
                    float ar_got_198 = __shfl_xor_sync(0xFFFFFFFF, ar_send_198, 2);
                    _er_f[16] = ((ar_bit_49 == 1) ? ar_got_198 : ar_x0_198);
                    _er_f[24] = ((ar_bit_49 == 1) ? ar_x1_198 : ar_got_198);
                    float ar_x0_199 = _er_f[20];
                    float ar_x1_199 = _er_f[28];
                    float ar_send_199 = ((ar_bit_49 == 1) ? ar_x0_199 : ar_x1_199);
                    float ar_got_199 = __shfl_xor_sync(0xFFFFFFFF, ar_send_199, 2);
                    _er_f[20] = ((ar_bit_49 == 1) ? ar_got_199 : ar_x0_199);
                    _er_f[28] = ((ar_bit_49 == 1) ? ar_x1_199 : ar_got_199);
                    int ar_bit_50 = lane & 1;
                    float ar_x0_200 = _er_f[0];
                    float ar_x1_200 = _er_f[4];
                    float ar_send_200 = ((ar_bit_50 == 1) ? ar_x0_200 : ar_x1_200);
                    float ar_got_200 = __shfl_xor_sync(0xFFFFFFFF, ar_send_200, 1);
                    _er_f[0] = ((ar_bit_50 == 1) ? ar_got_200 : ar_x0_200);
                    _er_f[4] = ((ar_bit_50 == 1) ? ar_x1_200 : ar_got_200);
                    float ar_x0_201 = _er_f[8];
                    float ar_x1_201 = _er_f[12];
                    float ar_send_201 = ((ar_bit_50 == 1) ? ar_x0_201 : ar_x1_201);
                    float ar_got_201 = __shfl_xor_sync(0xFFFFFFFF, ar_send_201, 1);
                    _er_f[8] = ((ar_bit_50 == 1) ? ar_got_201 : ar_x0_201);
                    _er_f[12] = ((ar_bit_50 == 1) ? ar_x1_201 : ar_got_201);
                    float ar_x0_202 = _er_f[16];
                    float ar_x1_202 = _er_f[20];
                    float ar_send_202 = ((ar_bit_50 == 1) ? ar_x0_202 : ar_x1_202);
                    float ar_got_202 = __shfl_xor_sync(0xFFFFFFFF, ar_send_202, 1);
                    _er_f[16] = ((ar_bit_50 == 1) ? ar_got_202 : ar_x0_202);
                    _er_f[20] = ((ar_bit_50 == 1) ? ar_x1_202 : ar_got_202);
                    float ar_x0_203 = _er_f[24];
                    float ar_x1_203 = _er_f[28];
                    float ar_send_203 = ((ar_bit_50 == 1) ? ar_x0_203 : ar_x1_203);
                    float ar_got_203 = __shfl_xor_sync(0xFFFFFFFF, ar_send_203, 1);
                    _er_f[24] = ((ar_bit_50 == 1) ? ar_got_203 : ar_x0_203);
                    _er_f[28] = ((ar_bit_50 == 1) ? ar_x1_203 : ar_got_203);
                    int ar_bit_51 = lane >> 2 & 1;
                    float ar_x0_204 = _er_f[1];
                    float ar_x1_204 = _er_f[17];
                    float ar_send_204 = ((ar_bit_51 == 1) ? ar_x0_204 : ar_x1_204);
                    float ar_got_204 = __shfl_xor_sync(0xFFFFFFFF, ar_send_204, 4);
                    _er_f[1] = ((ar_bit_51 == 1) ? ar_got_204 : ar_x0_204);
                    _er_f[17] = ((ar_bit_51 == 1) ? ar_x1_204 : ar_got_204);
                    float ar_x0_205 = _er_f[5];
                    float ar_x1_205 = _er_f[21];
                    float ar_send_205 = ((ar_bit_51 == 1) ? ar_x0_205 : ar_x1_205);
                    float ar_got_205 = __shfl_xor_sync(0xFFFFFFFF, ar_send_205, 4);
                    _er_f[5] = ((ar_bit_51 == 1) ? ar_got_205 : ar_x0_205);
                    _er_f[21] = ((ar_bit_51 == 1) ? ar_x1_205 : ar_got_205);
                    float ar_x0_206 = _er_f[9];
                    float ar_x1_206 = _er_f[25];
                    float ar_send_206 = ((ar_bit_51 == 1) ? ar_x0_206 : ar_x1_206);
                    float ar_got_206 = __shfl_xor_sync(0xFFFFFFFF, ar_send_206, 4);
                    _er_f[9] = ((ar_bit_51 == 1) ? ar_got_206 : ar_x0_206);
                    _er_f[25] = ((ar_bit_51 == 1) ? ar_x1_206 : ar_got_206);
                    float ar_x0_207 = _er_f[13];
                    float ar_x1_207 = _er_f[29];
                    float ar_send_207 = ((ar_bit_51 == 1) ? ar_x0_207 : ar_x1_207);
                    float ar_got_207 = __shfl_xor_sync(0xFFFFFFFF, ar_send_207, 4);
                    _er_f[13] = ((ar_bit_51 == 1) ? ar_got_207 : ar_x0_207);
                    _er_f[29] = ((ar_bit_51 == 1) ? ar_x1_207 : ar_got_207);
                    int ar_bit_52 = lane >> 1 & 1;
                    float ar_x0_208 = _er_f[1];
                    float ar_x1_208 = _er_f[9];
                    float ar_send_208 = ((ar_bit_52 == 1) ? ar_x0_208 : ar_x1_208);
                    float ar_got_208 = __shfl_xor_sync(0xFFFFFFFF, ar_send_208, 2);
                    _er_f[1] = ((ar_bit_52 == 1) ? ar_got_208 : ar_x0_208);
                    _er_f[9] = ((ar_bit_52 == 1) ? ar_x1_208 : ar_got_208);
                    float ar_x0_209 = _er_f[5];
                    float ar_x1_209 = _er_f[13];
                    float ar_send_209 = ((ar_bit_52 == 1) ? ar_x0_209 : ar_x1_209);
                    float ar_got_209 = __shfl_xor_sync(0xFFFFFFFF, ar_send_209, 2);
                    _er_f[5] = ((ar_bit_52 == 1) ? ar_got_209 : ar_x0_209);
                    _er_f[13] = ((ar_bit_52 == 1) ? ar_x1_209 : ar_got_209);
                    float ar_x0_210 = _er_f[17];
                    float ar_x1_210 = _er_f[25];
                    float ar_send_210 = ((ar_bit_52 == 1) ? ar_x0_210 : ar_x1_210);
                    float ar_got_210 = __shfl_xor_sync(0xFFFFFFFF, ar_send_210, 2);
                    _er_f[17] = ((ar_bit_52 == 1) ? ar_got_210 : ar_x0_210);
                    _er_f[25] = ((ar_bit_52 == 1) ? ar_x1_210 : ar_got_210);
                    float ar_x0_211 = _er_f[21];
                    float ar_x1_211 = _er_f[29];
                    float ar_send_211 = ((ar_bit_52 == 1) ? ar_x0_211 : ar_x1_211);
                    float ar_got_211 = __shfl_xor_sync(0xFFFFFFFF, ar_send_211, 2);
                    _er_f[21] = ((ar_bit_52 == 1) ? ar_got_211 : ar_x0_211);
                    _er_f[29] = ((ar_bit_52 == 1) ? ar_x1_211 : ar_got_211);
                    int ar_bit_53 = lane & 1;
                    float ar_x0_212 = _er_f[1];
                    float ar_x1_212 = _er_f[5];
                    float ar_send_212 = ((ar_bit_53 == 1) ? ar_x0_212 : ar_x1_212);
                    float ar_got_212 = __shfl_xor_sync(0xFFFFFFFF, ar_send_212, 1);
                    _er_f[1] = ((ar_bit_53 == 1) ? ar_got_212 : ar_x0_212);
                    _er_f[5] = ((ar_bit_53 == 1) ? ar_x1_212 : ar_got_212);
                    float ar_x0_213 = _er_f[9];
                    float ar_x1_213 = _er_f[13];
                    float ar_send_213 = ((ar_bit_53 == 1) ? ar_x0_213 : ar_x1_213);
                    float ar_got_213 = __shfl_xor_sync(0xFFFFFFFF, ar_send_213, 1);
                    _er_f[9] = ((ar_bit_53 == 1) ? ar_got_213 : ar_x0_213);
                    _er_f[13] = ((ar_bit_53 == 1) ? ar_x1_213 : ar_got_213);
                    float ar_x0_214 = _er_f[17];
                    float ar_x1_214 = _er_f[21];
                    float ar_send_214 = ((ar_bit_53 == 1) ? ar_x0_214 : ar_x1_214);
                    float ar_got_214 = __shfl_xor_sync(0xFFFFFFFF, ar_send_214, 1);
                    _er_f[17] = ((ar_bit_53 == 1) ? ar_got_214 : ar_x0_214);
                    _er_f[21] = ((ar_bit_53 == 1) ? ar_x1_214 : ar_got_214);
                    float ar_x0_215 = _er_f[25];
                    float ar_x1_215 = _er_f[29];
                    float ar_send_215 = ((ar_bit_53 == 1) ? ar_x0_215 : ar_x1_215);
                    float ar_got_215 = __shfl_xor_sync(0xFFFFFFFF, ar_send_215, 1);
                    _er_f[25] = ((ar_bit_53 == 1) ? ar_got_215 : ar_x0_215);
                    _er_f[29] = ((ar_bit_53 == 1) ? ar_x1_215 : ar_got_215);
                    int ar_bit_54 = lane >> 2 & 1;
                    float ar_x0_216 = _er_f[2];
                    float ar_x1_216 = _er_f[18];
                    float ar_send_216 = ((ar_bit_54 == 1) ? ar_x0_216 : ar_x1_216);
                    float ar_got_216 = __shfl_xor_sync(0xFFFFFFFF, ar_send_216, 4);
                    _er_f[2] = ((ar_bit_54 == 1) ? ar_got_216 : ar_x0_216);
                    _er_f[18] = ((ar_bit_54 == 1) ? ar_x1_216 : ar_got_216);
                    float ar_x0_217 = _er_f[6];
                    float ar_x1_217 = _er_f[22];
                    float ar_send_217 = ((ar_bit_54 == 1) ? ar_x0_217 : ar_x1_217);
                    float ar_got_217 = __shfl_xor_sync(0xFFFFFFFF, ar_send_217, 4);
                    _er_f[6] = ((ar_bit_54 == 1) ? ar_got_217 : ar_x0_217);
                    _er_f[22] = ((ar_bit_54 == 1) ? ar_x1_217 : ar_got_217);
                    float ar_x0_218 = _er_f[10];
                    float ar_x1_218 = _er_f[26];
                    float ar_send_218 = ((ar_bit_54 == 1) ? ar_x0_218 : ar_x1_218);
                    float ar_got_218 = __shfl_xor_sync(0xFFFFFFFF, ar_send_218, 4);
                    _er_f[10] = ((ar_bit_54 == 1) ? ar_got_218 : ar_x0_218);
                    _er_f[26] = ((ar_bit_54 == 1) ? ar_x1_218 : ar_got_218);
                    float ar_x0_219 = _er_f[14];
                    float ar_x1_219 = _er_f[30];
                    float ar_send_219 = ((ar_bit_54 == 1) ? ar_x0_219 : ar_x1_219);
                    float ar_got_219 = __shfl_xor_sync(0xFFFFFFFF, ar_send_219, 4);
                    _er_f[14] = ((ar_bit_54 == 1) ? ar_got_219 : ar_x0_219);
                    _er_f[30] = ((ar_bit_54 == 1) ? ar_x1_219 : ar_got_219);
                    int ar_bit_55 = lane >> 1 & 1;
                    float ar_x0_220 = _er_f[2];
                    float ar_x1_220 = _er_f[10];
                    float ar_send_220 = ((ar_bit_55 == 1) ? ar_x0_220 : ar_x1_220);
                    float ar_got_220 = __shfl_xor_sync(0xFFFFFFFF, ar_send_220, 2);
                    _er_f[2] = ((ar_bit_55 == 1) ? ar_got_220 : ar_x0_220);
                    _er_f[10] = ((ar_bit_55 == 1) ? ar_x1_220 : ar_got_220);
                    float ar_x0_221 = _er_f[6];
                    float ar_x1_221 = _er_f[14];
                    float ar_send_221 = ((ar_bit_55 == 1) ? ar_x0_221 : ar_x1_221);
                    float ar_got_221 = __shfl_xor_sync(0xFFFFFFFF, ar_send_221, 2);
                    _er_f[6] = ((ar_bit_55 == 1) ? ar_got_221 : ar_x0_221);
                    _er_f[14] = ((ar_bit_55 == 1) ? ar_x1_221 : ar_got_221);
                    float ar_x0_222 = _er_f[18];
                    float ar_x1_222 = _er_f[26];
                    float ar_send_222 = ((ar_bit_55 == 1) ? ar_x0_222 : ar_x1_222);
                    float ar_got_222 = __shfl_xor_sync(0xFFFFFFFF, ar_send_222, 2);
                    _er_f[18] = ((ar_bit_55 == 1) ? ar_got_222 : ar_x0_222);
                    _er_f[26] = ((ar_bit_55 == 1) ? ar_x1_222 : ar_got_222);
                    float ar_x0_223 = _er_f[22];
                    float ar_x1_223 = _er_f[30];
                    float ar_send_223 = ((ar_bit_55 == 1) ? ar_x0_223 : ar_x1_223);
                    float ar_got_223 = __shfl_xor_sync(0xFFFFFFFF, ar_send_223, 2);
                    _er_f[22] = ((ar_bit_55 == 1) ? ar_got_223 : ar_x0_223);
                    _er_f[30] = ((ar_bit_55 == 1) ? ar_x1_223 : ar_got_223);
                    int ar_bit_56 = lane & 1;
                    float ar_x0_224 = _er_f[2];
                    float ar_x1_224 = _er_f[6];
                    float ar_send_224 = ((ar_bit_56 == 1) ? ar_x0_224 : ar_x1_224);
                    float ar_got_224 = __shfl_xor_sync(0xFFFFFFFF, ar_send_224, 1);
                    _er_f[2] = ((ar_bit_56 == 1) ? ar_got_224 : ar_x0_224);
                    _er_f[6] = ((ar_bit_56 == 1) ? ar_x1_224 : ar_got_224);
                    float ar_x0_225 = _er_f[10];
                    float ar_x1_225 = _er_f[14];
                    float ar_send_225 = ((ar_bit_56 == 1) ? ar_x0_225 : ar_x1_225);
                    float ar_got_225 = __shfl_xor_sync(0xFFFFFFFF, ar_send_225, 1);
                    _er_f[10] = ((ar_bit_56 == 1) ? ar_got_225 : ar_x0_225);
                    _er_f[14] = ((ar_bit_56 == 1) ? ar_x1_225 : ar_got_225);
                    float ar_x0_226 = _er_f[18];
                    float ar_x1_226 = _er_f[22];
                    float ar_send_226 = ((ar_bit_56 == 1) ? ar_x0_226 : ar_x1_226);
                    float ar_got_226 = __shfl_xor_sync(0xFFFFFFFF, ar_send_226, 1);
                    _er_f[18] = ((ar_bit_56 == 1) ? ar_got_226 : ar_x0_226);
                    _er_f[22] = ((ar_bit_56 == 1) ? ar_x1_226 : ar_got_226);
                    float ar_x0_227 = _er_f[26];
                    float ar_x1_227 = _er_f[30];
                    float ar_send_227 = ((ar_bit_56 == 1) ? ar_x0_227 : ar_x1_227);
                    float ar_got_227 = __shfl_xor_sync(0xFFFFFFFF, ar_send_227, 1);
                    _er_f[26] = ((ar_bit_56 == 1) ? ar_got_227 : ar_x0_227);
                    _er_f[30] = ((ar_bit_56 == 1) ? ar_x1_227 : ar_got_227);
                    int ar_bit_57 = lane >> 2 & 1;
                    float ar_x0_228 = _er_f[3];
                    float ar_x1_228 = _er_f[19];
                    float ar_send_228 = ((ar_bit_57 == 1) ? ar_x0_228 : ar_x1_228);
                    float ar_got_228 = __shfl_xor_sync(0xFFFFFFFF, ar_send_228, 4);
                    _er_f[3] = ((ar_bit_57 == 1) ? ar_got_228 : ar_x0_228);
                    _er_f[19] = ((ar_bit_57 == 1) ? ar_x1_228 : ar_got_228);
                    float ar_x0_229 = _er_f[7];
                    float ar_x1_229 = _er_f[23];
                    float ar_send_229 = ((ar_bit_57 == 1) ? ar_x0_229 : ar_x1_229);
                    float ar_got_229 = __shfl_xor_sync(0xFFFFFFFF, ar_send_229, 4);
                    _er_f[7] = ((ar_bit_57 == 1) ? ar_got_229 : ar_x0_229);
                    _er_f[23] = ((ar_bit_57 == 1) ? ar_x1_229 : ar_got_229);
                    float ar_x0_230 = _er_f[11];
                    float ar_x1_230 = _er_f[27];
                    float ar_send_230 = ((ar_bit_57 == 1) ? ar_x0_230 : ar_x1_230);
                    float ar_got_230 = __shfl_xor_sync(0xFFFFFFFF, ar_send_230, 4);
                    _er_f[11] = ((ar_bit_57 == 1) ? ar_got_230 : ar_x0_230);
                    _er_f[27] = ((ar_bit_57 == 1) ? ar_x1_230 : ar_got_230);
                    float ar_x0_231 = _er_f[15];
                    float ar_x1_231 = _er_f[31];
                    float ar_send_231 = ((ar_bit_57 == 1) ? ar_x0_231 : ar_x1_231);
                    float ar_got_231 = __shfl_xor_sync(0xFFFFFFFF, ar_send_231, 4);
                    _er_f[15] = ((ar_bit_57 == 1) ? ar_got_231 : ar_x0_231);
                    _er_f[31] = ((ar_bit_57 == 1) ? ar_x1_231 : ar_got_231);
                    int ar_bit_58 = lane >> 1 & 1;
                    float ar_x0_232 = _er_f[3];
                    float ar_x1_232 = _er_f[11];
                    float ar_send_232 = ((ar_bit_58 == 1) ? ar_x0_232 : ar_x1_232);
                    float ar_got_232 = __shfl_xor_sync(0xFFFFFFFF, ar_send_232, 2);
                    _er_f[3] = ((ar_bit_58 == 1) ? ar_got_232 : ar_x0_232);
                    _er_f[11] = ((ar_bit_58 == 1) ? ar_x1_232 : ar_got_232);
                    float ar_x0_233 = _er_f[7];
                    float ar_x1_233 = _er_f[15];
                    float ar_send_233 = ((ar_bit_58 == 1) ? ar_x0_233 : ar_x1_233);
                    float ar_got_233 = __shfl_xor_sync(0xFFFFFFFF, ar_send_233, 2);
                    _er_f[7] = ((ar_bit_58 == 1) ? ar_got_233 : ar_x0_233);
                    _er_f[15] = ((ar_bit_58 == 1) ? ar_x1_233 : ar_got_233);
                    float ar_x0_234 = _er_f[19];
                    float ar_x1_234 = _er_f[27];
                    float ar_send_234 = ((ar_bit_58 == 1) ? ar_x0_234 : ar_x1_234);
                    float ar_got_234 = __shfl_xor_sync(0xFFFFFFFF, ar_send_234, 2);
                    _er_f[19] = ((ar_bit_58 == 1) ? ar_got_234 : ar_x0_234);
                    _er_f[27] = ((ar_bit_58 == 1) ? ar_x1_234 : ar_got_234);
                    float ar_x0_235 = _er_f[23];
                    float ar_x1_235 = _er_f[31];
                    float ar_send_235 = ((ar_bit_58 == 1) ? ar_x0_235 : ar_x1_235);
                    float ar_got_235 = __shfl_xor_sync(0xFFFFFFFF, ar_send_235, 2);
                    _er_f[23] = ((ar_bit_58 == 1) ? ar_got_235 : ar_x0_235);
                    _er_f[31] = ((ar_bit_58 == 1) ? ar_x1_235 : ar_got_235);
                    int ar_bit_59 = lane & 1;
                    float ar_x0_236 = _er_f[3];
                    float ar_x1_236 = _er_f[7];
                    float ar_send_236 = ((ar_bit_59 == 1) ? ar_x0_236 : ar_x1_236);
                    float ar_got_236 = __shfl_xor_sync(0xFFFFFFFF, ar_send_236, 1);
                    _er_f[3] = ((ar_bit_59 == 1) ? ar_got_236 : ar_x0_236);
                    _er_f[7] = ((ar_bit_59 == 1) ? ar_x1_236 : ar_got_236);
                    float ar_x0_237 = _er_f[11];
                    float ar_x1_237 = _er_f[15];
                    float ar_send_237 = ((ar_bit_59 == 1) ? ar_x0_237 : ar_x1_237);
                    float ar_got_237 = __shfl_xor_sync(0xFFFFFFFF, ar_send_237, 1);
                    _er_f[11] = ((ar_bit_59 == 1) ? ar_got_237 : ar_x0_237);
                    _er_f[15] = ((ar_bit_59 == 1) ? ar_x1_237 : ar_got_237);
                    float ar_x0_238 = _er_f[19];
                    float ar_x1_238 = _er_f[23];
                    float ar_send_238 = ((ar_bit_59 == 1) ? ar_x0_238 : ar_x1_238);
                    float ar_got_238 = __shfl_xor_sync(0xFFFFFFFF, ar_send_238, 1);
                    _er_f[19] = ((ar_bit_59 == 1) ? ar_got_238 : ar_x0_238);
                    _er_f[23] = ((ar_bit_59 == 1) ? ar_x1_238 : ar_got_238);
                    float ar_x0_239 = _er_f[27];
                    float ar_x1_239 = _er_f[31];
                    float ar_send_239 = ((ar_bit_59 == 1) ? ar_x0_239 : ar_x1_239);
                    float ar_got_239 = __shfl_xor_sync(0xFFFFFFFF, ar_send_239, 1);
                    _er_f[27] = ((ar_bit_59 == 1) ? ar_got_239 : ar_x0_239);
                    _er_f[31] = ((ar_bit_59 == 1) ? ar_x1_239 : ar_got_239);
                    int ar_ri_4 = er_at2 + 256;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_4])), "f"(_er_f[0]), "f"(_er_f[1]), "f"(_er_f[2]), "f"(_er_f[3]) : "memory");
                    ar_ri_4 = ar_ri_4 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_4])), "f"(_er_f[4]), "f"(_er_f[5]), "f"(_er_f[6]), "f"(_er_f[7]) : "memory");
                    ar_ri_4 = ar_ri_4 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_4])), "f"(_er_f[8]), "f"(_er_f[9]), "f"(_er_f[10]), "f"(_er_f[11]) : "memory");
                    ar_ri_4 = ar_ri_4 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_4])), "f"(_er_f[12]), "f"(_er_f[13]), "f"(_er_f[14]), "f"(_er_f[15]) : "memory");
                    ar_ri_4 = ar_ri_4 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_4])), "f"(_er_f[16]), "f"(_er_f[17]), "f"(_er_f[18]), "f"(_er_f[19]) : "memory");
                    ar_ri_4 = ar_ri_4 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_4])), "f"(_er_f[20]), "f"(_er_f[21]), "f"(_er_f[22]), "f"(_er_f[23]) : "memory");
                    ar_ri_4 = ar_ri_4 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_4])), "f"(_er_f[24]), "f"(_er_f[25]), "f"(_er_f[26]), "f"(_er_f[27]) : "memory");
                    ar_ri_4 = ar_ri_4 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_4])), "f"(_er_f[28]), "f"(_er_f[29]), "f"(_er_f[30]), "f"(_er_f[31]) : "memory");
                    int ar_bit_60 = lane >> 2 & 1;
                    float ar_x0_240 = _er_f[32];
                    float ar_x1_240 = _er_f[48];
                    float ar_send_240 = ((ar_bit_60 == 1) ? ar_x0_240 : ar_x1_240);
                    float ar_got_240 = __shfl_xor_sync(0xFFFFFFFF, ar_send_240, 4);
                    _er_f[32] = ((ar_bit_60 == 1) ? ar_got_240 : ar_x0_240);
                    _er_f[48] = ((ar_bit_60 == 1) ? ar_x1_240 : ar_got_240);
                    float ar_x0_241 = _er_f[36];
                    float ar_x1_241 = _er_f[52];
                    float ar_send_241 = ((ar_bit_60 == 1) ? ar_x0_241 : ar_x1_241);
                    float ar_got_241 = __shfl_xor_sync(0xFFFFFFFF, ar_send_241, 4);
                    _er_f[36] = ((ar_bit_60 == 1) ? ar_got_241 : ar_x0_241);
                    _er_f[52] = ((ar_bit_60 == 1) ? ar_x1_241 : ar_got_241);
                    float ar_x0_242 = _er_f[40];
                    float ar_x1_242 = _er_f[56];
                    float ar_send_242 = ((ar_bit_60 == 1) ? ar_x0_242 : ar_x1_242);
                    float ar_got_242 = __shfl_xor_sync(0xFFFFFFFF, ar_send_242, 4);
                    _er_f[40] = ((ar_bit_60 == 1) ? ar_got_242 : ar_x0_242);
                    _er_f[56] = ((ar_bit_60 == 1) ? ar_x1_242 : ar_got_242);
                    float ar_x0_243 = _er_f[44];
                    float ar_x1_243 = _er_f[60];
                    float ar_send_243 = ((ar_bit_60 == 1) ? ar_x0_243 : ar_x1_243);
                    float ar_got_243 = __shfl_xor_sync(0xFFFFFFFF, ar_send_243, 4);
                    _er_f[44] = ((ar_bit_60 == 1) ? ar_got_243 : ar_x0_243);
                    _er_f[60] = ((ar_bit_60 == 1) ? ar_x1_243 : ar_got_243);
                    int ar_bit_61 = lane >> 1 & 1;
                    float ar_x0_244 = _er_f[32];
                    float ar_x1_244 = _er_f[40];
                    float ar_send_244 = ((ar_bit_61 == 1) ? ar_x0_244 : ar_x1_244);
                    float ar_got_244 = __shfl_xor_sync(0xFFFFFFFF, ar_send_244, 2);
                    _er_f[32] = ((ar_bit_61 == 1) ? ar_got_244 : ar_x0_244);
                    _er_f[40] = ((ar_bit_61 == 1) ? ar_x1_244 : ar_got_244);
                    float ar_x0_245 = _er_f[36];
                    float ar_x1_245 = _er_f[44];
                    float ar_send_245 = ((ar_bit_61 == 1) ? ar_x0_245 : ar_x1_245);
                    float ar_got_245 = __shfl_xor_sync(0xFFFFFFFF, ar_send_245, 2);
                    _er_f[36] = ((ar_bit_61 == 1) ? ar_got_245 : ar_x0_245);
                    _er_f[44] = ((ar_bit_61 == 1) ? ar_x1_245 : ar_got_245);
                    float ar_x0_246 = _er_f[48];
                    float ar_x1_246 = _er_f[56];
                    float ar_send_246 = ((ar_bit_61 == 1) ? ar_x0_246 : ar_x1_246);
                    float ar_got_246 = __shfl_xor_sync(0xFFFFFFFF, ar_send_246, 2);
                    _er_f[48] = ((ar_bit_61 == 1) ? ar_got_246 : ar_x0_246);
                    _er_f[56] = ((ar_bit_61 == 1) ? ar_x1_246 : ar_got_246);
                    float ar_x0_247 = _er_f[52];
                    float ar_x1_247 = _er_f[60];
                    float ar_send_247 = ((ar_bit_61 == 1) ? ar_x0_247 : ar_x1_247);
                    float ar_got_247 = __shfl_xor_sync(0xFFFFFFFF, ar_send_247, 2);
                    _er_f[52] = ((ar_bit_61 == 1) ? ar_got_247 : ar_x0_247);
                    _er_f[60] = ((ar_bit_61 == 1) ? ar_x1_247 : ar_got_247);
                    int ar_bit_62 = lane & 1;
                    float ar_x0_248 = _er_f[32];
                    float ar_x1_248 = _er_f[36];
                    float ar_send_248 = ((ar_bit_62 == 1) ? ar_x0_248 : ar_x1_248);
                    float ar_got_248 = __shfl_xor_sync(0xFFFFFFFF, ar_send_248, 1);
                    _er_f[32] = ((ar_bit_62 == 1) ? ar_got_248 : ar_x0_248);
                    _er_f[36] = ((ar_bit_62 == 1) ? ar_x1_248 : ar_got_248);
                    float ar_x0_249 = _er_f[40];
                    float ar_x1_249 = _er_f[44];
                    float ar_send_249 = ((ar_bit_62 == 1) ? ar_x0_249 : ar_x1_249);
                    float ar_got_249 = __shfl_xor_sync(0xFFFFFFFF, ar_send_249, 1);
                    _er_f[40] = ((ar_bit_62 == 1) ? ar_got_249 : ar_x0_249);
                    _er_f[44] = ((ar_bit_62 == 1) ? ar_x1_249 : ar_got_249);
                    float ar_x0_250 = _er_f[48];
                    float ar_x1_250 = _er_f[52];
                    float ar_send_250 = ((ar_bit_62 == 1) ? ar_x0_250 : ar_x1_250);
                    float ar_got_250 = __shfl_xor_sync(0xFFFFFFFF, ar_send_250, 1);
                    _er_f[48] = ((ar_bit_62 == 1) ? ar_got_250 : ar_x0_250);
                    _er_f[52] = ((ar_bit_62 == 1) ? ar_x1_250 : ar_got_250);
                    float ar_x0_251 = _er_f[56];
                    float ar_x1_251 = _er_f[60];
                    float ar_send_251 = ((ar_bit_62 == 1) ? ar_x0_251 : ar_x1_251);
                    float ar_got_251 = __shfl_xor_sync(0xFFFFFFFF, ar_send_251, 1);
                    _er_f[56] = ((ar_bit_62 == 1) ? ar_got_251 : ar_x0_251);
                    _er_f[60] = ((ar_bit_62 == 1) ? ar_x1_251 : ar_got_251);
                    int ar_bit_63 = lane >> 2 & 1;
                    float ar_x0_252 = _er_f[33];
                    float ar_x1_252 = _er_f[49];
                    float ar_send_252 = ((ar_bit_63 == 1) ? ar_x0_252 : ar_x1_252);
                    float ar_got_252 = __shfl_xor_sync(0xFFFFFFFF, ar_send_252, 4);
                    _er_f[33] = ((ar_bit_63 == 1) ? ar_got_252 : ar_x0_252);
                    _er_f[49] = ((ar_bit_63 == 1) ? ar_x1_252 : ar_got_252);
                    float ar_x0_253 = _er_f[37];
                    float ar_x1_253 = _er_f[53];
                    float ar_send_253 = ((ar_bit_63 == 1) ? ar_x0_253 : ar_x1_253);
                    float ar_got_253 = __shfl_xor_sync(0xFFFFFFFF, ar_send_253, 4);
                    _er_f[37] = ((ar_bit_63 == 1) ? ar_got_253 : ar_x0_253);
                    _er_f[53] = ((ar_bit_63 == 1) ? ar_x1_253 : ar_got_253);
                    float ar_x0_254 = _er_f[41];
                    float ar_x1_254 = _er_f[57];
                    float ar_send_254 = ((ar_bit_63 == 1) ? ar_x0_254 : ar_x1_254);
                    float ar_got_254 = __shfl_xor_sync(0xFFFFFFFF, ar_send_254, 4);
                    _er_f[41] = ((ar_bit_63 == 1) ? ar_got_254 : ar_x0_254);
                    _er_f[57] = ((ar_bit_63 == 1) ? ar_x1_254 : ar_got_254);
                    float ar_x0_255 = _er_f[45];
                    float ar_x1_255 = _er_f[61];
                    float ar_send_255 = ((ar_bit_63 == 1) ? ar_x0_255 : ar_x1_255);
                    float ar_got_255 = __shfl_xor_sync(0xFFFFFFFF, ar_send_255, 4);
                    _er_f[45] = ((ar_bit_63 == 1) ? ar_got_255 : ar_x0_255);
                    _er_f[61] = ((ar_bit_63 == 1) ? ar_x1_255 : ar_got_255);
                    int ar_bit_64 = lane >> 1 & 1;
                    float ar_x0_256 = _er_f[33];
                    float ar_x1_256 = _er_f[41];
                    float ar_send_256 = ((ar_bit_64 == 1) ? ar_x0_256 : ar_x1_256);
                    float ar_got_256 = __shfl_xor_sync(0xFFFFFFFF, ar_send_256, 2);
                    _er_f[33] = ((ar_bit_64 == 1) ? ar_got_256 : ar_x0_256);
                    _er_f[41] = ((ar_bit_64 == 1) ? ar_x1_256 : ar_got_256);
                    float ar_x0_257 = _er_f[37];
                    float ar_x1_257 = _er_f[45];
                    float ar_send_257 = ((ar_bit_64 == 1) ? ar_x0_257 : ar_x1_257);
                    float ar_got_257 = __shfl_xor_sync(0xFFFFFFFF, ar_send_257, 2);
                    _er_f[37] = ((ar_bit_64 == 1) ? ar_got_257 : ar_x0_257);
                    _er_f[45] = ((ar_bit_64 == 1) ? ar_x1_257 : ar_got_257);
                    float ar_x0_258 = _er_f[49];
                    float ar_x1_258 = _er_f[57];
                    float ar_send_258 = ((ar_bit_64 == 1) ? ar_x0_258 : ar_x1_258);
                    float ar_got_258 = __shfl_xor_sync(0xFFFFFFFF, ar_send_258, 2);
                    _er_f[49] = ((ar_bit_64 == 1) ? ar_got_258 : ar_x0_258);
                    _er_f[57] = ((ar_bit_64 == 1) ? ar_x1_258 : ar_got_258);
                    float ar_x0_259 = _er_f[53];
                    float ar_x1_259 = _er_f[61];
                    float ar_send_259 = ((ar_bit_64 == 1) ? ar_x0_259 : ar_x1_259);
                    float ar_got_259 = __shfl_xor_sync(0xFFFFFFFF, ar_send_259, 2);
                    _er_f[53] = ((ar_bit_64 == 1) ? ar_got_259 : ar_x0_259);
                    _er_f[61] = ((ar_bit_64 == 1) ? ar_x1_259 : ar_got_259);
                    int ar_bit_65 = lane & 1;
                    float ar_x0_260 = _er_f[33];
                    float ar_x1_260 = _er_f[37];
                    float ar_send_260 = ((ar_bit_65 == 1) ? ar_x0_260 : ar_x1_260);
                    float ar_got_260 = __shfl_xor_sync(0xFFFFFFFF, ar_send_260, 1);
                    _er_f[33] = ((ar_bit_65 == 1) ? ar_got_260 : ar_x0_260);
                    _er_f[37] = ((ar_bit_65 == 1) ? ar_x1_260 : ar_got_260);
                    float ar_x0_261 = _er_f[41];
                    float ar_x1_261 = _er_f[45];
                    float ar_send_261 = ((ar_bit_65 == 1) ? ar_x0_261 : ar_x1_261);
                    float ar_got_261 = __shfl_xor_sync(0xFFFFFFFF, ar_send_261, 1);
                    _er_f[41] = ((ar_bit_65 == 1) ? ar_got_261 : ar_x0_261);
                    _er_f[45] = ((ar_bit_65 == 1) ? ar_x1_261 : ar_got_261);
                    float ar_x0_262 = _er_f[49];
                    float ar_x1_262 = _er_f[53];
                    float ar_send_262 = ((ar_bit_65 == 1) ? ar_x0_262 : ar_x1_262);
                    float ar_got_262 = __shfl_xor_sync(0xFFFFFFFF, ar_send_262, 1);
                    _er_f[49] = ((ar_bit_65 == 1) ? ar_got_262 : ar_x0_262);
                    _er_f[53] = ((ar_bit_65 == 1) ? ar_x1_262 : ar_got_262);
                    float ar_x0_263 = _er_f[57];
                    float ar_x1_263 = _er_f[61];
                    float ar_send_263 = ((ar_bit_65 == 1) ? ar_x0_263 : ar_x1_263);
                    float ar_got_263 = __shfl_xor_sync(0xFFFFFFFF, ar_send_263, 1);
                    _er_f[57] = ((ar_bit_65 == 1) ? ar_got_263 : ar_x0_263);
                    _er_f[61] = ((ar_bit_65 == 1) ? ar_x1_263 : ar_got_263);
                    int ar_bit_66 = lane >> 2 & 1;
                    float ar_x0_264 = _er_f[34];
                    float ar_x1_264 = _er_f[50];
                    float ar_send_264 = ((ar_bit_66 == 1) ? ar_x0_264 : ar_x1_264);
                    float ar_got_264 = __shfl_xor_sync(0xFFFFFFFF, ar_send_264, 4);
                    _er_f[34] = ((ar_bit_66 == 1) ? ar_got_264 : ar_x0_264);
                    _er_f[50] = ((ar_bit_66 == 1) ? ar_x1_264 : ar_got_264);
                    float ar_x0_265 = _er_f[38];
                    float ar_x1_265 = _er_f[54];
                    float ar_send_265 = ((ar_bit_66 == 1) ? ar_x0_265 : ar_x1_265);
                    float ar_got_265 = __shfl_xor_sync(0xFFFFFFFF, ar_send_265, 4);
                    _er_f[38] = ((ar_bit_66 == 1) ? ar_got_265 : ar_x0_265);
                    _er_f[54] = ((ar_bit_66 == 1) ? ar_x1_265 : ar_got_265);
                    float ar_x0_266 = _er_f[42];
                    float ar_x1_266 = _er_f[58];
                    float ar_send_266 = ((ar_bit_66 == 1) ? ar_x0_266 : ar_x1_266);
                    float ar_got_266 = __shfl_xor_sync(0xFFFFFFFF, ar_send_266, 4);
                    _er_f[42] = ((ar_bit_66 == 1) ? ar_got_266 : ar_x0_266);
                    _er_f[58] = ((ar_bit_66 == 1) ? ar_x1_266 : ar_got_266);
                    float ar_x0_267 = _er_f[46];
                    float ar_x1_267 = _er_f[62];
                    float ar_send_267 = ((ar_bit_66 == 1) ? ar_x0_267 : ar_x1_267);
                    float ar_got_267 = __shfl_xor_sync(0xFFFFFFFF, ar_send_267, 4);
                    _er_f[46] = ((ar_bit_66 == 1) ? ar_got_267 : ar_x0_267);
                    _er_f[62] = ((ar_bit_66 == 1) ? ar_x1_267 : ar_got_267);
                    int ar_bit_67 = lane >> 1 & 1;
                    float ar_x0_268 = _er_f[34];
                    float ar_x1_268 = _er_f[42];
                    float ar_send_268 = ((ar_bit_67 == 1) ? ar_x0_268 : ar_x1_268);
                    float ar_got_268 = __shfl_xor_sync(0xFFFFFFFF, ar_send_268, 2);
                    _er_f[34] = ((ar_bit_67 == 1) ? ar_got_268 : ar_x0_268);
                    _er_f[42] = ((ar_bit_67 == 1) ? ar_x1_268 : ar_got_268);
                    float ar_x0_269 = _er_f[38];
                    float ar_x1_269 = _er_f[46];
                    float ar_send_269 = ((ar_bit_67 == 1) ? ar_x0_269 : ar_x1_269);
                    float ar_got_269 = __shfl_xor_sync(0xFFFFFFFF, ar_send_269, 2);
                    _er_f[38] = ((ar_bit_67 == 1) ? ar_got_269 : ar_x0_269);
                    _er_f[46] = ((ar_bit_67 == 1) ? ar_x1_269 : ar_got_269);
                    float ar_x0_270 = _er_f[50];
                    float ar_x1_270 = _er_f[58];
                    float ar_send_270 = ((ar_bit_67 == 1) ? ar_x0_270 : ar_x1_270);
                    float ar_got_270 = __shfl_xor_sync(0xFFFFFFFF, ar_send_270, 2);
                    _er_f[50] = ((ar_bit_67 == 1) ? ar_got_270 : ar_x0_270);
                    _er_f[58] = ((ar_bit_67 == 1) ? ar_x1_270 : ar_got_270);
                    float ar_x0_271 = _er_f[54];
                    float ar_x1_271 = _er_f[62];
                    float ar_send_271 = ((ar_bit_67 == 1) ? ar_x0_271 : ar_x1_271);
                    float ar_got_271 = __shfl_xor_sync(0xFFFFFFFF, ar_send_271, 2);
                    _er_f[54] = ((ar_bit_67 == 1) ? ar_got_271 : ar_x0_271);
                    _er_f[62] = ((ar_bit_67 == 1) ? ar_x1_271 : ar_got_271);
                    int ar_bit_68 = lane & 1;
                    float ar_x0_272 = _er_f[34];
                    float ar_x1_272 = _er_f[38];
                    float ar_send_272 = ((ar_bit_68 == 1) ? ar_x0_272 : ar_x1_272);
                    float ar_got_272 = __shfl_xor_sync(0xFFFFFFFF, ar_send_272, 1);
                    _er_f[34] = ((ar_bit_68 == 1) ? ar_got_272 : ar_x0_272);
                    _er_f[38] = ((ar_bit_68 == 1) ? ar_x1_272 : ar_got_272);
                    float ar_x0_273 = _er_f[42];
                    float ar_x1_273 = _er_f[46];
                    float ar_send_273 = ((ar_bit_68 == 1) ? ar_x0_273 : ar_x1_273);
                    float ar_got_273 = __shfl_xor_sync(0xFFFFFFFF, ar_send_273, 1);
                    _er_f[42] = ((ar_bit_68 == 1) ? ar_got_273 : ar_x0_273);
                    _er_f[46] = ((ar_bit_68 == 1) ? ar_x1_273 : ar_got_273);
                    float ar_x0_274 = _er_f[50];
                    float ar_x1_274 = _er_f[54];
                    float ar_send_274 = ((ar_bit_68 == 1) ? ar_x0_274 : ar_x1_274);
                    float ar_got_274 = __shfl_xor_sync(0xFFFFFFFF, ar_send_274, 1);
                    _er_f[50] = ((ar_bit_68 == 1) ? ar_got_274 : ar_x0_274);
                    _er_f[54] = ((ar_bit_68 == 1) ? ar_x1_274 : ar_got_274);
                    float ar_x0_275 = _er_f[58];
                    float ar_x1_275 = _er_f[62];
                    float ar_send_275 = ((ar_bit_68 == 1) ? ar_x0_275 : ar_x1_275);
                    float ar_got_275 = __shfl_xor_sync(0xFFFFFFFF, ar_send_275, 1);
                    _er_f[58] = ((ar_bit_68 == 1) ? ar_got_275 : ar_x0_275);
                    _er_f[62] = ((ar_bit_68 == 1) ? ar_x1_275 : ar_got_275);
                    int ar_bit_69 = lane >> 2 & 1;
                    float ar_x0_276 = _er_f[35];
                    float ar_x1_276 = _er_f[51];
                    float ar_send_276 = ((ar_bit_69 == 1) ? ar_x0_276 : ar_x1_276);
                    float ar_got_276 = __shfl_xor_sync(0xFFFFFFFF, ar_send_276, 4);
                    _er_f[35] = ((ar_bit_69 == 1) ? ar_got_276 : ar_x0_276);
                    _er_f[51] = ((ar_bit_69 == 1) ? ar_x1_276 : ar_got_276);
                    float ar_x0_277 = _er_f[39];
                    float ar_x1_277 = _er_f[55];
                    float ar_send_277 = ((ar_bit_69 == 1) ? ar_x0_277 : ar_x1_277);
                    float ar_got_277 = __shfl_xor_sync(0xFFFFFFFF, ar_send_277, 4);
                    _er_f[39] = ((ar_bit_69 == 1) ? ar_got_277 : ar_x0_277);
                    _er_f[55] = ((ar_bit_69 == 1) ? ar_x1_277 : ar_got_277);
                    float ar_x0_278 = _er_f[43];
                    float ar_x1_278 = _er_f[59];
                    float ar_send_278 = ((ar_bit_69 == 1) ? ar_x0_278 : ar_x1_278);
                    float ar_got_278 = __shfl_xor_sync(0xFFFFFFFF, ar_send_278, 4);
                    _er_f[43] = ((ar_bit_69 == 1) ? ar_got_278 : ar_x0_278);
                    _er_f[59] = ((ar_bit_69 == 1) ? ar_x1_278 : ar_got_278);
                    float ar_x0_279 = _er_f[47];
                    float ar_x1_279 = _er_f[63];
                    float ar_send_279 = ((ar_bit_69 == 1) ? ar_x0_279 : ar_x1_279);
                    float ar_got_279 = __shfl_xor_sync(0xFFFFFFFF, ar_send_279, 4);
                    _er_f[47] = ((ar_bit_69 == 1) ? ar_got_279 : ar_x0_279);
                    _er_f[63] = ((ar_bit_69 == 1) ? ar_x1_279 : ar_got_279);
                    int ar_bit_70 = lane >> 1 & 1;
                    float ar_x0_280 = _er_f[35];
                    float ar_x1_280 = _er_f[43];
                    float ar_send_280 = ((ar_bit_70 == 1) ? ar_x0_280 : ar_x1_280);
                    float ar_got_280 = __shfl_xor_sync(0xFFFFFFFF, ar_send_280, 2);
                    _er_f[35] = ((ar_bit_70 == 1) ? ar_got_280 : ar_x0_280);
                    _er_f[43] = ((ar_bit_70 == 1) ? ar_x1_280 : ar_got_280);
                    float ar_x0_281 = _er_f[39];
                    float ar_x1_281 = _er_f[47];
                    float ar_send_281 = ((ar_bit_70 == 1) ? ar_x0_281 : ar_x1_281);
                    float ar_got_281 = __shfl_xor_sync(0xFFFFFFFF, ar_send_281, 2);
                    _er_f[39] = ((ar_bit_70 == 1) ? ar_got_281 : ar_x0_281);
                    _er_f[47] = ((ar_bit_70 == 1) ? ar_x1_281 : ar_got_281);
                    float ar_x0_282 = _er_f[51];
                    float ar_x1_282 = _er_f[59];
                    float ar_send_282 = ((ar_bit_70 == 1) ? ar_x0_282 : ar_x1_282);
                    float ar_got_282 = __shfl_xor_sync(0xFFFFFFFF, ar_send_282, 2);
                    _er_f[51] = ((ar_bit_70 == 1) ? ar_got_282 : ar_x0_282);
                    _er_f[59] = ((ar_bit_70 == 1) ? ar_x1_282 : ar_got_282);
                    float ar_x0_283 = _er_f[55];
                    float ar_x1_283 = _er_f[63];
                    float ar_send_283 = ((ar_bit_70 == 1) ? ar_x0_283 : ar_x1_283);
                    float ar_got_283 = __shfl_xor_sync(0xFFFFFFFF, ar_send_283, 2);
                    _er_f[55] = ((ar_bit_70 == 1) ? ar_got_283 : ar_x0_283);
                    _er_f[63] = ((ar_bit_70 == 1) ? ar_x1_283 : ar_got_283);
                    int ar_bit_71 = lane & 1;
                    float ar_x0_284 = _er_f[35];
                    float ar_x1_284 = _er_f[39];
                    float ar_send_284 = ((ar_bit_71 == 1) ? ar_x0_284 : ar_x1_284);
                    float ar_got_284 = __shfl_xor_sync(0xFFFFFFFF, ar_send_284, 1);
                    _er_f[35] = ((ar_bit_71 == 1) ? ar_got_284 : ar_x0_284);
                    _er_f[39] = ((ar_bit_71 == 1) ? ar_x1_284 : ar_got_284);
                    float ar_x0_285 = _er_f[43];
                    float ar_x1_285 = _er_f[47];
                    float ar_send_285 = ((ar_bit_71 == 1) ? ar_x0_285 : ar_x1_285);
                    float ar_got_285 = __shfl_xor_sync(0xFFFFFFFF, ar_send_285, 1);
                    _er_f[43] = ((ar_bit_71 == 1) ? ar_got_285 : ar_x0_285);
                    _er_f[47] = ((ar_bit_71 == 1) ? ar_x1_285 : ar_got_285);
                    float ar_x0_286 = _er_f[51];
                    float ar_x1_286 = _er_f[55];
                    float ar_send_286 = ((ar_bit_71 == 1) ? ar_x0_286 : ar_x1_286);
                    float ar_got_286 = __shfl_xor_sync(0xFFFFFFFF, ar_send_286, 1);
                    _er_f[51] = ((ar_bit_71 == 1) ? ar_got_286 : ar_x0_286);
                    _er_f[55] = ((ar_bit_71 == 1) ? ar_x1_286 : ar_got_286);
                    float ar_x0_287 = _er_f[59];
                    float ar_x1_287 = _er_f[63];
                    float ar_send_287 = ((ar_bit_71 == 1) ? ar_x0_287 : ar_x1_287);
                    float ar_got_287 = __shfl_xor_sync(0xFFFFFFFF, ar_send_287, 1);
                    _er_f[59] = ((ar_bit_71 == 1) ? ar_got_287 : ar_x0_287);
                    _er_f[63] = ((ar_bit_71 == 1) ? ar_x1_287 : ar_got_287);
                    int ar_ri_5 = er_at2 + 256 + 32;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_5])), "f"(_er_f[32]), "f"(_er_f[33]), "f"(_er_f[34]), "f"(_er_f[35]) : "memory");
                    ar_ri_5 = ar_ri_5 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_5])), "f"(_er_f[36]), "f"(_er_f[37]), "f"(_er_f[38]), "f"(_er_f[39]) : "memory");
                    ar_ri_5 = ar_ri_5 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_5])), "f"(_er_f[40]), "f"(_er_f[41]), "f"(_er_f[42]), "f"(_er_f[43]) : "memory");
                    ar_ri_5 = ar_ri_5 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_5])), "f"(_er_f[44]), "f"(_er_f[45]), "f"(_er_f[46]), "f"(_er_f[47]) : "memory");
                    ar_ri_5 = ar_ri_5 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_5])), "f"(_er_f[48]), "f"(_er_f[49]), "f"(_er_f[50]), "f"(_er_f[51]) : "memory");
                    ar_ri_5 = ar_ri_5 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_5])), "f"(_er_f[52]), "f"(_er_f[53]), "f"(_er_f[54]), "f"(_er_f[55]) : "memory");
                    ar_ri_5 = ar_ri_5 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_5])), "f"(_er_f[56]), "f"(_er_f[57]), "f"(_er_f[58]), "f"(_er_f[59]) : "memory");
                    ar_ri_5 = ar_ri_5 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_5])), "f"(_er_f[60]), "f"(_er_f[61]), "f"(_er_f[62]), "f"(_er_f[63]) : "memory");
                    int ar_bit_72 = lane >> 2 & 1;
                    float ar_x0_288 = _er_f[64];
                    float ar_x1_288 = _er_f[80];
                    float ar_send_288 = ((ar_bit_72 == 1) ? ar_x0_288 : ar_x1_288);
                    float ar_got_288 = __shfl_xor_sync(0xFFFFFFFF, ar_send_288, 4);
                    _er_f[64] = ((ar_bit_72 == 1) ? ar_got_288 : ar_x0_288);
                    _er_f[80] = ((ar_bit_72 == 1) ? ar_x1_288 : ar_got_288);
                    float ar_x0_289 = _er_f[68];
                    float ar_x1_289 = _er_f[84];
                    float ar_send_289 = ((ar_bit_72 == 1) ? ar_x0_289 : ar_x1_289);
                    float ar_got_289 = __shfl_xor_sync(0xFFFFFFFF, ar_send_289, 4);
                    _er_f[68] = ((ar_bit_72 == 1) ? ar_got_289 : ar_x0_289);
                    _er_f[84] = ((ar_bit_72 == 1) ? ar_x1_289 : ar_got_289);
                    float ar_x0_290 = _er_f[72];
                    float ar_x1_290 = _er_f[88];
                    float ar_send_290 = ((ar_bit_72 == 1) ? ar_x0_290 : ar_x1_290);
                    float ar_got_290 = __shfl_xor_sync(0xFFFFFFFF, ar_send_290, 4);
                    _er_f[72] = ((ar_bit_72 == 1) ? ar_got_290 : ar_x0_290);
                    _er_f[88] = ((ar_bit_72 == 1) ? ar_x1_290 : ar_got_290);
                    float ar_x0_291 = _er_f[76];
                    float ar_x1_291 = _er_f[92];
                    float ar_send_291 = ((ar_bit_72 == 1) ? ar_x0_291 : ar_x1_291);
                    float ar_got_291 = __shfl_xor_sync(0xFFFFFFFF, ar_send_291, 4);
                    _er_f[76] = ((ar_bit_72 == 1) ? ar_got_291 : ar_x0_291);
                    _er_f[92] = ((ar_bit_72 == 1) ? ar_x1_291 : ar_got_291);
                    int ar_bit_73 = lane >> 1 & 1;
                    float ar_x0_292 = _er_f[64];
                    float ar_x1_292 = _er_f[72];
                    float ar_send_292 = ((ar_bit_73 == 1) ? ar_x0_292 : ar_x1_292);
                    float ar_got_292 = __shfl_xor_sync(0xFFFFFFFF, ar_send_292, 2);
                    _er_f[64] = ((ar_bit_73 == 1) ? ar_got_292 : ar_x0_292);
                    _er_f[72] = ((ar_bit_73 == 1) ? ar_x1_292 : ar_got_292);
                    float ar_x0_293 = _er_f[68];
                    float ar_x1_293 = _er_f[76];
                    float ar_send_293 = ((ar_bit_73 == 1) ? ar_x0_293 : ar_x1_293);
                    float ar_got_293 = __shfl_xor_sync(0xFFFFFFFF, ar_send_293, 2);
                    _er_f[68] = ((ar_bit_73 == 1) ? ar_got_293 : ar_x0_293);
                    _er_f[76] = ((ar_bit_73 == 1) ? ar_x1_293 : ar_got_293);
                    float ar_x0_294 = _er_f[80];
                    float ar_x1_294 = _er_f[88];
                    float ar_send_294 = ((ar_bit_73 == 1) ? ar_x0_294 : ar_x1_294);
                    float ar_got_294 = __shfl_xor_sync(0xFFFFFFFF, ar_send_294, 2);
                    _er_f[80] = ((ar_bit_73 == 1) ? ar_got_294 : ar_x0_294);
                    _er_f[88] = ((ar_bit_73 == 1) ? ar_x1_294 : ar_got_294);
                    float ar_x0_295 = _er_f[84];
                    float ar_x1_295 = _er_f[92];
                    float ar_send_295 = ((ar_bit_73 == 1) ? ar_x0_295 : ar_x1_295);
                    float ar_got_295 = __shfl_xor_sync(0xFFFFFFFF, ar_send_295, 2);
                    _er_f[84] = ((ar_bit_73 == 1) ? ar_got_295 : ar_x0_295);
                    _er_f[92] = ((ar_bit_73 == 1) ? ar_x1_295 : ar_got_295);
                    int ar_bit_74 = lane & 1;
                    float ar_x0_296 = _er_f[64];
                    float ar_x1_296 = _er_f[68];
                    float ar_send_296 = ((ar_bit_74 == 1) ? ar_x0_296 : ar_x1_296);
                    float ar_got_296 = __shfl_xor_sync(0xFFFFFFFF, ar_send_296, 1);
                    _er_f[64] = ((ar_bit_74 == 1) ? ar_got_296 : ar_x0_296);
                    _er_f[68] = ((ar_bit_74 == 1) ? ar_x1_296 : ar_got_296);
                    float ar_x0_297 = _er_f[72];
                    float ar_x1_297 = _er_f[76];
                    float ar_send_297 = ((ar_bit_74 == 1) ? ar_x0_297 : ar_x1_297);
                    float ar_got_297 = __shfl_xor_sync(0xFFFFFFFF, ar_send_297, 1);
                    _er_f[72] = ((ar_bit_74 == 1) ? ar_got_297 : ar_x0_297);
                    _er_f[76] = ((ar_bit_74 == 1) ? ar_x1_297 : ar_got_297);
                    float ar_x0_298 = _er_f[80];
                    float ar_x1_298 = _er_f[84];
                    float ar_send_298 = ((ar_bit_74 == 1) ? ar_x0_298 : ar_x1_298);
                    float ar_got_298 = __shfl_xor_sync(0xFFFFFFFF, ar_send_298, 1);
                    _er_f[80] = ((ar_bit_74 == 1) ? ar_got_298 : ar_x0_298);
                    _er_f[84] = ((ar_bit_74 == 1) ? ar_x1_298 : ar_got_298);
                    float ar_x0_299 = _er_f[88];
                    float ar_x1_299 = _er_f[92];
                    float ar_send_299 = ((ar_bit_74 == 1) ? ar_x0_299 : ar_x1_299);
                    float ar_got_299 = __shfl_xor_sync(0xFFFFFFFF, ar_send_299, 1);
                    _er_f[88] = ((ar_bit_74 == 1) ? ar_got_299 : ar_x0_299);
                    _er_f[92] = ((ar_bit_74 == 1) ? ar_x1_299 : ar_got_299);
                    int ar_bit_75 = lane >> 2 & 1;
                    float ar_x0_300 = _er_f[65];
                    float ar_x1_300 = _er_f[81];
                    float ar_send_300 = ((ar_bit_75 == 1) ? ar_x0_300 : ar_x1_300);
                    float ar_got_300 = __shfl_xor_sync(0xFFFFFFFF, ar_send_300, 4);
                    _er_f[65] = ((ar_bit_75 == 1) ? ar_got_300 : ar_x0_300);
                    _er_f[81] = ((ar_bit_75 == 1) ? ar_x1_300 : ar_got_300);
                    float ar_x0_301 = _er_f[69];
                    float ar_x1_301 = _er_f[85];
                    float ar_send_301 = ((ar_bit_75 == 1) ? ar_x0_301 : ar_x1_301);
                    float ar_got_301 = __shfl_xor_sync(0xFFFFFFFF, ar_send_301, 4);
                    _er_f[69] = ((ar_bit_75 == 1) ? ar_got_301 : ar_x0_301);
                    _er_f[85] = ((ar_bit_75 == 1) ? ar_x1_301 : ar_got_301);
                    float ar_x0_302 = _er_f[73];
                    float ar_x1_302 = _er_f[89];
                    float ar_send_302 = ((ar_bit_75 == 1) ? ar_x0_302 : ar_x1_302);
                    float ar_got_302 = __shfl_xor_sync(0xFFFFFFFF, ar_send_302, 4);
                    _er_f[73] = ((ar_bit_75 == 1) ? ar_got_302 : ar_x0_302);
                    _er_f[89] = ((ar_bit_75 == 1) ? ar_x1_302 : ar_got_302);
                    float ar_x0_303 = _er_f[77];
                    float ar_x1_303 = _er_f[93];
                    float ar_send_303 = ((ar_bit_75 == 1) ? ar_x0_303 : ar_x1_303);
                    float ar_got_303 = __shfl_xor_sync(0xFFFFFFFF, ar_send_303, 4);
                    _er_f[77] = ((ar_bit_75 == 1) ? ar_got_303 : ar_x0_303);
                    _er_f[93] = ((ar_bit_75 == 1) ? ar_x1_303 : ar_got_303);
                    int ar_bit_76 = lane >> 1 & 1;
                    float ar_x0_304 = _er_f[65];
                    float ar_x1_304 = _er_f[73];
                    float ar_send_304 = ((ar_bit_76 == 1) ? ar_x0_304 : ar_x1_304);
                    float ar_got_304 = __shfl_xor_sync(0xFFFFFFFF, ar_send_304, 2);
                    _er_f[65] = ((ar_bit_76 == 1) ? ar_got_304 : ar_x0_304);
                    _er_f[73] = ((ar_bit_76 == 1) ? ar_x1_304 : ar_got_304);
                    float ar_x0_305 = _er_f[69];
                    float ar_x1_305 = _er_f[77];
                    float ar_send_305 = ((ar_bit_76 == 1) ? ar_x0_305 : ar_x1_305);
                    float ar_got_305 = __shfl_xor_sync(0xFFFFFFFF, ar_send_305, 2);
                    _er_f[69] = ((ar_bit_76 == 1) ? ar_got_305 : ar_x0_305);
                    _er_f[77] = ((ar_bit_76 == 1) ? ar_x1_305 : ar_got_305);
                    float ar_x0_306 = _er_f[81];
                    float ar_x1_306 = _er_f[89];
                    float ar_send_306 = ((ar_bit_76 == 1) ? ar_x0_306 : ar_x1_306);
                    float ar_got_306 = __shfl_xor_sync(0xFFFFFFFF, ar_send_306, 2);
                    _er_f[81] = ((ar_bit_76 == 1) ? ar_got_306 : ar_x0_306);
                    _er_f[89] = ((ar_bit_76 == 1) ? ar_x1_306 : ar_got_306);
                    float ar_x0_307 = _er_f[85];
                    float ar_x1_307 = _er_f[93];
                    float ar_send_307 = ((ar_bit_76 == 1) ? ar_x0_307 : ar_x1_307);
                    float ar_got_307 = __shfl_xor_sync(0xFFFFFFFF, ar_send_307, 2);
                    _er_f[85] = ((ar_bit_76 == 1) ? ar_got_307 : ar_x0_307);
                    _er_f[93] = ((ar_bit_76 == 1) ? ar_x1_307 : ar_got_307);
                    int ar_bit_77 = lane & 1;
                    float ar_x0_308 = _er_f[65];
                    float ar_x1_308 = _er_f[69];
                    float ar_send_308 = ((ar_bit_77 == 1) ? ar_x0_308 : ar_x1_308);
                    float ar_got_308 = __shfl_xor_sync(0xFFFFFFFF, ar_send_308, 1);
                    _er_f[65] = ((ar_bit_77 == 1) ? ar_got_308 : ar_x0_308);
                    _er_f[69] = ((ar_bit_77 == 1) ? ar_x1_308 : ar_got_308);
                    float ar_x0_309 = _er_f[73];
                    float ar_x1_309 = _er_f[77];
                    float ar_send_309 = ((ar_bit_77 == 1) ? ar_x0_309 : ar_x1_309);
                    float ar_got_309 = __shfl_xor_sync(0xFFFFFFFF, ar_send_309, 1);
                    _er_f[73] = ((ar_bit_77 == 1) ? ar_got_309 : ar_x0_309);
                    _er_f[77] = ((ar_bit_77 == 1) ? ar_x1_309 : ar_got_309);
                    float ar_x0_310 = _er_f[81];
                    float ar_x1_310 = _er_f[85];
                    float ar_send_310 = ((ar_bit_77 == 1) ? ar_x0_310 : ar_x1_310);
                    float ar_got_310 = __shfl_xor_sync(0xFFFFFFFF, ar_send_310, 1);
                    _er_f[81] = ((ar_bit_77 == 1) ? ar_got_310 : ar_x0_310);
                    _er_f[85] = ((ar_bit_77 == 1) ? ar_x1_310 : ar_got_310);
                    float ar_x0_311 = _er_f[89];
                    float ar_x1_311 = _er_f[93];
                    float ar_send_311 = ((ar_bit_77 == 1) ? ar_x0_311 : ar_x1_311);
                    float ar_got_311 = __shfl_xor_sync(0xFFFFFFFF, ar_send_311, 1);
                    _er_f[89] = ((ar_bit_77 == 1) ? ar_got_311 : ar_x0_311);
                    _er_f[93] = ((ar_bit_77 == 1) ? ar_x1_311 : ar_got_311);
                    int ar_bit_78 = lane >> 2 & 1;
                    float ar_x0_312 = _er_f[66];
                    float ar_x1_312 = _er_f[82];
                    float ar_send_312 = ((ar_bit_78 == 1) ? ar_x0_312 : ar_x1_312);
                    float ar_got_312 = __shfl_xor_sync(0xFFFFFFFF, ar_send_312, 4);
                    _er_f[66] = ((ar_bit_78 == 1) ? ar_got_312 : ar_x0_312);
                    _er_f[82] = ((ar_bit_78 == 1) ? ar_x1_312 : ar_got_312);
                    float ar_x0_313 = _er_f[70];
                    float ar_x1_313 = _er_f[86];
                    float ar_send_313 = ((ar_bit_78 == 1) ? ar_x0_313 : ar_x1_313);
                    float ar_got_313 = __shfl_xor_sync(0xFFFFFFFF, ar_send_313, 4);
                    _er_f[70] = ((ar_bit_78 == 1) ? ar_got_313 : ar_x0_313);
                    _er_f[86] = ((ar_bit_78 == 1) ? ar_x1_313 : ar_got_313);
                    float ar_x0_314 = _er_f[74];
                    float ar_x1_314 = _er_f[90];
                    float ar_send_314 = ((ar_bit_78 == 1) ? ar_x0_314 : ar_x1_314);
                    float ar_got_314 = __shfl_xor_sync(0xFFFFFFFF, ar_send_314, 4);
                    _er_f[74] = ((ar_bit_78 == 1) ? ar_got_314 : ar_x0_314);
                    _er_f[90] = ((ar_bit_78 == 1) ? ar_x1_314 : ar_got_314);
                    float ar_x0_315 = _er_f[78];
                    float ar_x1_315 = _er_f[94];
                    float ar_send_315 = ((ar_bit_78 == 1) ? ar_x0_315 : ar_x1_315);
                    float ar_got_315 = __shfl_xor_sync(0xFFFFFFFF, ar_send_315, 4);
                    _er_f[78] = ((ar_bit_78 == 1) ? ar_got_315 : ar_x0_315);
                    _er_f[94] = ((ar_bit_78 == 1) ? ar_x1_315 : ar_got_315);
                    int ar_bit_79 = lane >> 1 & 1;
                    float ar_x0_316 = _er_f[66];
                    float ar_x1_316 = _er_f[74];
                    float ar_send_316 = ((ar_bit_79 == 1) ? ar_x0_316 : ar_x1_316);
                    float ar_got_316 = __shfl_xor_sync(0xFFFFFFFF, ar_send_316, 2);
                    _er_f[66] = ((ar_bit_79 == 1) ? ar_got_316 : ar_x0_316);
                    _er_f[74] = ((ar_bit_79 == 1) ? ar_x1_316 : ar_got_316);
                    float ar_x0_317 = _er_f[70];
                    float ar_x1_317 = _er_f[78];
                    float ar_send_317 = ((ar_bit_79 == 1) ? ar_x0_317 : ar_x1_317);
                    float ar_got_317 = __shfl_xor_sync(0xFFFFFFFF, ar_send_317, 2);
                    _er_f[70] = ((ar_bit_79 == 1) ? ar_got_317 : ar_x0_317);
                    _er_f[78] = ((ar_bit_79 == 1) ? ar_x1_317 : ar_got_317);
                    float ar_x0_318 = _er_f[82];
                    float ar_x1_318 = _er_f[90];
                    float ar_send_318 = ((ar_bit_79 == 1) ? ar_x0_318 : ar_x1_318);
                    float ar_got_318 = __shfl_xor_sync(0xFFFFFFFF, ar_send_318, 2);
                    _er_f[82] = ((ar_bit_79 == 1) ? ar_got_318 : ar_x0_318);
                    _er_f[90] = ((ar_bit_79 == 1) ? ar_x1_318 : ar_got_318);
                    float ar_x0_319 = _er_f[86];
                    float ar_x1_319 = _er_f[94];
                    float ar_send_319 = ((ar_bit_79 == 1) ? ar_x0_319 : ar_x1_319);
                    float ar_got_319 = __shfl_xor_sync(0xFFFFFFFF, ar_send_319, 2);
                    _er_f[86] = ((ar_bit_79 == 1) ? ar_got_319 : ar_x0_319);
                    _er_f[94] = ((ar_bit_79 == 1) ? ar_x1_319 : ar_got_319);
                    int ar_bit_80 = lane & 1;
                    float ar_x0_320 = _er_f[66];
                    float ar_x1_320 = _er_f[70];
                    float ar_send_320 = ((ar_bit_80 == 1) ? ar_x0_320 : ar_x1_320);
                    float ar_got_320 = __shfl_xor_sync(0xFFFFFFFF, ar_send_320, 1);
                    _er_f[66] = ((ar_bit_80 == 1) ? ar_got_320 : ar_x0_320);
                    _er_f[70] = ((ar_bit_80 == 1) ? ar_x1_320 : ar_got_320);
                    float ar_x0_321 = _er_f[74];
                    float ar_x1_321 = _er_f[78];
                    float ar_send_321 = ((ar_bit_80 == 1) ? ar_x0_321 : ar_x1_321);
                    float ar_got_321 = __shfl_xor_sync(0xFFFFFFFF, ar_send_321, 1);
                    _er_f[74] = ((ar_bit_80 == 1) ? ar_got_321 : ar_x0_321);
                    _er_f[78] = ((ar_bit_80 == 1) ? ar_x1_321 : ar_got_321);
                    float ar_x0_322 = _er_f[82];
                    float ar_x1_322 = _er_f[86];
                    float ar_send_322 = ((ar_bit_80 == 1) ? ar_x0_322 : ar_x1_322);
                    float ar_got_322 = __shfl_xor_sync(0xFFFFFFFF, ar_send_322, 1);
                    _er_f[82] = ((ar_bit_80 == 1) ? ar_got_322 : ar_x0_322);
                    _er_f[86] = ((ar_bit_80 == 1) ? ar_x1_322 : ar_got_322);
                    float ar_x0_323 = _er_f[90];
                    float ar_x1_323 = _er_f[94];
                    float ar_send_323 = ((ar_bit_80 == 1) ? ar_x0_323 : ar_x1_323);
                    float ar_got_323 = __shfl_xor_sync(0xFFFFFFFF, ar_send_323, 1);
                    _er_f[90] = ((ar_bit_80 == 1) ? ar_got_323 : ar_x0_323);
                    _er_f[94] = ((ar_bit_80 == 1) ? ar_x1_323 : ar_got_323);
                    int ar_bit_81 = lane >> 2 & 1;
                    float ar_x0_324 = _er_f[67];
                    float ar_x1_324 = _er_f[83];
                    float ar_send_324 = ((ar_bit_81 == 1) ? ar_x0_324 : ar_x1_324);
                    float ar_got_324 = __shfl_xor_sync(0xFFFFFFFF, ar_send_324, 4);
                    _er_f[67] = ((ar_bit_81 == 1) ? ar_got_324 : ar_x0_324);
                    _er_f[83] = ((ar_bit_81 == 1) ? ar_x1_324 : ar_got_324);
                    float ar_x0_325 = _er_f[71];
                    float ar_x1_325 = _er_f[87];
                    float ar_send_325 = ((ar_bit_81 == 1) ? ar_x0_325 : ar_x1_325);
                    float ar_got_325 = __shfl_xor_sync(0xFFFFFFFF, ar_send_325, 4);
                    _er_f[71] = ((ar_bit_81 == 1) ? ar_got_325 : ar_x0_325);
                    _er_f[87] = ((ar_bit_81 == 1) ? ar_x1_325 : ar_got_325);
                    float ar_x0_326 = _er_f[75];
                    float ar_x1_326 = _er_f[91];
                    float ar_send_326 = ((ar_bit_81 == 1) ? ar_x0_326 : ar_x1_326);
                    float ar_got_326 = __shfl_xor_sync(0xFFFFFFFF, ar_send_326, 4);
                    _er_f[75] = ((ar_bit_81 == 1) ? ar_got_326 : ar_x0_326);
                    _er_f[91] = ((ar_bit_81 == 1) ? ar_x1_326 : ar_got_326);
                    float ar_x0_327 = _er_f[79];
                    float ar_x1_327 = _er_f[95];
                    float ar_send_327 = ((ar_bit_81 == 1) ? ar_x0_327 : ar_x1_327);
                    float ar_got_327 = __shfl_xor_sync(0xFFFFFFFF, ar_send_327, 4);
                    _er_f[79] = ((ar_bit_81 == 1) ? ar_got_327 : ar_x0_327);
                    _er_f[95] = ((ar_bit_81 == 1) ? ar_x1_327 : ar_got_327);
                    int ar_bit_82 = lane >> 1 & 1;
                    float ar_x0_328 = _er_f[67];
                    float ar_x1_328 = _er_f[75];
                    float ar_send_328 = ((ar_bit_82 == 1) ? ar_x0_328 : ar_x1_328);
                    float ar_got_328 = __shfl_xor_sync(0xFFFFFFFF, ar_send_328, 2);
                    _er_f[67] = ((ar_bit_82 == 1) ? ar_got_328 : ar_x0_328);
                    _er_f[75] = ((ar_bit_82 == 1) ? ar_x1_328 : ar_got_328);
                    float ar_x0_329 = _er_f[71];
                    float ar_x1_329 = _er_f[79];
                    float ar_send_329 = ((ar_bit_82 == 1) ? ar_x0_329 : ar_x1_329);
                    float ar_got_329 = __shfl_xor_sync(0xFFFFFFFF, ar_send_329, 2);
                    _er_f[71] = ((ar_bit_82 == 1) ? ar_got_329 : ar_x0_329);
                    _er_f[79] = ((ar_bit_82 == 1) ? ar_x1_329 : ar_got_329);
                    float ar_x0_330 = _er_f[83];
                    float ar_x1_330 = _er_f[91];
                    float ar_send_330 = ((ar_bit_82 == 1) ? ar_x0_330 : ar_x1_330);
                    float ar_got_330 = __shfl_xor_sync(0xFFFFFFFF, ar_send_330, 2);
                    _er_f[83] = ((ar_bit_82 == 1) ? ar_got_330 : ar_x0_330);
                    _er_f[91] = ((ar_bit_82 == 1) ? ar_x1_330 : ar_got_330);
                    float ar_x0_331 = _er_f[87];
                    float ar_x1_331 = _er_f[95];
                    float ar_send_331 = ((ar_bit_82 == 1) ? ar_x0_331 : ar_x1_331);
                    float ar_got_331 = __shfl_xor_sync(0xFFFFFFFF, ar_send_331, 2);
                    _er_f[87] = ((ar_bit_82 == 1) ? ar_got_331 : ar_x0_331);
                    _er_f[95] = ((ar_bit_82 == 1) ? ar_x1_331 : ar_got_331);
                    int ar_bit_83 = lane & 1;
                    float ar_x0_332 = _er_f[67];
                    float ar_x1_332 = _er_f[71];
                    float ar_send_332 = ((ar_bit_83 == 1) ? ar_x0_332 : ar_x1_332);
                    float ar_got_332 = __shfl_xor_sync(0xFFFFFFFF, ar_send_332, 1);
                    _er_f[67] = ((ar_bit_83 == 1) ? ar_got_332 : ar_x0_332);
                    _er_f[71] = ((ar_bit_83 == 1) ? ar_x1_332 : ar_got_332);
                    float ar_x0_333 = _er_f[75];
                    float ar_x1_333 = _er_f[79];
                    float ar_send_333 = ((ar_bit_83 == 1) ? ar_x0_333 : ar_x1_333);
                    float ar_got_333 = __shfl_xor_sync(0xFFFFFFFF, ar_send_333, 1);
                    _er_f[75] = ((ar_bit_83 == 1) ? ar_got_333 : ar_x0_333);
                    _er_f[79] = ((ar_bit_83 == 1) ? ar_x1_333 : ar_got_333);
                    float ar_x0_334 = _er_f[83];
                    float ar_x1_334 = _er_f[87];
                    float ar_send_334 = ((ar_bit_83 == 1) ? ar_x0_334 : ar_x1_334);
                    float ar_got_334 = __shfl_xor_sync(0xFFFFFFFF, ar_send_334, 1);
                    _er_f[83] = ((ar_bit_83 == 1) ? ar_got_334 : ar_x0_334);
                    _er_f[87] = ((ar_bit_83 == 1) ? ar_x1_334 : ar_got_334);
                    float ar_x0_335 = _er_f[91];
                    float ar_x1_335 = _er_f[95];
                    float ar_send_335 = ((ar_bit_83 == 1) ? ar_x0_335 : ar_x1_335);
                    float ar_got_335 = __shfl_xor_sync(0xFFFFFFFF, ar_send_335, 1);
                    _er_f[91] = ((ar_bit_83 == 1) ? ar_got_335 : ar_x0_335);
                    _er_f[95] = ((ar_bit_83 == 1) ? ar_x1_335 : ar_got_335);
                    int ar_ri_6 = er_at2 + 256 + 64;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_6])), "f"(_er_f[64]), "f"(_er_f[65]), "f"(_er_f[66]), "f"(_er_f[67]) : "memory");
                    ar_ri_6 = ar_ri_6 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_6])), "f"(_er_f[68]), "f"(_er_f[69]), "f"(_er_f[70]), "f"(_er_f[71]) : "memory");
                    ar_ri_6 = ar_ri_6 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_6])), "f"(_er_f[72]), "f"(_er_f[73]), "f"(_er_f[74]), "f"(_er_f[75]) : "memory");
                    ar_ri_6 = ar_ri_6 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_6])), "f"(_er_f[76]), "f"(_er_f[77]), "f"(_er_f[78]), "f"(_er_f[79]) : "memory");
                    ar_ri_6 = ar_ri_6 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_6])), "f"(_er_f[80]), "f"(_er_f[81]), "f"(_er_f[82]), "f"(_er_f[83]) : "memory");
                    ar_ri_6 = ar_ri_6 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_6])), "f"(_er_f[84]), "f"(_er_f[85]), "f"(_er_f[86]), "f"(_er_f[87]) : "memory");
                    ar_ri_6 = ar_ri_6 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_6])), "f"(_er_f[88]), "f"(_er_f[89]), "f"(_er_f[90]), "f"(_er_f[91]) : "memory");
                    ar_ri_6 = ar_ri_6 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_6])), "f"(_er_f[92]), "f"(_er_f[93]), "f"(_er_f[94]), "f"(_er_f[95]) : "memory");
                    int ar_bit_84 = lane >> 2 & 1;
                    float ar_x0_336 = _er_f[96];
                    float ar_x1_336 = _er_f[112];
                    float ar_send_336 = ((ar_bit_84 == 1) ? ar_x0_336 : ar_x1_336);
                    float ar_got_336 = __shfl_xor_sync(0xFFFFFFFF, ar_send_336, 4);
                    _er_f[96] = ((ar_bit_84 == 1) ? ar_got_336 : ar_x0_336);
                    _er_f[112] = ((ar_bit_84 == 1) ? ar_x1_336 : ar_got_336);
                    float ar_x0_337 = _er_f[100];
                    float ar_x1_337 = _er_f[116];
                    float ar_send_337 = ((ar_bit_84 == 1) ? ar_x0_337 : ar_x1_337);
                    float ar_got_337 = __shfl_xor_sync(0xFFFFFFFF, ar_send_337, 4);
                    _er_f[100] = ((ar_bit_84 == 1) ? ar_got_337 : ar_x0_337);
                    _er_f[116] = ((ar_bit_84 == 1) ? ar_x1_337 : ar_got_337);
                    float ar_x0_338 = _er_f[104];
                    float ar_x1_338 = _er_f[120];
                    float ar_send_338 = ((ar_bit_84 == 1) ? ar_x0_338 : ar_x1_338);
                    float ar_got_338 = __shfl_xor_sync(0xFFFFFFFF, ar_send_338, 4);
                    _er_f[104] = ((ar_bit_84 == 1) ? ar_got_338 : ar_x0_338);
                    _er_f[120] = ((ar_bit_84 == 1) ? ar_x1_338 : ar_got_338);
                    float ar_x0_339 = _er_f[108];
                    float ar_x1_339 = _er_f[124];
                    float ar_send_339 = ((ar_bit_84 == 1) ? ar_x0_339 : ar_x1_339);
                    float ar_got_339 = __shfl_xor_sync(0xFFFFFFFF, ar_send_339, 4);
                    _er_f[108] = ((ar_bit_84 == 1) ? ar_got_339 : ar_x0_339);
                    _er_f[124] = ((ar_bit_84 == 1) ? ar_x1_339 : ar_got_339);
                    int ar_bit_85 = lane >> 1 & 1;
                    float ar_x0_340 = _er_f[96];
                    float ar_x1_340 = _er_f[104];
                    float ar_send_340 = ((ar_bit_85 == 1) ? ar_x0_340 : ar_x1_340);
                    float ar_got_340 = __shfl_xor_sync(0xFFFFFFFF, ar_send_340, 2);
                    _er_f[96] = ((ar_bit_85 == 1) ? ar_got_340 : ar_x0_340);
                    _er_f[104] = ((ar_bit_85 == 1) ? ar_x1_340 : ar_got_340);
                    float ar_x0_341 = _er_f[100];
                    float ar_x1_341 = _er_f[108];
                    float ar_send_341 = ((ar_bit_85 == 1) ? ar_x0_341 : ar_x1_341);
                    float ar_got_341 = __shfl_xor_sync(0xFFFFFFFF, ar_send_341, 2);
                    _er_f[100] = ((ar_bit_85 == 1) ? ar_got_341 : ar_x0_341);
                    _er_f[108] = ((ar_bit_85 == 1) ? ar_x1_341 : ar_got_341);
                    float ar_x0_342 = _er_f[112];
                    float ar_x1_342 = _er_f[120];
                    float ar_send_342 = ((ar_bit_85 == 1) ? ar_x0_342 : ar_x1_342);
                    float ar_got_342 = __shfl_xor_sync(0xFFFFFFFF, ar_send_342, 2);
                    _er_f[112] = ((ar_bit_85 == 1) ? ar_got_342 : ar_x0_342);
                    _er_f[120] = ((ar_bit_85 == 1) ? ar_x1_342 : ar_got_342);
                    float ar_x0_343 = _er_f[116];
                    float ar_x1_343 = _er_f[124];
                    float ar_send_343 = ((ar_bit_85 == 1) ? ar_x0_343 : ar_x1_343);
                    float ar_got_343 = __shfl_xor_sync(0xFFFFFFFF, ar_send_343, 2);
                    _er_f[116] = ((ar_bit_85 == 1) ? ar_got_343 : ar_x0_343);
                    _er_f[124] = ((ar_bit_85 == 1) ? ar_x1_343 : ar_got_343);
                    int ar_bit_86 = lane & 1;
                    float ar_x0_344 = _er_f[96];
                    float ar_x1_344 = _er_f[100];
                    float ar_send_344 = ((ar_bit_86 == 1) ? ar_x0_344 : ar_x1_344);
                    float ar_got_344 = __shfl_xor_sync(0xFFFFFFFF, ar_send_344, 1);
                    _er_f[96] = ((ar_bit_86 == 1) ? ar_got_344 : ar_x0_344);
                    _er_f[100] = ((ar_bit_86 == 1) ? ar_x1_344 : ar_got_344);
                    float ar_x0_345 = _er_f[104];
                    float ar_x1_345 = _er_f[108];
                    float ar_send_345 = ((ar_bit_86 == 1) ? ar_x0_345 : ar_x1_345);
                    float ar_got_345 = __shfl_xor_sync(0xFFFFFFFF, ar_send_345, 1);
                    _er_f[104] = ((ar_bit_86 == 1) ? ar_got_345 : ar_x0_345);
                    _er_f[108] = ((ar_bit_86 == 1) ? ar_x1_345 : ar_got_345);
                    float ar_x0_346 = _er_f[112];
                    float ar_x1_346 = _er_f[116];
                    float ar_send_346 = ((ar_bit_86 == 1) ? ar_x0_346 : ar_x1_346);
                    float ar_got_346 = __shfl_xor_sync(0xFFFFFFFF, ar_send_346, 1);
                    _er_f[112] = ((ar_bit_86 == 1) ? ar_got_346 : ar_x0_346);
                    _er_f[116] = ((ar_bit_86 == 1) ? ar_x1_346 : ar_got_346);
                    float ar_x0_347 = _er_f[120];
                    float ar_x1_347 = _er_f[124];
                    float ar_send_347 = ((ar_bit_86 == 1) ? ar_x0_347 : ar_x1_347);
                    float ar_got_347 = __shfl_xor_sync(0xFFFFFFFF, ar_send_347, 1);
                    _er_f[120] = ((ar_bit_86 == 1) ? ar_got_347 : ar_x0_347);
                    _er_f[124] = ((ar_bit_86 == 1) ? ar_x1_347 : ar_got_347);
                    int ar_bit_87 = lane >> 2 & 1;
                    float ar_x0_348 = _er_f[97];
                    float ar_x1_348 = _er_f[113];
                    float ar_send_348 = ((ar_bit_87 == 1) ? ar_x0_348 : ar_x1_348);
                    float ar_got_348 = __shfl_xor_sync(0xFFFFFFFF, ar_send_348, 4);
                    _er_f[97] = ((ar_bit_87 == 1) ? ar_got_348 : ar_x0_348);
                    _er_f[113] = ((ar_bit_87 == 1) ? ar_x1_348 : ar_got_348);
                    float ar_x0_349 = _er_f[101];
                    float ar_x1_349 = _er_f[117];
                    float ar_send_349 = ((ar_bit_87 == 1) ? ar_x0_349 : ar_x1_349);
                    float ar_got_349 = __shfl_xor_sync(0xFFFFFFFF, ar_send_349, 4);
                    _er_f[101] = ((ar_bit_87 == 1) ? ar_got_349 : ar_x0_349);
                    _er_f[117] = ((ar_bit_87 == 1) ? ar_x1_349 : ar_got_349);
                    float ar_x0_350 = _er_f[105];
                    float ar_x1_350 = _er_f[121];
                    float ar_send_350 = ((ar_bit_87 == 1) ? ar_x0_350 : ar_x1_350);
                    float ar_got_350 = __shfl_xor_sync(0xFFFFFFFF, ar_send_350, 4);
                    _er_f[105] = ((ar_bit_87 == 1) ? ar_got_350 : ar_x0_350);
                    _er_f[121] = ((ar_bit_87 == 1) ? ar_x1_350 : ar_got_350);
                    float ar_x0_351 = _er_f[109];
                    float ar_x1_351 = _er_f[125];
                    float ar_send_351 = ((ar_bit_87 == 1) ? ar_x0_351 : ar_x1_351);
                    float ar_got_351 = __shfl_xor_sync(0xFFFFFFFF, ar_send_351, 4);
                    _er_f[109] = ((ar_bit_87 == 1) ? ar_got_351 : ar_x0_351);
                    _er_f[125] = ((ar_bit_87 == 1) ? ar_x1_351 : ar_got_351);
                    int ar_bit_88 = lane >> 1 & 1;
                    float ar_x0_352 = _er_f[97];
                    float ar_x1_352 = _er_f[105];
                    float ar_send_352 = ((ar_bit_88 == 1) ? ar_x0_352 : ar_x1_352);
                    float ar_got_352 = __shfl_xor_sync(0xFFFFFFFF, ar_send_352, 2);
                    _er_f[97] = ((ar_bit_88 == 1) ? ar_got_352 : ar_x0_352);
                    _er_f[105] = ((ar_bit_88 == 1) ? ar_x1_352 : ar_got_352);
                    float ar_x0_353 = _er_f[101];
                    float ar_x1_353 = _er_f[109];
                    float ar_send_353 = ((ar_bit_88 == 1) ? ar_x0_353 : ar_x1_353);
                    float ar_got_353 = __shfl_xor_sync(0xFFFFFFFF, ar_send_353, 2);
                    _er_f[101] = ((ar_bit_88 == 1) ? ar_got_353 : ar_x0_353);
                    _er_f[109] = ((ar_bit_88 == 1) ? ar_x1_353 : ar_got_353);
                    float ar_x0_354 = _er_f[113];
                    float ar_x1_354 = _er_f[121];
                    float ar_send_354 = ((ar_bit_88 == 1) ? ar_x0_354 : ar_x1_354);
                    float ar_got_354 = __shfl_xor_sync(0xFFFFFFFF, ar_send_354, 2);
                    _er_f[113] = ((ar_bit_88 == 1) ? ar_got_354 : ar_x0_354);
                    _er_f[121] = ((ar_bit_88 == 1) ? ar_x1_354 : ar_got_354);
                    float ar_x0_355 = _er_f[117];
                    float ar_x1_355 = _er_f[125];
                    float ar_send_355 = ((ar_bit_88 == 1) ? ar_x0_355 : ar_x1_355);
                    float ar_got_355 = __shfl_xor_sync(0xFFFFFFFF, ar_send_355, 2);
                    _er_f[117] = ((ar_bit_88 == 1) ? ar_got_355 : ar_x0_355);
                    _er_f[125] = ((ar_bit_88 == 1) ? ar_x1_355 : ar_got_355);
                    int ar_bit_89 = lane & 1;
                    float ar_x0_356 = _er_f[97];
                    float ar_x1_356 = _er_f[101];
                    float ar_send_356 = ((ar_bit_89 == 1) ? ar_x0_356 : ar_x1_356);
                    float ar_got_356 = __shfl_xor_sync(0xFFFFFFFF, ar_send_356, 1);
                    _er_f[97] = ((ar_bit_89 == 1) ? ar_got_356 : ar_x0_356);
                    _er_f[101] = ((ar_bit_89 == 1) ? ar_x1_356 : ar_got_356);
                    float ar_x0_357 = _er_f[105];
                    float ar_x1_357 = _er_f[109];
                    float ar_send_357 = ((ar_bit_89 == 1) ? ar_x0_357 : ar_x1_357);
                    float ar_got_357 = __shfl_xor_sync(0xFFFFFFFF, ar_send_357, 1);
                    _er_f[105] = ((ar_bit_89 == 1) ? ar_got_357 : ar_x0_357);
                    _er_f[109] = ((ar_bit_89 == 1) ? ar_x1_357 : ar_got_357);
                    float ar_x0_358 = _er_f[113];
                    float ar_x1_358 = _er_f[117];
                    float ar_send_358 = ((ar_bit_89 == 1) ? ar_x0_358 : ar_x1_358);
                    float ar_got_358 = __shfl_xor_sync(0xFFFFFFFF, ar_send_358, 1);
                    _er_f[113] = ((ar_bit_89 == 1) ? ar_got_358 : ar_x0_358);
                    _er_f[117] = ((ar_bit_89 == 1) ? ar_x1_358 : ar_got_358);
                    float ar_x0_359 = _er_f[121];
                    float ar_x1_359 = _er_f[125];
                    float ar_send_359 = ((ar_bit_89 == 1) ? ar_x0_359 : ar_x1_359);
                    float ar_got_359 = __shfl_xor_sync(0xFFFFFFFF, ar_send_359, 1);
                    _er_f[121] = ((ar_bit_89 == 1) ? ar_got_359 : ar_x0_359);
                    _er_f[125] = ((ar_bit_89 == 1) ? ar_x1_359 : ar_got_359);
                    int ar_bit_90 = lane >> 2 & 1;
                    float ar_x0_360 = _er_f[98];
                    float ar_x1_360 = _er_f[114];
                    float ar_send_360 = ((ar_bit_90 == 1) ? ar_x0_360 : ar_x1_360);
                    float ar_got_360 = __shfl_xor_sync(0xFFFFFFFF, ar_send_360, 4);
                    _er_f[98] = ((ar_bit_90 == 1) ? ar_got_360 : ar_x0_360);
                    _er_f[114] = ((ar_bit_90 == 1) ? ar_x1_360 : ar_got_360);
                    float ar_x0_361 = _er_f[102];
                    float ar_x1_361 = _er_f[118];
                    float ar_send_361 = ((ar_bit_90 == 1) ? ar_x0_361 : ar_x1_361);
                    float ar_got_361 = __shfl_xor_sync(0xFFFFFFFF, ar_send_361, 4);
                    _er_f[102] = ((ar_bit_90 == 1) ? ar_got_361 : ar_x0_361);
                    _er_f[118] = ((ar_bit_90 == 1) ? ar_x1_361 : ar_got_361);
                    float ar_x0_362 = _er_f[106];
                    float ar_x1_362 = _er_f[122];
                    float ar_send_362 = ((ar_bit_90 == 1) ? ar_x0_362 : ar_x1_362);
                    float ar_got_362 = __shfl_xor_sync(0xFFFFFFFF, ar_send_362, 4);
                    _er_f[106] = ((ar_bit_90 == 1) ? ar_got_362 : ar_x0_362);
                    _er_f[122] = ((ar_bit_90 == 1) ? ar_x1_362 : ar_got_362);
                    float ar_x0_363 = _er_f[110];
                    float ar_x1_363 = _er_f[126];
                    float ar_send_363 = ((ar_bit_90 == 1) ? ar_x0_363 : ar_x1_363);
                    float ar_got_363 = __shfl_xor_sync(0xFFFFFFFF, ar_send_363, 4);
                    _er_f[110] = ((ar_bit_90 == 1) ? ar_got_363 : ar_x0_363);
                    _er_f[126] = ((ar_bit_90 == 1) ? ar_x1_363 : ar_got_363);
                    int ar_bit_91 = lane >> 1 & 1;
                    float ar_x0_364 = _er_f[98];
                    float ar_x1_364 = _er_f[106];
                    float ar_send_364 = ((ar_bit_91 == 1) ? ar_x0_364 : ar_x1_364);
                    float ar_got_364 = __shfl_xor_sync(0xFFFFFFFF, ar_send_364, 2);
                    _er_f[98] = ((ar_bit_91 == 1) ? ar_got_364 : ar_x0_364);
                    _er_f[106] = ((ar_bit_91 == 1) ? ar_x1_364 : ar_got_364);
                    float ar_x0_365 = _er_f[102];
                    float ar_x1_365 = _er_f[110];
                    float ar_send_365 = ((ar_bit_91 == 1) ? ar_x0_365 : ar_x1_365);
                    float ar_got_365 = __shfl_xor_sync(0xFFFFFFFF, ar_send_365, 2);
                    _er_f[102] = ((ar_bit_91 == 1) ? ar_got_365 : ar_x0_365);
                    _er_f[110] = ((ar_bit_91 == 1) ? ar_x1_365 : ar_got_365);
                    float ar_x0_366 = _er_f[114];
                    float ar_x1_366 = _er_f[122];
                    float ar_send_366 = ((ar_bit_91 == 1) ? ar_x0_366 : ar_x1_366);
                    float ar_got_366 = __shfl_xor_sync(0xFFFFFFFF, ar_send_366, 2);
                    _er_f[114] = ((ar_bit_91 == 1) ? ar_got_366 : ar_x0_366);
                    _er_f[122] = ((ar_bit_91 == 1) ? ar_x1_366 : ar_got_366);
                    float ar_x0_367 = _er_f[118];
                    float ar_x1_367 = _er_f[126];
                    float ar_send_367 = ((ar_bit_91 == 1) ? ar_x0_367 : ar_x1_367);
                    float ar_got_367 = __shfl_xor_sync(0xFFFFFFFF, ar_send_367, 2);
                    _er_f[118] = ((ar_bit_91 == 1) ? ar_got_367 : ar_x0_367);
                    _er_f[126] = ((ar_bit_91 == 1) ? ar_x1_367 : ar_got_367);
                    int ar_bit_92 = lane & 1;
                    float ar_x0_368 = _er_f[98];
                    float ar_x1_368 = _er_f[102];
                    float ar_send_368 = ((ar_bit_92 == 1) ? ar_x0_368 : ar_x1_368);
                    float ar_got_368 = __shfl_xor_sync(0xFFFFFFFF, ar_send_368, 1);
                    _er_f[98] = ((ar_bit_92 == 1) ? ar_got_368 : ar_x0_368);
                    _er_f[102] = ((ar_bit_92 == 1) ? ar_x1_368 : ar_got_368);
                    float ar_x0_369 = _er_f[106];
                    float ar_x1_369 = _er_f[110];
                    float ar_send_369 = ((ar_bit_92 == 1) ? ar_x0_369 : ar_x1_369);
                    float ar_got_369 = __shfl_xor_sync(0xFFFFFFFF, ar_send_369, 1);
                    _er_f[106] = ((ar_bit_92 == 1) ? ar_got_369 : ar_x0_369);
                    _er_f[110] = ((ar_bit_92 == 1) ? ar_x1_369 : ar_got_369);
                    float ar_x0_370 = _er_f[114];
                    float ar_x1_370 = _er_f[118];
                    float ar_send_370 = ((ar_bit_92 == 1) ? ar_x0_370 : ar_x1_370);
                    float ar_got_370 = __shfl_xor_sync(0xFFFFFFFF, ar_send_370, 1);
                    _er_f[114] = ((ar_bit_92 == 1) ? ar_got_370 : ar_x0_370);
                    _er_f[118] = ((ar_bit_92 == 1) ? ar_x1_370 : ar_got_370);
                    float ar_x0_371 = _er_f[122];
                    float ar_x1_371 = _er_f[126];
                    float ar_send_371 = ((ar_bit_92 == 1) ? ar_x0_371 : ar_x1_371);
                    float ar_got_371 = __shfl_xor_sync(0xFFFFFFFF, ar_send_371, 1);
                    _er_f[122] = ((ar_bit_92 == 1) ? ar_got_371 : ar_x0_371);
                    _er_f[126] = ((ar_bit_92 == 1) ? ar_x1_371 : ar_got_371);
                    int ar_bit_93 = lane >> 2 & 1;
                    float ar_x0_372 = _er_f[99];
                    float ar_x1_372 = _er_f[115];
                    float ar_send_372 = ((ar_bit_93 == 1) ? ar_x0_372 : ar_x1_372);
                    float ar_got_372 = __shfl_xor_sync(0xFFFFFFFF, ar_send_372, 4);
                    _er_f[99] = ((ar_bit_93 == 1) ? ar_got_372 : ar_x0_372);
                    _er_f[115] = ((ar_bit_93 == 1) ? ar_x1_372 : ar_got_372);
                    float ar_x0_373 = _er_f[103];
                    float ar_x1_373 = _er_f[119];
                    float ar_send_373 = ((ar_bit_93 == 1) ? ar_x0_373 : ar_x1_373);
                    float ar_got_373 = __shfl_xor_sync(0xFFFFFFFF, ar_send_373, 4);
                    _er_f[103] = ((ar_bit_93 == 1) ? ar_got_373 : ar_x0_373);
                    _er_f[119] = ((ar_bit_93 == 1) ? ar_x1_373 : ar_got_373);
                    float ar_x0_374 = _er_f[107];
                    float ar_x1_374 = _er_f[123];
                    float ar_send_374 = ((ar_bit_93 == 1) ? ar_x0_374 : ar_x1_374);
                    float ar_got_374 = __shfl_xor_sync(0xFFFFFFFF, ar_send_374, 4);
                    _er_f[107] = ((ar_bit_93 == 1) ? ar_got_374 : ar_x0_374);
                    _er_f[123] = ((ar_bit_93 == 1) ? ar_x1_374 : ar_got_374);
                    float ar_x0_375 = _er_f[111];
                    float ar_x1_375 = _er_f[127];
                    float ar_send_375 = ((ar_bit_93 == 1) ? ar_x0_375 : ar_x1_375);
                    float ar_got_375 = __shfl_xor_sync(0xFFFFFFFF, ar_send_375, 4);
                    _er_f[111] = ((ar_bit_93 == 1) ? ar_got_375 : ar_x0_375);
                    _er_f[127] = ((ar_bit_93 == 1) ? ar_x1_375 : ar_got_375);
                    int ar_bit_94 = lane >> 1 & 1;
                    float ar_x0_376 = _er_f[99];
                    float ar_x1_376 = _er_f[107];
                    float ar_send_376 = ((ar_bit_94 == 1) ? ar_x0_376 : ar_x1_376);
                    float ar_got_376 = __shfl_xor_sync(0xFFFFFFFF, ar_send_376, 2);
                    _er_f[99] = ((ar_bit_94 == 1) ? ar_got_376 : ar_x0_376);
                    _er_f[107] = ((ar_bit_94 == 1) ? ar_x1_376 : ar_got_376);
                    float ar_x0_377 = _er_f[103];
                    float ar_x1_377 = _er_f[111];
                    float ar_send_377 = ((ar_bit_94 == 1) ? ar_x0_377 : ar_x1_377);
                    float ar_got_377 = __shfl_xor_sync(0xFFFFFFFF, ar_send_377, 2);
                    _er_f[103] = ((ar_bit_94 == 1) ? ar_got_377 : ar_x0_377);
                    _er_f[111] = ((ar_bit_94 == 1) ? ar_x1_377 : ar_got_377);
                    float ar_x0_378 = _er_f[115];
                    float ar_x1_378 = _er_f[123];
                    float ar_send_378 = ((ar_bit_94 == 1) ? ar_x0_378 : ar_x1_378);
                    float ar_got_378 = __shfl_xor_sync(0xFFFFFFFF, ar_send_378, 2);
                    _er_f[115] = ((ar_bit_94 == 1) ? ar_got_378 : ar_x0_378);
                    _er_f[123] = ((ar_bit_94 == 1) ? ar_x1_378 : ar_got_378);
                    float ar_x0_379 = _er_f[119];
                    float ar_x1_379 = _er_f[127];
                    float ar_send_379 = ((ar_bit_94 == 1) ? ar_x0_379 : ar_x1_379);
                    float ar_got_379 = __shfl_xor_sync(0xFFFFFFFF, ar_send_379, 2);
                    _er_f[119] = ((ar_bit_94 == 1) ? ar_got_379 : ar_x0_379);
                    _er_f[127] = ((ar_bit_94 == 1) ? ar_x1_379 : ar_got_379);
                    int ar_bit_95 = lane & 1;
                    float ar_x0_380 = _er_f[99];
                    float ar_x1_380 = _er_f[103];
                    float ar_send_380 = ((ar_bit_95 == 1) ? ar_x0_380 : ar_x1_380);
                    float ar_got_380 = __shfl_xor_sync(0xFFFFFFFF, ar_send_380, 1);
                    _er_f[99] = ((ar_bit_95 == 1) ? ar_got_380 : ar_x0_380);
                    _er_f[103] = ((ar_bit_95 == 1) ? ar_x1_380 : ar_got_380);
                    float ar_x0_381 = _er_f[107];
                    float ar_x1_381 = _er_f[111];
                    float ar_send_381 = ((ar_bit_95 == 1) ? ar_x0_381 : ar_x1_381);
                    float ar_got_381 = __shfl_xor_sync(0xFFFFFFFF, ar_send_381, 1);
                    _er_f[107] = ((ar_bit_95 == 1) ? ar_got_381 : ar_x0_381);
                    _er_f[111] = ((ar_bit_95 == 1) ? ar_x1_381 : ar_got_381);
                    float ar_x0_382 = _er_f[115];
                    float ar_x1_382 = _er_f[119];
                    float ar_send_382 = ((ar_bit_95 == 1) ? ar_x0_382 : ar_x1_382);
                    float ar_got_382 = __shfl_xor_sync(0xFFFFFFFF, ar_send_382, 1);
                    _er_f[115] = ((ar_bit_95 == 1) ? ar_got_382 : ar_x0_382);
                    _er_f[119] = ((ar_bit_95 == 1) ? ar_x1_382 : ar_got_382);
                    float ar_x0_383 = _er_f[123];
                    float ar_x1_383 = _er_f[127];
                    float ar_send_383 = ((ar_bit_95 == 1) ? ar_x0_383 : ar_x1_383);
                    float ar_got_383 = __shfl_xor_sync(0xFFFFFFFF, ar_send_383, 1);
                    _er_f[123] = ((ar_bit_95 == 1) ? ar_got_383 : ar_x0_383);
                    _er_f[127] = ((ar_bit_95 == 1) ? ar_x1_383 : ar_got_383);
                    int ar_ri_7 = er_at2 + 256 + 96;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_7])), "f"(_er_f[96]), "f"(_er_f[97]), "f"(_er_f[98]), "f"(_er_f[99]) : "memory");
                    ar_ri_7 = ar_ri_7 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_7])), "f"(_er_f[100]), "f"(_er_f[101]), "f"(_er_f[102]), "f"(_er_f[103]) : "memory");
                    ar_ri_7 = ar_ri_7 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_7])), "f"(_er_f[104]), "f"(_er_f[105]), "f"(_er_f[106]), "f"(_er_f[107]) : "memory");
                    ar_ri_7 = ar_ri_7 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_7])), "f"(_er_f[108]), "f"(_er_f[109]), "f"(_er_f[110]), "f"(_er_f[111]) : "memory");
                    ar_ri_7 = ar_ri_7 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_7])), "f"(_er_f[112]), "f"(_er_f[113]), "f"(_er_f[114]), "f"(_er_f[115]) : "memory");
                    ar_ri_7 = ar_ri_7 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_7])), "f"(_er_f[116]), "f"(_er_f[117]), "f"(_er_f[118]), "f"(_er_f[119]) : "memory");
                    ar_ri_7 = ar_ri_7 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_7])), "f"(_er_f[120]), "f"(_er_f[121]), "f"(_er_f[122]), "f"(_er_f[123]) : "memory");
                    ar_ri_7 = ar_ri_7 + ldc;
                    asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_7])), "f"(_er_f[124]), "f"(_er_f[125]), "f"(_er_f[126]), "f"(_er_f[127]) : "memory");
                }
            }
            unsigned int total_eh = total_tiles_e * 2;
            unsigned int tail_base_eh = total_eh / num_clusters_e * num_clusters_e;
            unsigned int num_tail_eh = total_eh - tail_base_eh;
            unsigned int whole_end_eh = total_eh;
            unsigned int units_eh = 0;
            if (tail_splits > 1) {
                whole_end_eh = tail_base_eh;
                units_eh = num_tail_eh * (unsigned int)tail_splits;
            }
            unsigned int num_whole_eh = whole_end_eh - tail_base_e * 2;
            #pragma unroll 1
            for (unsigned int v_e = cluster_e; v_e < num_whole_eh + units_eh; v_e += num_clusters_e) {
                unsigned int j_eh = 0;
                int s_eh = 0;
                int splits_eh = 1;
                unsigned int hs_e = tail_base_e * 2 + v_e;
                if (v_e >= num_whole_eh) {
                    unsigned int u_eh = v_e - num_whole_eh;
                    unsigned int s_u_eh = u_eh / num_tail_eh;
                    j_eh = u_eh - s_u_eh * num_tail_eh;
                    hs_e = tail_base_eh + j_eh;
                    s_eh = (int)s_u_eh;
                    splits_eh = tail_splits;
                }
                unsigned int t_eh = hs_e / 2;
                unsigned int e_ueh = t_eh / tiles_per_group_e;
                unsigned int rem_eh = t_eh - e_ueh * tiles_per_group_e;
                unsigned int n_block_eh = rem_eh / grid_k_e;
                unsigned int k_block_eh = (rem_eh - n_block_eh * grid_k_e) * 2 + (hs_e - t_eh * 2);
                int e_ieh = (int)e_ueh;
                int end_teh = offs[e_ieh];
                int prev_teh = e_ieh - 1;
                if (prev_teh < 0) {
                    prev_teh = 0;
                }
                int start_teh = offs[prev_teh];
                if (e_ieh == 0) {
                    start_teh = 0;
                }
                int ac_live_h = 1;
                int kb_eh = (end_teh - start_teh + (BLOCK_K - 1)) / BLOCK_K;
                if (kb_eh == 0) {
                    kb_eh = 1;
                }
                int lo_eh = s_eh * kb_eh / splits_eh;
                int hi_eh = (s_eh + 1) * kb_eh / splits_eh;
                int n_steps_eh = hi_eh - lo_eh;
                int whole_eh = 0;
                if (lo_eh == 0) {
                    if (hi_eh == kb_eh) {
                        whole_eh = 1;
                    }
                }
                if (n_steps_eh > 0) {
                    mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _eh_f[128];
                    int eh_row = taddr + (unsigned int)(cta_rank * 128 + epi_warp * 32 << 16) + (unsigned int)(epi_half * 128);
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_eh_f[0]), "=f"(_eh_f[1]), "=f"(_eh_f[2]), "=f"(_eh_f[3]), "=f"(_eh_f[4]), "=f"(_eh_f[5]), "=f"(_eh_f[6]), "=f"(_eh_f[7]), "=f"(_eh_f[8]), "=f"(_eh_f[9]), "=f"(_eh_f[10]), "=f"(_eh_f[11]), "=f"(_eh_f[12]), "=f"(_eh_f[13]), "=f"(_eh_f[14]), "=f"(_eh_f[15]), "=f"(_eh_f[16]), "=f"(_eh_f[17]), "=f"(_eh_f[18]), "=f"(_eh_f[19]), "=f"(_eh_f[20]), "=f"(_eh_f[21]), "=f"(_eh_f[22]), "=f"(_eh_f[23]), "=f"(_eh_f[24]), "=f"(_eh_f[25]), "=f"(_eh_f[26]), "=f"(_eh_f[27]), "=f"(_eh_f[28]), "=f"(_eh_f[29]), "=f"(_eh_f[30]), "=f"(_eh_f[31])
                        : "r"(eh_row));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_eh_f[32]), "=f"(_eh_f[33]), "=f"(_eh_f[34]), "=f"(_eh_f[35]), "=f"(_eh_f[36]), "=f"(_eh_f[37]), "=f"(_eh_f[38]), "=f"(_eh_f[39]), "=f"(_eh_f[40]), "=f"(_eh_f[41]), "=f"(_eh_f[42]), "=f"(_eh_f[43]), "=f"(_eh_f[44]), "=f"(_eh_f[45]), "=f"(_eh_f[46]), "=f"(_eh_f[47]), "=f"(_eh_f[48]), "=f"(_eh_f[49]), "=f"(_eh_f[50]), "=f"(_eh_f[51]), "=f"(_eh_f[52]), "=f"(_eh_f[53]), "=f"(_eh_f[54]), "=f"(_eh_f[55]), "=f"(_eh_f[56]), "=f"(_eh_f[57]), "=f"(_eh_f[58]), "=f"(_eh_f[59]), "=f"(_eh_f[60]), "=f"(_eh_f[61]), "=f"(_eh_f[62]), "=f"(_eh_f[63])
                        : "r"(eh_row + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_eh_f[64]), "=f"(_eh_f[65]), "=f"(_eh_f[66]), "=f"(_eh_f[67]), "=f"(_eh_f[68]), "=f"(_eh_f[69]), "=f"(_eh_f[70]), "=f"(_eh_f[71]), "=f"(_eh_f[72]), "=f"(_eh_f[73]), "=f"(_eh_f[74]), "=f"(_eh_f[75]), "=f"(_eh_f[76]), "=f"(_eh_f[77]), "=f"(_eh_f[78]), "=f"(_eh_f[79]), "=f"(_eh_f[80]), "=f"(_eh_f[81]), "=f"(_eh_f[82]), "=f"(_eh_f[83]), "=f"(_eh_f[84]), "=f"(_eh_f[85]), "=f"(_eh_f[86]), "=f"(_eh_f[87]), "=f"(_eh_f[88]), "=f"(_eh_f[89]), "=f"(_eh_f[90]), "=f"(_eh_f[91]), "=f"(_eh_f[92]), "=f"(_eh_f[93]), "=f"(_eh_f[94]), "=f"(_eh_f[95])
                        : "r"(eh_row + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_eh_f[96]), "=f"(_eh_f[97]), "=f"(_eh_f[98]), "=f"(_eh_f[99]), "=f"(_eh_f[100]), "=f"(_eh_f[101]), "=f"(_eh_f[102]), "=f"(_eh_f[103]), "=f"(_eh_f[104]), "=f"(_eh_f[105]), "=f"(_eh_f[106]), "=f"(_eh_f[107]), "=f"(_eh_f[108]), "=f"(_eh_f[109]), "=f"(_eh_f[110]), "=f"(_eh_f[111]), "=f"(_eh_f[112]), "=f"(_eh_f[113]), "=f"(_eh_f[114]), "=f"(_eh_f[115]), "=f"(_eh_f[116]), "=f"(_eh_f[117]), "=f"(_eh_f[118]), "=f"(_eh_f[119]), "=f"(_eh_f[120]), "=f"(_eh_f[121]), "=f"(_eh_f[122]), "=f"(_eh_f[123]), "=f"(_eh_f[124]), "=f"(_eh_f[125]), "=f"(_eh_f[126]), "=f"(_eh_f[127])
                        : "r"(eh_row + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epi_free0_addr) & 0xFEFFFFFF) : "memory");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                    }
                    _phase_mainloop_done ^= 1;
                    if (whole_eh == 1) {
                        int eh_out = (int)e_ueh * stride_e + ((int)n_block_eh * BLOCK_N + cta_rank * BLOCK_M + epi_tid) * ldc + (int)k_block_eh * BLOCK_N + epi_half * 128;
                        if (ac_live_h == 1) {
                            int ar_bit_96 = lane >> 2 & 1;
                            float ar_x0_384 = _eh_f[0];
                            float ar_x1_384 = _eh_f[16];
                            float ar_send_384 = ((ar_bit_96 == 1) ? ar_x0_384 : ar_x1_384);
                            float ar_got_384 = __shfl_xor_sync(0xFFFFFFFF, ar_send_384, 4);
                            _eh_f[0] = ((ar_bit_96 == 1) ? ar_got_384 : ar_x0_384);
                            _eh_f[16] = ((ar_bit_96 == 1) ? ar_x1_384 : ar_got_384);
                            float ar_x0_385 = _eh_f[4];
                            float ar_x1_385 = _eh_f[20];
                            float ar_send_385 = ((ar_bit_96 == 1) ? ar_x0_385 : ar_x1_385);
                            float ar_got_385 = __shfl_xor_sync(0xFFFFFFFF, ar_send_385, 4);
                            _eh_f[4] = ((ar_bit_96 == 1) ? ar_got_385 : ar_x0_385);
                            _eh_f[20] = ((ar_bit_96 == 1) ? ar_x1_385 : ar_got_385);
                            float ar_x0_386 = _eh_f[8];
                            float ar_x1_386 = _eh_f[24];
                            float ar_send_386 = ((ar_bit_96 == 1) ? ar_x0_386 : ar_x1_386);
                            float ar_got_386 = __shfl_xor_sync(0xFFFFFFFF, ar_send_386, 4);
                            _eh_f[8] = ((ar_bit_96 == 1) ? ar_got_386 : ar_x0_386);
                            _eh_f[24] = ((ar_bit_96 == 1) ? ar_x1_386 : ar_got_386);
                            float ar_x0_387 = _eh_f[12];
                            float ar_x1_387 = _eh_f[28];
                            float ar_send_387 = ((ar_bit_96 == 1) ? ar_x0_387 : ar_x1_387);
                            float ar_got_387 = __shfl_xor_sync(0xFFFFFFFF, ar_send_387, 4);
                            _eh_f[12] = ((ar_bit_96 == 1) ? ar_got_387 : ar_x0_387);
                            _eh_f[28] = ((ar_bit_96 == 1) ? ar_x1_387 : ar_got_387);
                            int ar_bit_97 = lane >> 1 & 1;
                            float ar_x0_388 = _eh_f[0];
                            float ar_x1_388 = _eh_f[8];
                            float ar_send_388 = ((ar_bit_97 == 1) ? ar_x0_388 : ar_x1_388);
                            float ar_got_388 = __shfl_xor_sync(0xFFFFFFFF, ar_send_388, 2);
                            _eh_f[0] = ((ar_bit_97 == 1) ? ar_got_388 : ar_x0_388);
                            _eh_f[8] = ((ar_bit_97 == 1) ? ar_x1_388 : ar_got_388);
                            float ar_x0_389 = _eh_f[4];
                            float ar_x1_389 = _eh_f[12];
                            float ar_send_389 = ((ar_bit_97 == 1) ? ar_x0_389 : ar_x1_389);
                            float ar_got_389 = __shfl_xor_sync(0xFFFFFFFF, ar_send_389, 2);
                            _eh_f[4] = ((ar_bit_97 == 1) ? ar_got_389 : ar_x0_389);
                            _eh_f[12] = ((ar_bit_97 == 1) ? ar_x1_389 : ar_got_389);
                            float ar_x0_390 = _eh_f[16];
                            float ar_x1_390 = _eh_f[24];
                            float ar_send_390 = ((ar_bit_97 == 1) ? ar_x0_390 : ar_x1_390);
                            float ar_got_390 = __shfl_xor_sync(0xFFFFFFFF, ar_send_390, 2);
                            _eh_f[16] = ((ar_bit_97 == 1) ? ar_got_390 : ar_x0_390);
                            _eh_f[24] = ((ar_bit_97 == 1) ? ar_x1_390 : ar_got_390);
                            float ar_x0_391 = _eh_f[20];
                            float ar_x1_391 = _eh_f[28];
                            float ar_send_391 = ((ar_bit_97 == 1) ? ar_x0_391 : ar_x1_391);
                            float ar_got_391 = __shfl_xor_sync(0xFFFFFFFF, ar_send_391, 2);
                            _eh_f[20] = ((ar_bit_97 == 1) ? ar_got_391 : ar_x0_391);
                            _eh_f[28] = ((ar_bit_97 == 1) ? ar_x1_391 : ar_got_391);
                            int ar_bit_98 = lane & 1;
                            float ar_x0_392 = _eh_f[0];
                            float ar_x1_392 = _eh_f[4];
                            float ar_send_392 = ((ar_bit_98 == 1) ? ar_x0_392 : ar_x1_392);
                            float ar_got_392 = __shfl_xor_sync(0xFFFFFFFF, ar_send_392, 1);
                            _eh_f[0] = ((ar_bit_98 == 1) ? ar_got_392 : ar_x0_392);
                            _eh_f[4] = ((ar_bit_98 == 1) ? ar_x1_392 : ar_got_392);
                            float ar_x0_393 = _eh_f[8];
                            float ar_x1_393 = _eh_f[12];
                            float ar_send_393 = ((ar_bit_98 == 1) ? ar_x0_393 : ar_x1_393);
                            float ar_got_393 = __shfl_xor_sync(0xFFFFFFFF, ar_send_393, 1);
                            _eh_f[8] = ((ar_bit_98 == 1) ? ar_got_393 : ar_x0_393);
                            _eh_f[12] = ((ar_bit_98 == 1) ? ar_x1_393 : ar_got_393);
                            float ar_x0_394 = _eh_f[16];
                            float ar_x1_394 = _eh_f[20];
                            float ar_send_394 = ((ar_bit_98 == 1) ? ar_x0_394 : ar_x1_394);
                            float ar_got_394 = __shfl_xor_sync(0xFFFFFFFF, ar_send_394, 1);
                            _eh_f[16] = ((ar_bit_98 == 1) ? ar_got_394 : ar_x0_394);
                            _eh_f[20] = ((ar_bit_98 == 1) ? ar_x1_394 : ar_got_394);
                            float ar_x0_395 = _eh_f[24];
                            float ar_x1_395 = _eh_f[28];
                            float ar_send_395 = ((ar_bit_98 == 1) ? ar_x0_395 : ar_x1_395);
                            float ar_got_395 = __shfl_xor_sync(0xFFFFFFFF, ar_send_395, 1);
                            _eh_f[24] = ((ar_bit_98 == 1) ? ar_got_395 : ar_x0_395);
                            _eh_f[28] = ((ar_bit_98 == 1) ? ar_x1_395 : ar_got_395);
                            int ar_bit_99 = lane >> 2 & 1;
                            float ar_x0_396 = _eh_f[1];
                            float ar_x1_396 = _eh_f[17];
                            float ar_send_396 = ((ar_bit_99 == 1) ? ar_x0_396 : ar_x1_396);
                            float ar_got_396 = __shfl_xor_sync(0xFFFFFFFF, ar_send_396, 4);
                            _eh_f[1] = ((ar_bit_99 == 1) ? ar_got_396 : ar_x0_396);
                            _eh_f[17] = ((ar_bit_99 == 1) ? ar_x1_396 : ar_got_396);
                            float ar_x0_397 = _eh_f[5];
                            float ar_x1_397 = _eh_f[21];
                            float ar_send_397 = ((ar_bit_99 == 1) ? ar_x0_397 : ar_x1_397);
                            float ar_got_397 = __shfl_xor_sync(0xFFFFFFFF, ar_send_397, 4);
                            _eh_f[5] = ((ar_bit_99 == 1) ? ar_got_397 : ar_x0_397);
                            _eh_f[21] = ((ar_bit_99 == 1) ? ar_x1_397 : ar_got_397);
                            float ar_x0_398 = _eh_f[9];
                            float ar_x1_398 = _eh_f[25];
                            float ar_send_398 = ((ar_bit_99 == 1) ? ar_x0_398 : ar_x1_398);
                            float ar_got_398 = __shfl_xor_sync(0xFFFFFFFF, ar_send_398, 4);
                            _eh_f[9] = ((ar_bit_99 == 1) ? ar_got_398 : ar_x0_398);
                            _eh_f[25] = ((ar_bit_99 == 1) ? ar_x1_398 : ar_got_398);
                            float ar_x0_399 = _eh_f[13];
                            float ar_x1_399 = _eh_f[29];
                            float ar_send_399 = ((ar_bit_99 == 1) ? ar_x0_399 : ar_x1_399);
                            float ar_got_399 = __shfl_xor_sync(0xFFFFFFFF, ar_send_399, 4);
                            _eh_f[13] = ((ar_bit_99 == 1) ? ar_got_399 : ar_x0_399);
                            _eh_f[29] = ((ar_bit_99 == 1) ? ar_x1_399 : ar_got_399);
                            int ar_bit_100 = lane >> 1 & 1;
                            float ar_x0_400 = _eh_f[1];
                            float ar_x1_400 = _eh_f[9];
                            float ar_send_400 = ((ar_bit_100 == 1) ? ar_x0_400 : ar_x1_400);
                            float ar_got_400 = __shfl_xor_sync(0xFFFFFFFF, ar_send_400, 2);
                            _eh_f[1] = ((ar_bit_100 == 1) ? ar_got_400 : ar_x0_400);
                            _eh_f[9] = ((ar_bit_100 == 1) ? ar_x1_400 : ar_got_400);
                            float ar_x0_401 = _eh_f[5];
                            float ar_x1_401 = _eh_f[13];
                            float ar_send_401 = ((ar_bit_100 == 1) ? ar_x0_401 : ar_x1_401);
                            float ar_got_401 = __shfl_xor_sync(0xFFFFFFFF, ar_send_401, 2);
                            _eh_f[5] = ((ar_bit_100 == 1) ? ar_got_401 : ar_x0_401);
                            _eh_f[13] = ((ar_bit_100 == 1) ? ar_x1_401 : ar_got_401);
                            float ar_x0_402 = _eh_f[17];
                            float ar_x1_402 = _eh_f[25];
                            float ar_send_402 = ((ar_bit_100 == 1) ? ar_x0_402 : ar_x1_402);
                            float ar_got_402 = __shfl_xor_sync(0xFFFFFFFF, ar_send_402, 2);
                            _eh_f[17] = ((ar_bit_100 == 1) ? ar_got_402 : ar_x0_402);
                            _eh_f[25] = ((ar_bit_100 == 1) ? ar_x1_402 : ar_got_402);
                            float ar_x0_403 = _eh_f[21];
                            float ar_x1_403 = _eh_f[29];
                            float ar_send_403 = ((ar_bit_100 == 1) ? ar_x0_403 : ar_x1_403);
                            float ar_got_403 = __shfl_xor_sync(0xFFFFFFFF, ar_send_403, 2);
                            _eh_f[21] = ((ar_bit_100 == 1) ? ar_got_403 : ar_x0_403);
                            _eh_f[29] = ((ar_bit_100 == 1) ? ar_x1_403 : ar_got_403);
                            int ar_bit_101 = lane & 1;
                            float ar_x0_404 = _eh_f[1];
                            float ar_x1_404 = _eh_f[5];
                            float ar_send_404 = ((ar_bit_101 == 1) ? ar_x0_404 : ar_x1_404);
                            float ar_got_404 = __shfl_xor_sync(0xFFFFFFFF, ar_send_404, 1);
                            _eh_f[1] = ((ar_bit_101 == 1) ? ar_got_404 : ar_x0_404);
                            _eh_f[5] = ((ar_bit_101 == 1) ? ar_x1_404 : ar_got_404);
                            float ar_x0_405 = _eh_f[9];
                            float ar_x1_405 = _eh_f[13];
                            float ar_send_405 = ((ar_bit_101 == 1) ? ar_x0_405 : ar_x1_405);
                            float ar_got_405 = __shfl_xor_sync(0xFFFFFFFF, ar_send_405, 1);
                            _eh_f[9] = ((ar_bit_101 == 1) ? ar_got_405 : ar_x0_405);
                            _eh_f[13] = ((ar_bit_101 == 1) ? ar_x1_405 : ar_got_405);
                            float ar_x0_406 = _eh_f[17];
                            float ar_x1_406 = _eh_f[21];
                            float ar_send_406 = ((ar_bit_101 == 1) ? ar_x0_406 : ar_x1_406);
                            float ar_got_406 = __shfl_xor_sync(0xFFFFFFFF, ar_send_406, 1);
                            _eh_f[17] = ((ar_bit_101 == 1) ? ar_got_406 : ar_x0_406);
                            _eh_f[21] = ((ar_bit_101 == 1) ? ar_x1_406 : ar_got_406);
                            float ar_x0_407 = _eh_f[25];
                            float ar_x1_407 = _eh_f[29];
                            float ar_send_407 = ((ar_bit_101 == 1) ? ar_x0_407 : ar_x1_407);
                            float ar_got_407 = __shfl_xor_sync(0xFFFFFFFF, ar_send_407, 1);
                            _eh_f[25] = ((ar_bit_101 == 1) ? ar_got_407 : ar_x0_407);
                            _eh_f[29] = ((ar_bit_101 == 1) ? ar_x1_407 : ar_got_407);
                            int ar_bit_102 = lane >> 2 & 1;
                            float ar_x0_408 = _eh_f[2];
                            float ar_x1_408 = _eh_f[18];
                            float ar_send_408 = ((ar_bit_102 == 1) ? ar_x0_408 : ar_x1_408);
                            float ar_got_408 = __shfl_xor_sync(0xFFFFFFFF, ar_send_408, 4);
                            _eh_f[2] = ((ar_bit_102 == 1) ? ar_got_408 : ar_x0_408);
                            _eh_f[18] = ((ar_bit_102 == 1) ? ar_x1_408 : ar_got_408);
                            float ar_x0_409 = _eh_f[6];
                            float ar_x1_409 = _eh_f[22];
                            float ar_send_409 = ((ar_bit_102 == 1) ? ar_x0_409 : ar_x1_409);
                            float ar_got_409 = __shfl_xor_sync(0xFFFFFFFF, ar_send_409, 4);
                            _eh_f[6] = ((ar_bit_102 == 1) ? ar_got_409 : ar_x0_409);
                            _eh_f[22] = ((ar_bit_102 == 1) ? ar_x1_409 : ar_got_409);
                            float ar_x0_410 = _eh_f[10];
                            float ar_x1_410 = _eh_f[26];
                            float ar_send_410 = ((ar_bit_102 == 1) ? ar_x0_410 : ar_x1_410);
                            float ar_got_410 = __shfl_xor_sync(0xFFFFFFFF, ar_send_410, 4);
                            _eh_f[10] = ((ar_bit_102 == 1) ? ar_got_410 : ar_x0_410);
                            _eh_f[26] = ((ar_bit_102 == 1) ? ar_x1_410 : ar_got_410);
                            float ar_x0_411 = _eh_f[14];
                            float ar_x1_411 = _eh_f[30];
                            float ar_send_411 = ((ar_bit_102 == 1) ? ar_x0_411 : ar_x1_411);
                            float ar_got_411 = __shfl_xor_sync(0xFFFFFFFF, ar_send_411, 4);
                            _eh_f[14] = ((ar_bit_102 == 1) ? ar_got_411 : ar_x0_411);
                            _eh_f[30] = ((ar_bit_102 == 1) ? ar_x1_411 : ar_got_411);
                            int ar_bit_103 = lane >> 1 & 1;
                            float ar_x0_412 = _eh_f[2];
                            float ar_x1_412 = _eh_f[10];
                            float ar_send_412 = ((ar_bit_103 == 1) ? ar_x0_412 : ar_x1_412);
                            float ar_got_412 = __shfl_xor_sync(0xFFFFFFFF, ar_send_412, 2);
                            _eh_f[2] = ((ar_bit_103 == 1) ? ar_got_412 : ar_x0_412);
                            _eh_f[10] = ((ar_bit_103 == 1) ? ar_x1_412 : ar_got_412);
                            float ar_x0_413 = _eh_f[6];
                            float ar_x1_413 = _eh_f[14];
                            float ar_send_413 = ((ar_bit_103 == 1) ? ar_x0_413 : ar_x1_413);
                            float ar_got_413 = __shfl_xor_sync(0xFFFFFFFF, ar_send_413, 2);
                            _eh_f[6] = ((ar_bit_103 == 1) ? ar_got_413 : ar_x0_413);
                            _eh_f[14] = ((ar_bit_103 == 1) ? ar_x1_413 : ar_got_413);
                            float ar_x0_414 = _eh_f[18];
                            float ar_x1_414 = _eh_f[26];
                            float ar_send_414 = ((ar_bit_103 == 1) ? ar_x0_414 : ar_x1_414);
                            float ar_got_414 = __shfl_xor_sync(0xFFFFFFFF, ar_send_414, 2);
                            _eh_f[18] = ((ar_bit_103 == 1) ? ar_got_414 : ar_x0_414);
                            _eh_f[26] = ((ar_bit_103 == 1) ? ar_x1_414 : ar_got_414);
                            float ar_x0_415 = _eh_f[22];
                            float ar_x1_415 = _eh_f[30];
                            float ar_send_415 = ((ar_bit_103 == 1) ? ar_x0_415 : ar_x1_415);
                            float ar_got_415 = __shfl_xor_sync(0xFFFFFFFF, ar_send_415, 2);
                            _eh_f[22] = ((ar_bit_103 == 1) ? ar_got_415 : ar_x0_415);
                            _eh_f[30] = ((ar_bit_103 == 1) ? ar_x1_415 : ar_got_415);
                            int ar_bit_104 = lane & 1;
                            float ar_x0_416 = _eh_f[2];
                            float ar_x1_416 = _eh_f[6];
                            float ar_send_416 = ((ar_bit_104 == 1) ? ar_x0_416 : ar_x1_416);
                            float ar_got_416 = __shfl_xor_sync(0xFFFFFFFF, ar_send_416, 1);
                            _eh_f[2] = ((ar_bit_104 == 1) ? ar_got_416 : ar_x0_416);
                            _eh_f[6] = ((ar_bit_104 == 1) ? ar_x1_416 : ar_got_416);
                            float ar_x0_417 = _eh_f[10];
                            float ar_x1_417 = _eh_f[14];
                            float ar_send_417 = ((ar_bit_104 == 1) ? ar_x0_417 : ar_x1_417);
                            float ar_got_417 = __shfl_xor_sync(0xFFFFFFFF, ar_send_417, 1);
                            _eh_f[10] = ((ar_bit_104 == 1) ? ar_got_417 : ar_x0_417);
                            _eh_f[14] = ((ar_bit_104 == 1) ? ar_x1_417 : ar_got_417);
                            float ar_x0_418 = _eh_f[18];
                            float ar_x1_418 = _eh_f[22];
                            float ar_send_418 = ((ar_bit_104 == 1) ? ar_x0_418 : ar_x1_418);
                            float ar_got_418 = __shfl_xor_sync(0xFFFFFFFF, ar_send_418, 1);
                            _eh_f[18] = ((ar_bit_104 == 1) ? ar_got_418 : ar_x0_418);
                            _eh_f[22] = ((ar_bit_104 == 1) ? ar_x1_418 : ar_got_418);
                            float ar_x0_419 = _eh_f[26];
                            float ar_x1_419 = _eh_f[30];
                            float ar_send_419 = ((ar_bit_104 == 1) ? ar_x0_419 : ar_x1_419);
                            float ar_got_419 = __shfl_xor_sync(0xFFFFFFFF, ar_send_419, 1);
                            _eh_f[26] = ((ar_bit_104 == 1) ? ar_got_419 : ar_x0_419);
                            _eh_f[30] = ((ar_bit_104 == 1) ? ar_x1_419 : ar_got_419);
                            int ar_bit_105 = lane >> 2 & 1;
                            float ar_x0_420 = _eh_f[3];
                            float ar_x1_420 = _eh_f[19];
                            float ar_send_420 = ((ar_bit_105 == 1) ? ar_x0_420 : ar_x1_420);
                            float ar_got_420 = __shfl_xor_sync(0xFFFFFFFF, ar_send_420, 4);
                            _eh_f[3] = ((ar_bit_105 == 1) ? ar_got_420 : ar_x0_420);
                            _eh_f[19] = ((ar_bit_105 == 1) ? ar_x1_420 : ar_got_420);
                            float ar_x0_421 = _eh_f[7];
                            float ar_x1_421 = _eh_f[23];
                            float ar_send_421 = ((ar_bit_105 == 1) ? ar_x0_421 : ar_x1_421);
                            float ar_got_421 = __shfl_xor_sync(0xFFFFFFFF, ar_send_421, 4);
                            _eh_f[7] = ((ar_bit_105 == 1) ? ar_got_421 : ar_x0_421);
                            _eh_f[23] = ((ar_bit_105 == 1) ? ar_x1_421 : ar_got_421);
                            float ar_x0_422 = _eh_f[11];
                            float ar_x1_422 = _eh_f[27];
                            float ar_send_422 = ((ar_bit_105 == 1) ? ar_x0_422 : ar_x1_422);
                            float ar_got_422 = __shfl_xor_sync(0xFFFFFFFF, ar_send_422, 4);
                            _eh_f[11] = ((ar_bit_105 == 1) ? ar_got_422 : ar_x0_422);
                            _eh_f[27] = ((ar_bit_105 == 1) ? ar_x1_422 : ar_got_422);
                            float ar_x0_423 = _eh_f[15];
                            float ar_x1_423 = _eh_f[31];
                            float ar_send_423 = ((ar_bit_105 == 1) ? ar_x0_423 : ar_x1_423);
                            float ar_got_423 = __shfl_xor_sync(0xFFFFFFFF, ar_send_423, 4);
                            _eh_f[15] = ((ar_bit_105 == 1) ? ar_got_423 : ar_x0_423);
                            _eh_f[31] = ((ar_bit_105 == 1) ? ar_x1_423 : ar_got_423);
                            int ar_bit_106 = lane >> 1 & 1;
                            float ar_x0_424 = _eh_f[3];
                            float ar_x1_424 = _eh_f[11];
                            float ar_send_424 = ((ar_bit_106 == 1) ? ar_x0_424 : ar_x1_424);
                            float ar_got_424 = __shfl_xor_sync(0xFFFFFFFF, ar_send_424, 2);
                            _eh_f[3] = ((ar_bit_106 == 1) ? ar_got_424 : ar_x0_424);
                            _eh_f[11] = ((ar_bit_106 == 1) ? ar_x1_424 : ar_got_424);
                            float ar_x0_425 = _eh_f[7];
                            float ar_x1_425 = _eh_f[15];
                            float ar_send_425 = ((ar_bit_106 == 1) ? ar_x0_425 : ar_x1_425);
                            float ar_got_425 = __shfl_xor_sync(0xFFFFFFFF, ar_send_425, 2);
                            _eh_f[7] = ((ar_bit_106 == 1) ? ar_got_425 : ar_x0_425);
                            _eh_f[15] = ((ar_bit_106 == 1) ? ar_x1_425 : ar_got_425);
                            float ar_x0_426 = _eh_f[19];
                            float ar_x1_426 = _eh_f[27];
                            float ar_send_426 = ((ar_bit_106 == 1) ? ar_x0_426 : ar_x1_426);
                            float ar_got_426 = __shfl_xor_sync(0xFFFFFFFF, ar_send_426, 2);
                            _eh_f[19] = ((ar_bit_106 == 1) ? ar_got_426 : ar_x0_426);
                            _eh_f[27] = ((ar_bit_106 == 1) ? ar_x1_426 : ar_got_426);
                            float ar_x0_427 = _eh_f[23];
                            float ar_x1_427 = _eh_f[31];
                            float ar_send_427 = ((ar_bit_106 == 1) ? ar_x0_427 : ar_x1_427);
                            float ar_got_427 = __shfl_xor_sync(0xFFFFFFFF, ar_send_427, 2);
                            _eh_f[23] = ((ar_bit_106 == 1) ? ar_got_427 : ar_x0_427);
                            _eh_f[31] = ((ar_bit_106 == 1) ? ar_x1_427 : ar_got_427);
                            int ar_bit_107 = lane & 1;
                            float ar_x0_428 = _eh_f[3];
                            float ar_x1_428 = _eh_f[7];
                            float ar_send_428 = ((ar_bit_107 == 1) ? ar_x0_428 : ar_x1_428);
                            float ar_got_428 = __shfl_xor_sync(0xFFFFFFFF, ar_send_428, 1);
                            _eh_f[3] = ((ar_bit_107 == 1) ? ar_got_428 : ar_x0_428);
                            _eh_f[7] = ((ar_bit_107 == 1) ? ar_x1_428 : ar_got_428);
                            float ar_x0_429 = _eh_f[11];
                            float ar_x1_429 = _eh_f[15];
                            float ar_send_429 = ((ar_bit_107 == 1) ? ar_x0_429 : ar_x1_429);
                            float ar_got_429 = __shfl_xor_sync(0xFFFFFFFF, ar_send_429, 1);
                            _eh_f[11] = ((ar_bit_107 == 1) ? ar_got_429 : ar_x0_429);
                            _eh_f[15] = ((ar_bit_107 == 1) ? ar_x1_429 : ar_got_429);
                            float ar_x0_430 = _eh_f[19];
                            float ar_x1_430 = _eh_f[23];
                            float ar_send_430 = ((ar_bit_107 == 1) ? ar_x0_430 : ar_x1_430);
                            float ar_got_430 = __shfl_xor_sync(0xFFFFFFFF, ar_send_430, 1);
                            _eh_f[19] = ((ar_bit_107 == 1) ? ar_got_430 : ar_x0_430);
                            _eh_f[23] = ((ar_bit_107 == 1) ? ar_x1_430 : ar_got_430);
                            float ar_x0_431 = _eh_f[27];
                            float ar_x1_431 = _eh_f[31];
                            float ar_send_431 = ((ar_bit_107 == 1) ? ar_x0_431 : ar_x1_431);
                            float ar_got_431 = __shfl_xor_sync(0xFFFFFFFF, ar_send_431, 1);
                            _eh_f[27] = ((ar_bit_107 == 1) ? ar_got_431 : ar_x0_431);
                            _eh_f[31] = ((ar_bit_107 == 1) ? ar_x1_431 : ar_got_431);
                            int ar_ri_8 = eh_out + (lane & 7) * (4 - ldc);
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_8])), "f"(_eh_f[0]), "f"(_eh_f[1]), "f"(_eh_f[2]), "f"(_eh_f[3]) : "memory");
                            ar_ri_8 = ar_ri_8 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_8])), "f"(_eh_f[4]), "f"(_eh_f[5]), "f"(_eh_f[6]), "f"(_eh_f[7]) : "memory");
                            ar_ri_8 = ar_ri_8 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_8])), "f"(_eh_f[8]), "f"(_eh_f[9]), "f"(_eh_f[10]), "f"(_eh_f[11]) : "memory");
                            ar_ri_8 = ar_ri_8 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_8])), "f"(_eh_f[12]), "f"(_eh_f[13]), "f"(_eh_f[14]), "f"(_eh_f[15]) : "memory");
                            ar_ri_8 = ar_ri_8 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_8])), "f"(_eh_f[16]), "f"(_eh_f[17]), "f"(_eh_f[18]), "f"(_eh_f[19]) : "memory");
                            ar_ri_8 = ar_ri_8 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_8])), "f"(_eh_f[20]), "f"(_eh_f[21]), "f"(_eh_f[22]), "f"(_eh_f[23]) : "memory");
                            ar_ri_8 = ar_ri_8 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_8])), "f"(_eh_f[24]), "f"(_eh_f[25]), "f"(_eh_f[26]), "f"(_eh_f[27]) : "memory");
                            ar_ri_8 = ar_ri_8 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_8])), "f"(_eh_f[28]), "f"(_eh_f[29]), "f"(_eh_f[30]), "f"(_eh_f[31]) : "memory");
                            int ar_bit_108 = lane >> 2 & 1;
                            float ar_x0_432 = _eh_f[32];
                            float ar_x1_432 = _eh_f[48];
                            float ar_send_432 = ((ar_bit_108 == 1) ? ar_x0_432 : ar_x1_432);
                            float ar_got_432 = __shfl_xor_sync(0xFFFFFFFF, ar_send_432, 4);
                            _eh_f[32] = ((ar_bit_108 == 1) ? ar_got_432 : ar_x0_432);
                            _eh_f[48] = ((ar_bit_108 == 1) ? ar_x1_432 : ar_got_432);
                            float ar_x0_433 = _eh_f[36];
                            float ar_x1_433 = _eh_f[52];
                            float ar_send_433 = ((ar_bit_108 == 1) ? ar_x0_433 : ar_x1_433);
                            float ar_got_433 = __shfl_xor_sync(0xFFFFFFFF, ar_send_433, 4);
                            _eh_f[36] = ((ar_bit_108 == 1) ? ar_got_433 : ar_x0_433);
                            _eh_f[52] = ((ar_bit_108 == 1) ? ar_x1_433 : ar_got_433);
                            float ar_x0_434 = _eh_f[40];
                            float ar_x1_434 = _eh_f[56];
                            float ar_send_434 = ((ar_bit_108 == 1) ? ar_x0_434 : ar_x1_434);
                            float ar_got_434 = __shfl_xor_sync(0xFFFFFFFF, ar_send_434, 4);
                            _eh_f[40] = ((ar_bit_108 == 1) ? ar_got_434 : ar_x0_434);
                            _eh_f[56] = ((ar_bit_108 == 1) ? ar_x1_434 : ar_got_434);
                            float ar_x0_435 = _eh_f[44];
                            float ar_x1_435 = _eh_f[60];
                            float ar_send_435 = ((ar_bit_108 == 1) ? ar_x0_435 : ar_x1_435);
                            float ar_got_435 = __shfl_xor_sync(0xFFFFFFFF, ar_send_435, 4);
                            _eh_f[44] = ((ar_bit_108 == 1) ? ar_got_435 : ar_x0_435);
                            _eh_f[60] = ((ar_bit_108 == 1) ? ar_x1_435 : ar_got_435);
                            int ar_bit_109 = lane >> 1 & 1;
                            float ar_x0_436 = _eh_f[32];
                            float ar_x1_436 = _eh_f[40];
                            float ar_send_436 = ((ar_bit_109 == 1) ? ar_x0_436 : ar_x1_436);
                            float ar_got_436 = __shfl_xor_sync(0xFFFFFFFF, ar_send_436, 2);
                            _eh_f[32] = ((ar_bit_109 == 1) ? ar_got_436 : ar_x0_436);
                            _eh_f[40] = ((ar_bit_109 == 1) ? ar_x1_436 : ar_got_436);
                            float ar_x0_437 = _eh_f[36];
                            float ar_x1_437 = _eh_f[44];
                            float ar_send_437 = ((ar_bit_109 == 1) ? ar_x0_437 : ar_x1_437);
                            float ar_got_437 = __shfl_xor_sync(0xFFFFFFFF, ar_send_437, 2);
                            _eh_f[36] = ((ar_bit_109 == 1) ? ar_got_437 : ar_x0_437);
                            _eh_f[44] = ((ar_bit_109 == 1) ? ar_x1_437 : ar_got_437);
                            float ar_x0_438 = _eh_f[48];
                            float ar_x1_438 = _eh_f[56];
                            float ar_send_438 = ((ar_bit_109 == 1) ? ar_x0_438 : ar_x1_438);
                            float ar_got_438 = __shfl_xor_sync(0xFFFFFFFF, ar_send_438, 2);
                            _eh_f[48] = ((ar_bit_109 == 1) ? ar_got_438 : ar_x0_438);
                            _eh_f[56] = ((ar_bit_109 == 1) ? ar_x1_438 : ar_got_438);
                            float ar_x0_439 = _eh_f[52];
                            float ar_x1_439 = _eh_f[60];
                            float ar_send_439 = ((ar_bit_109 == 1) ? ar_x0_439 : ar_x1_439);
                            float ar_got_439 = __shfl_xor_sync(0xFFFFFFFF, ar_send_439, 2);
                            _eh_f[52] = ((ar_bit_109 == 1) ? ar_got_439 : ar_x0_439);
                            _eh_f[60] = ((ar_bit_109 == 1) ? ar_x1_439 : ar_got_439);
                            int ar_bit_110 = lane & 1;
                            float ar_x0_440 = _eh_f[32];
                            float ar_x1_440 = _eh_f[36];
                            float ar_send_440 = ((ar_bit_110 == 1) ? ar_x0_440 : ar_x1_440);
                            float ar_got_440 = __shfl_xor_sync(0xFFFFFFFF, ar_send_440, 1);
                            _eh_f[32] = ((ar_bit_110 == 1) ? ar_got_440 : ar_x0_440);
                            _eh_f[36] = ((ar_bit_110 == 1) ? ar_x1_440 : ar_got_440);
                            float ar_x0_441 = _eh_f[40];
                            float ar_x1_441 = _eh_f[44];
                            float ar_send_441 = ((ar_bit_110 == 1) ? ar_x0_441 : ar_x1_441);
                            float ar_got_441 = __shfl_xor_sync(0xFFFFFFFF, ar_send_441, 1);
                            _eh_f[40] = ((ar_bit_110 == 1) ? ar_got_441 : ar_x0_441);
                            _eh_f[44] = ((ar_bit_110 == 1) ? ar_x1_441 : ar_got_441);
                            float ar_x0_442 = _eh_f[48];
                            float ar_x1_442 = _eh_f[52];
                            float ar_send_442 = ((ar_bit_110 == 1) ? ar_x0_442 : ar_x1_442);
                            float ar_got_442 = __shfl_xor_sync(0xFFFFFFFF, ar_send_442, 1);
                            _eh_f[48] = ((ar_bit_110 == 1) ? ar_got_442 : ar_x0_442);
                            _eh_f[52] = ((ar_bit_110 == 1) ? ar_x1_442 : ar_got_442);
                            float ar_x0_443 = _eh_f[56];
                            float ar_x1_443 = _eh_f[60];
                            float ar_send_443 = ((ar_bit_110 == 1) ? ar_x0_443 : ar_x1_443);
                            float ar_got_443 = __shfl_xor_sync(0xFFFFFFFF, ar_send_443, 1);
                            _eh_f[56] = ((ar_bit_110 == 1) ? ar_got_443 : ar_x0_443);
                            _eh_f[60] = ((ar_bit_110 == 1) ? ar_x1_443 : ar_got_443);
                            int ar_bit_111 = lane >> 2 & 1;
                            float ar_x0_444 = _eh_f[33];
                            float ar_x1_444 = _eh_f[49];
                            float ar_send_444 = ((ar_bit_111 == 1) ? ar_x0_444 : ar_x1_444);
                            float ar_got_444 = __shfl_xor_sync(0xFFFFFFFF, ar_send_444, 4);
                            _eh_f[33] = ((ar_bit_111 == 1) ? ar_got_444 : ar_x0_444);
                            _eh_f[49] = ((ar_bit_111 == 1) ? ar_x1_444 : ar_got_444);
                            float ar_x0_445 = _eh_f[37];
                            float ar_x1_445 = _eh_f[53];
                            float ar_send_445 = ((ar_bit_111 == 1) ? ar_x0_445 : ar_x1_445);
                            float ar_got_445 = __shfl_xor_sync(0xFFFFFFFF, ar_send_445, 4);
                            _eh_f[37] = ((ar_bit_111 == 1) ? ar_got_445 : ar_x0_445);
                            _eh_f[53] = ((ar_bit_111 == 1) ? ar_x1_445 : ar_got_445);
                            float ar_x0_446 = _eh_f[41];
                            float ar_x1_446 = _eh_f[57];
                            float ar_send_446 = ((ar_bit_111 == 1) ? ar_x0_446 : ar_x1_446);
                            float ar_got_446 = __shfl_xor_sync(0xFFFFFFFF, ar_send_446, 4);
                            _eh_f[41] = ((ar_bit_111 == 1) ? ar_got_446 : ar_x0_446);
                            _eh_f[57] = ((ar_bit_111 == 1) ? ar_x1_446 : ar_got_446);
                            float ar_x0_447 = _eh_f[45];
                            float ar_x1_447 = _eh_f[61];
                            float ar_send_447 = ((ar_bit_111 == 1) ? ar_x0_447 : ar_x1_447);
                            float ar_got_447 = __shfl_xor_sync(0xFFFFFFFF, ar_send_447, 4);
                            _eh_f[45] = ((ar_bit_111 == 1) ? ar_got_447 : ar_x0_447);
                            _eh_f[61] = ((ar_bit_111 == 1) ? ar_x1_447 : ar_got_447);
                            int ar_bit_112 = lane >> 1 & 1;
                            float ar_x0_448 = _eh_f[33];
                            float ar_x1_448 = _eh_f[41];
                            float ar_send_448 = ((ar_bit_112 == 1) ? ar_x0_448 : ar_x1_448);
                            float ar_got_448 = __shfl_xor_sync(0xFFFFFFFF, ar_send_448, 2);
                            _eh_f[33] = ((ar_bit_112 == 1) ? ar_got_448 : ar_x0_448);
                            _eh_f[41] = ((ar_bit_112 == 1) ? ar_x1_448 : ar_got_448);
                            float ar_x0_449 = _eh_f[37];
                            float ar_x1_449 = _eh_f[45];
                            float ar_send_449 = ((ar_bit_112 == 1) ? ar_x0_449 : ar_x1_449);
                            float ar_got_449 = __shfl_xor_sync(0xFFFFFFFF, ar_send_449, 2);
                            _eh_f[37] = ((ar_bit_112 == 1) ? ar_got_449 : ar_x0_449);
                            _eh_f[45] = ((ar_bit_112 == 1) ? ar_x1_449 : ar_got_449);
                            float ar_x0_450 = _eh_f[49];
                            float ar_x1_450 = _eh_f[57];
                            float ar_send_450 = ((ar_bit_112 == 1) ? ar_x0_450 : ar_x1_450);
                            float ar_got_450 = __shfl_xor_sync(0xFFFFFFFF, ar_send_450, 2);
                            _eh_f[49] = ((ar_bit_112 == 1) ? ar_got_450 : ar_x0_450);
                            _eh_f[57] = ((ar_bit_112 == 1) ? ar_x1_450 : ar_got_450);
                            float ar_x0_451 = _eh_f[53];
                            float ar_x1_451 = _eh_f[61];
                            float ar_send_451 = ((ar_bit_112 == 1) ? ar_x0_451 : ar_x1_451);
                            float ar_got_451 = __shfl_xor_sync(0xFFFFFFFF, ar_send_451, 2);
                            _eh_f[53] = ((ar_bit_112 == 1) ? ar_got_451 : ar_x0_451);
                            _eh_f[61] = ((ar_bit_112 == 1) ? ar_x1_451 : ar_got_451);
                            int ar_bit_113 = lane & 1;
                            float ar_x0_452 = _eh_f[33];
                            float ar_x1_452 = _eh_f[37];
                            float ar_send_452 = ((ar_bit_113 == 1) ? ar_x0_452 : ar_x1_452);
                            float ar_got_452 = __shfl_xor_sync(0xFFFFFFFF, ar_send_452, 1);
                            _eh_f[33] = ((ar_bit_113 == 1) ? ar_got_452 : ar_x0_452);
                            _eh_f[37] = ((ar_bit_113 == 1) ? ar_x1_452 : ar_got_452);
                            float ar_x0_453 = _eh_f[41];
                            float ar_x1_453 = _eh_f[45];
                            float ar_send_453 = ((ar_bit_113 == 1) ? ar_x0_453 : ar_x1_453);
                            float ar_got_453 = __shfl_xor_sync(0xFFFFFFFF, ar_send_453, 1);
                            _eh_f[41] = ((ar_bit_113 == 1) ? ar_got_453 : ar_x0_453);
                            _eh_f[45] = ((ar_bit_113 == 1) ? ar_x1_453 : ar_got_453);
                            float ar_x0_454 = _eh_f[49];
                            float ar_x1_454 = _eh_f[53];
                            float ar_send_454 = ((ar_bit_113 == 1) ? ar_x0_454 : ar_x1_454);
                            float ar_got_454 = __shfl_xor_sync(0xFFFFFFFF, ar_send_454, 1);
                            _eh_f[49] = ((ar_bit_113 == 1) ? ar_got_454 : ar_x0_454);
                            _eh_f[53] = ((ar_bit_113 == 1) ? ar_x1_454 : ar_got_454);
                            float ar_x0_455 = _eh_f[57];
                            float ar_x1_455 = _eh_f[61];
                            float ar_send_455 = ((ar_bit_113 == 1) ? ar_x0_455 : ar_x1_455);
                            float ar_got_455 = __shfl_xor_sync(0xFFFFFFFF, ar_send_455, 1);
                            _eh_f[57] = ((ar_bit_113 == 1) ? ar_got_455 : ar_x0_455);
                            _eh_f[61] = ((ar_bit_113 == 1) ? ar_x1_455 : ar_got_455);
                            int ar_bit_114 = lane >> 2 & 1;
                            float ar_x0_456 = _eh_f[34];
                            float ar_x1_456 = _eh_f[50];
                            float ar_send_456 = ((ar_bit_114 == 1) ? ar_x0_456 : ar_x1_456);
                            float ar_got_456 = __shfl_xor_sync(0xFFFFFFFF, ar_send_456, 4);
                            _eh_f[34] = ((ar_bit_114 == 1) ? ar_got_456 : ar_x0_456);
                            _eh_f[50] = ((ar_bit_114 == 1) ? ar_x1_456 : ar_got_456);
                            float ar_x0_457 = _eh_f[38];
                            float ar_x1_457 = _eh_f[54];
                            float ar_send_457 = ((ar_bit_114 == 1) ? ar_x0_457 : ar_x1_457);
                            float ar_got_457 = __shfl_xor_sync(0xFFFFFFFF, ar_send_457, 4);
                            _eh_f[38] = ((ar_bit_114 == 1) ? ar_got_457 : ar_x0_457);
                            _eh_f[54] = ((ar_bit_114 == 1) ? ar_x1_457 : ar_got_457);
                            float ar_x0_458 = _eh_f[42];
                            float ar_x1_458 = _eh_f[58];
                            float ar_send_458 = ((ar_bit_114 == 1) ? ar_x0_458 : ar_x1_458);
                            float ar_got_458 = __shfl_xor_sync(0xFFFFFFFF, ar_send_458, 4);
                            _eh_f[42] = ((ar_bit_114 == 1) ? ar_got_458 : ar_x0_458);
                            _eh_f[58] = ((ar_bit_114 == 1) ? ar_x1_458 : ar_got_458);
                            float ar_x0_459 = _eh_f[46];
                            float ar_x1_459 = _eh_f[62];
                            float ar_send_459 = ((ar_bit_114 == 1) ? ar_x0_459 : ar_x1_459);
                            float ar_got_459 = __shfl_xor_sync(0xFFFFFFFF, ar_send_459, 4);
                            _eh_f[46] = ((ar_bit_114 == 1) ? ar_got_459 : ar_x0_459);
                            _eh_f[62] = ((ar_bit_114 == 1) ? ar_x1_459 : ar_got_459);
                            int ar_bit_115 = lane >> 1 & 1;
                            float ar_x0_460 = _eh_f[34];
                            float ar_x1_460 = _eh_f[42];
                            float ar_send_460 = ((ar_bit_115 == 1) ? ar_x0_460 : ar_x1_460);
                            float ar_got_460 = __shfl_xor_sync(0xFFFFFFFF, ar_send_460, 2);
                            _eh_f[34] = ((ar_bit_115 == 1) ? ar_got_460 : ar_x0_460);
                            _eh_f[42] = ((ar_bit_115 == 1) ? ar_x1_460 : ar_got_460);
                            float ar_x0_461 = _eh_f[38];
                            float ar_x1_461 = _eh_f[46];
                            float ar_send_461 = ((ar_bit_115 == 1) ? ar_x0_461 : ar_x1_461);
                            float ar_got_461 = __shfl_xor_sync(0xFFFFFFFF, ar_send_461, 2);
                            _eh_f[38] = ((ar_bit_115 == 1) ? ar_got_461 : ar_x0_461);
                            _eh_f[46] = ((ar_bit_115 == 1) ? ar_x1_461 : ar_got_461);
                            float ar_x0_462 = _eh_f[50];
                            float ar_x1_462 = _eh_f[58];
                            float ar_send_462 = ((ar_bit_115 == 1) ? ar_x0_462 : ar_x1_462);
                            float ar_got_462 = __shfl_xor_sync(0xFFFFFFFF, ar_send_462, 2);
                            _eh_f[50] = ((ar_bit_115 == 1) ? ar_got_462 : ar_x0_462);
                            _eh_f[58] = ((ar_bit_115 == 1) ? ar_x1_462 : ar_got_462);
                            float ar_x0_463 = _eh_f[54];
                            float ar_x1_463 = _eh_f[62];
                            float ar_send_463 = ((ar_bit_115 == 1) ? ar_x0_463 : ar_x1_463);
                            float ar_got_463 = __shfl_xor_sync(0xFFFFFFFF, ar_send_463, 2);
                            _eh_f[54] = ((ar_bit_115 == 1) ? ar_got_463 : ar_x0_463);
                            _eh_f[62] = ((ar_bit_115 == 1) ? ar_x1_463 : ar_got_463);
                            int ar_bit_116 = lane & 1;
                            float ar_x0_464 = _eh_f[34];
                            float ar_x1_464 = _eh_f[38];
                            float ar_send_464 = ((ar_bit_116 == 1) ? ar_x0_464 : ar_x1_464);
                            float ar_got_464 = __shfl_xor_sync(0xFFFFFFFF, ar_send_464, 1);
                            _eh_f[34] = ((ar_bit_116 == 1) ? ar_got_464 : ar_x0_464);
                            _eh_f[38] = ((ar_bit_116 == 1) ? ar_x1_464 : ar_got_464);
                            float ar_x0_465 = _eh_f[42];
                            float ar_x1_465 = _eh_f[46];
                            float ar_send_465 = ((ar_bit_116 == 1) ? ar_x0_465 : ar_x1_465);
                            float ar_got_465 = __shfl_xor_sync(0xFFFFFFFF, ar_send_465, 1);
                            _eh_f[42] = ((ar_bit_116 == 1) ? ar_got_465 : ar_x0_465);
                            _eh_f[46] = ((ar_bit_116 == 1) ? ar_x1_465 : ar_got_465);
                            float ar_x0_466 = _eh_f[50];
                            float ar_x1_466 = _eh_f[54];
                            float ar_send_466 = ((ar_bit_116 == 1) ? ar_x0_466 : ar_x1_466);
                            float ar_got_466 = __shfl_xor_sync(0xFFFFFFFF, ar_send_466, 1);
                            _eh_f[50] = ((ar_bit_116 == 1) ? ar_got_466 : ar_x0_466);
                            _eh_f[54] = ((ar_bit_116 == 1) ? ar_x1_466 : ar_got_466);
                            float ar_x0_467 = _eh_f[58];
                            float ar_x1_467 = _eh_f[62];
                            float ar_send_467 = ((ar_bit_116 == 1) ? ar_x0_467 : ar_x1_467);
                            float ar_got_467 = __shfl_xor_sync(0xFFFFFFFF, ar_send_467, 1);
                            _eh_f[58] = ((ar_bit_116 == 1) ? ar_got_467 : ar_x0_467);
                            _eh_f[62] = ((ar_bit_116 == 1) ? ar_x1_467 : ar_got_467);
                            int ar_bit_117 = lane >> 2 & 1;
                            float ar_x0_468 = _eh_f[35];
                            float ar_x1_468 = _eh_f[51];
                            float ar_send_468 = ((ar_bit_117 == 1) ? ar_x0_468 : ar_x1_468);
                            float ar_got_468 = __shfl_xor_sync(0xFFFFFFFF, ar_send_468, 4);
                            _eh_f[35] = ((ar_bit_117 == 1) ? ar_got_468 : ar_x0_468);
                            _eh_f[51] = ((ar_bit_117 == 1) ? ar_x1_468 : ar_got_468);
                            float ar_x0_469 = _eh_f[39];
                            float ar_x1_469 = _eh_f[55];
                            float ar_send_469 = ((ar_bit_117 == 1) ? ar_x0_469 : ar_x1_469);
                            float ar_got_469 = __shfl_xor_sync(0xFFFFFFFF, ar_send_469, 4);
                            _eh_f[39] = ((ar_bit_117 == 1) ? ar_got_469 : ar_x0_469);
                            _eh_f[55] = ((ar_bit_117 == 1) ? ar_x1_469 : ar_got_469);
                            float ar_x0_470 = _eh_f[43];
                            float ar_x1_470 = _eh_f[59];
                            float ar_send_470 = ((ar_bit_117 == 1) ? ar_x0_470 : ar_x1_470);
                            float ar_got_470 = __shfl_xor_sync(0xFFFFFFFF, ar_send_470, 4);
                            _eh_f[43] = ((ar_bit_117 == 1) ? ar_got_470 : ar_x0_470);
                            _eh_f[59] = ((ar_bit_117 == 1) ? ar_x1_470 : ar_got_470);
                            float ar_x0_471 = _eh_f[47];
                            float ar_x1_471 = _eh_f[63];
                            float ar_send_471 = ((ar_bit_117 == 1) ? ar_x0_471 : ar_x1_471);
                            float ar_got_471 = __shfl_xor_sync(0xFFFFFFFF, ar_send_471, 4);
                            _eh_f[47] = ((ar_bit_117 == 1) ? ar_got_471 : ar_x0_471);
                            _eh_f[63] = ((ar_bit_117 == 1) ? ar_x1_471 : ar_got_471);
                            int ar_bit_118 = lane >> 1 & 1;
                            float ar_x0_472 = _eh_f[35];
                            float ar_x1_472 = _eh_f[43];
                            float ar_send_472 = ((ar_bit_118 == 1) ? ar_x0_472 : ar_x1_472);
                            float ar_got_472 = __shfl_xor_sync(0xFFFFFFFF, ar_send_472, 2);
                            _eh_f[35] = ((ar_bit_118 == 1) ? ar_got_472 : ar_x0_472);
                            _eh_f[43] = ((ar_bit_118 == 1) ? ar_x1_472 : ar_got_472);
                            float ar_x0_473 = _eh_f[39];
                            float ar_x1_473 = _eh_f[47];
                            float ar_send_473 = ((ar_bit_118 == 1) ? ar_x0_473 : ar_x1_473);
                            float ar_got_473 = __shfl_xor_sync(0xFFFFFFFF, ar_send_473, 2);
                            _eh_f[39] = ((ar_bit_118 == 1) ? ar_got_473 : ar_x0_473);
                            _eh_f[47] = ((ar_bit_118 == 1) ? ar_x1_473 : ar_got_473);
                            float ar_x0_474 = _eh_f[51];
                            float ar_x1_474 = _eh_f[59];
                            float ar_send_474 = ((ar_bit_118 == 1) ? ar_x0_474 : ar_x1_474);
                            float ar_got_474 = __shfl_xor_sync(0xFFFFFFFF, ar_send_474, 2);
                            _eh_f[51] = ((ar_bit_118 == 1) ? ar_got_474 : ar_x0_474);
                            _eh_f[59] = ((ar_bit_118 == 1) ? ar_x1_474 : ar_got_474);
                            float ar_x0_475 = _eh_f[55];
                            float ar_x1_475 = _eh_f[63];
                            float ar_send_475 = ((ar_bit_118 == 1) ? ar_x0_475 : ar_x1_475);
                            float ar_got_475 = __shfl_xor_sync(0xFFFFFFFF, ar_send_475, 2);
                            _eh_f[55] = ((ar_bit_118 == 1) ? ar_got_475 : ar_x0_475);
                            _eh_f[63] = ((ar_bit_118 == 1) ? ar_x1_475 : ar_got_475);
                            int ar_bit_119 = lane & 1;
                            float ar_x0_476 = _eh_f[35];
                            float ar_x1_476 = _eh_f[39];
                            float ar_send_476 = ((ar_bit_119 == 1) ? ar_x0_476 : ar_x1_476);
                            float ar_got_476 = __shfl_xor_sync(0xFFFFFFFF, ar_send_476, 1);
                            _eh_f[35] = ((ar_bit_119 == 1) ? ar_got_476 : ar_x0_476);
                            _eh_f[39] = ((ar_bit_119 == 1) ? ar_x1_476 : ar_got_476);
                            float ar_x0_477 = _eh_f[43];
                            float ar_x1_477 = _eh_f[47];
                            float ar_send_477 = ((ar_bit_119 == 1) ? ar_x0_477 : ar_x1_477);
                            float ar_got_477 = __shfl_xor_sync(0xFFFFFFFF, ar_send_477, 1);
                            _eh_f[43] = ((ar_bit_119 == 1) ? ar_got_477 : ar_x0_477);
                            _eh_f[47] = ((ar_bit_119 == 1) ? ar_x1_477 : ar_got_477);
                            float ar_x0_478 = _eh_f[51];
                            float ar_x1_478 = _eh_f[55];
                            float ar_send_478 = ((ar_bit_119 == 1) ? ar_x0_478 : ar_x1_478);
                            float ar_got_478 = __shfl_xor_sync(0xFFFFFFFF, ar_send_478, 1);
                            _eh_f[51] = ((ar_bit_119 == 1) ? ar_got_478 : ar_x0_478);
                            _eh_f[55] = ((ar_bit_119 == 1) ? ar_x1_478 : ar_got_478);
                            float ar_x0_479 = _eh_f[59];
                            float ar_x1_479 = _eh_f[63];
                            float ar_send_479 = ((ar_bit_119 == 1) ? ar_x0_479 : ar_x1_479);
                            float ar_got_479 = __shfl_xor_sync(0xFFFFFFFF, ar_send_479, 1);
                            _eh_f[59] = ((ar_bit_119 == 1) ? ar_got_479 : ar_x0_479);
                            _eh_f[63] = ((ar_bit_119 == 1) ? ar_x1_479 : ar_got_479);
                            int ar_ri_9 = eh_out + (lane & 7) * (4 - ldc) + 32;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_9])), "f"(_eh_f[32]), "f"(_eh_f[33]), "f"(_eh_f[34]), "f"(_eh_f[35]) : "memory");
                            ar_ri_9 = ar_ri_9 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_9])), "f"(_eh_f[36]), "f"(_eh_f[37]), "f"(_eh_f[38]), "f"(_eh_f[39]) : "memory");
                            ar_ri_9 = ar_ri_9 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_9])), "f"(_eh_f[40]), "f"(_eh_f[41]), "f"(_eh_f[42]), "f"(_eh_f[43]) : "memory");
                            ar_ri_9 = ar_ri_9 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_9])), "f"(_eh_f[44]), "f"(_eh_f[45]), "f"(_eh_f[46]), "f"(_eh_f[47]) : "memory");
                            ar_ri_9 = ar_ri_9 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_9])), "f"(_eh_f[48]), "f"(_eh_f[49]), "f"(_eh_f[50]), "f"(_eh_f[51]) : "memory");
                            ar_ri_9 = ar_ri_9 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_9])), "f"(_eh_f[52]), "f"(_eh_f[53]), "f"(_eh_f[54]), "f"(_eh_f[55]) : "memory");
                            ar_ri_9 = ar_ri_9 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_9])), "f"(_eh_f[56]), "f"(_eh_f[57]), "f"(_eh_f[58]), "f"(_eh_f[59]) : "memory");
                            ar_ri_9 = ar_ri_9 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_9])), "f"(_eh_f[60]), "f"(_eh_f[61]), "f"(_eh_f[62]), "f"(_eh_f[63]) : "memory");
                            int ar_bit_120 = lane >> 2 & 1;
                            float ar_x0_480 = _eh_f[64];
                            float ar_x1_480 = _eh_f[80];
                            float ar_send_480 = ((ar_bit_120 == 1) ? ar_x0_480 : ar_x1_480);
                            float ar_got_480 = __shfl_xor_sync(0xFFFFFFFF, ar_send_480, 4);
                            _eh_f[64] = ((ar_bit_120 == 1) ? ar_got_480 : ar_x0_480);
                            _eh_f[80] = ((ar_bit_120 == 1) ? ar_x1_480 : ar_got_480);
                            float ar_x0_481 = _eh_f[68];
                            float ar_x1_481 = _eh_f[84];
                            float ar_send_481 = ((ar_bit_120 == 1) ? ar_x0_481 : ar_x1_481);
                            float ar_got_481 = __shfl_xor_sync(0xFFFFFFFF, ar_send_481, 4);
                            _eh_f[68] = ((ar_bit_120 == 1) ? ar_got_481 : ar_x0_481);
                            _eh_f[84] = ((ar_bit_120 == 1) ? ar_x1_481 : ar_got_481);
                            float ar_x0_482 = _eh_f[72];
                            float ar_x1_482 = _eh_f[88];
                            float ar_send_482 = ((ar_bit_120 == 1) ? ar_x0_482 : ar_x1_482);
                            float ar_got_482 = __shfl_xor_sync(0xFFFFFFFF, ar_send_482, 4);
                            _eh_f[72] = ((ar_bit_120 == 1) ? ar_got_482 : ar_x0_482);
                            _eh_f[88] = ((ar_bit_120 == 1) ? ar_x1_482 : ar_got_482);
                            float ar_x0_483 = _eh_f[76];
                            float ar_x1_483 = _eh_f[92];
                            float ar_send_483 = ((ar_bit_120 == 1) ? ar_x0_483 : ar_x1_483);
                            float ar_got_483 = __shfl_xor_sync(0xFFFFFFFF, ar_send_483, 4);
                            _eh_f[76] = ((ar_bit_120 == 1) ? ar_got_483 : ar_x0_483);
                            _eh_f[92] = ((ar_bit_120 == 1) ? ar_x1_483 : ar_got_483);
                            int ar_bit_121 = lane >> 1 & 1;
                            float ar_x0_484 = _eh_f[64];
                            float ar_x1_484 = _eh_f[72];
                            float ar_send_484 = ((ar_bit_121 == 1) ? ar_x0_484 : ar_x1_484);
                            float ar_got_484 = __shfl_xor_sync(0xFFFFFFFF, ar_send_484, 2);
                            _eh_f[64] = ((ar_bit_121 == 1) ? ar_got_484 : ar_x0_484);
                            _eh_f[72] = ((ar_bit_121 == 1) ? ar_x1_484 : ar_got_484);
                            float ar_x0_485 = _eh_f[68];
                            float ar_x1_485 = _eh_f[76];
                            float ar_send_485 = ((ar_bit_121 == 1) ? ar_x0_485 : ar_x1_485);
                            float ar_got_485 = __shfl_xor_sync(0xFFFFFFFF, ar_send_485, 2);
                            _eh_f[68] = ((ar_bit_121 == 1) ? ar_got_485 : ar_x0_485);
                            _eh_f[76] = ((ar_bit_121 == 1) ? ar_x1_485 : ar_got_485);
                            float ar_x0_486 = _eh_f[80];
                            float ar_x1_486 = _eh_f[88];
                            float ar_send_486 = ((ar_bit_121 == 1) ? ar_x0_486 : ar_x1_486);
                            float ar_got_486 = __shfl_xor_sync(0xFFFFFFFF, ar_send_486, 2);
                            _eh_f[80] = ((ar_bit_121 == 1) ? ar_got_486 : ar_x0_486);
                            _eh_f[88] = ((ar_bit_121 == 1) ? ar_x1_486 : ar_got_486);
                            float ar_x0_487 = _eh_f[84];
                            float ar_x1_487 = _eh_f[92];
                            float ar_send_487 = ((ar_bit_121 == 1) ? ar_x0_487 : ar_x1_487);
                            float ar_got_487 = __shfl_xor_sync(0xFFFFFFFF, ar_send_487, 2);
                            _eh_f[84] = ((ar_bit_121 == 1) ? ar_got_487 : ar_x0_487);
                            _eh_f[92] = ((ar_bit_121 == 1) ? ar_x1_487 : ar_got_487);
                            int ar_bit_122 = lane & 1;
                            float ar_x0_488 = _eh_f[64];
                            float ar_x1_488 = _eh_f[68];
                            float ar_send_488 = ((ar_bit_122 == 1) ? ar_x0_488 : ar_x1_488);
                            float ar_got_488 = __shfl_xor_sync(0xFFFFFFFF, ar_send_488, 1);
                            _eh_f[64] = ((ar_bit_122 == 1) ? ar_got_488 : ar_x0_488);
                            _eh_f[68] = ((ar_bit_122 == 1) ? ar_x1_488 : ar_got_488);
                            float ar_x0_489 = _eh_f[72];
                            float ar_x1_489 = _eh_f[76];
                            float ar_send_489 = ((ar_bit_122 == 1) ? ar_x0_489 : ar_x1_489);
                            float ar_got_489 = __shfl_xor_sync(0xFFFFFFFF, ar_send_489, 1);
                            _eh_f[72] = ((ar_bit_122 == 1) ? ar_got_489 : ar_x0_489);
                            _eh_f[76] = ((ar_bit_122 == 1) ? ar_x1_489 : ar_got_489);
                            float ar_x0_490 = _eh_f[80];
                            float ar_x1_490 = _eh_f[84];
                            float ar_send_490 = ((ar_bit_122 == 1) ? ar_x0_490 : ar_x1_490);
                            float ar_got_490 = __shfl_xor_sync(0xFFFFFFFF, ar_send_490, 1);
                            _eh_f[80] = ((ar_bit_122 == 1) ? ar_got_490 : ar_x0_490);
                            _eh_f[84] = ((ar_bit_122 == 1) ? ar_x1_490 : ar_got_490);
                            float ar_x0_491 = _eh_f[88];
                            float ar_x1_491 = _eh_f[92];
                            float ar_send_491 = ((ar_bit_122 == 1) ? ar_x0_491 : ar_x1_491);
                            float ar_got_491 = __shfl_xor_sync(0xFFFFFFFF, ar_send_491, 1);
                            _eh_f[88] = ((ar_bit_122 == 1) ? ar_got_491 : ar_x0_491);
                            _eh_f[92] = ((ar_bit_122 == 1) ? ar_x1_491 : ar_got_491);
                            int ar_bit_123 = lane >> 2 & 1;
                            float ar_x0_492 = _eh_f[65];
                            float ar_x1_492 = _eh_f[81];
                            float ar_send_492 = ((ar_bit_123 == 1) ? ar_x0_492 : ar_x1_492);
                            float ar_got_492 = __shfl_xor_sync(0xFFFFFFFF, ar_send_492, 4);
                            _eh_f[65] = ((ar_bit_123 == 1) ? ar_got_492 : ar_x0_492);
                            _eh_f[81] = ((ar_bit_123 == 1) ? ar_x1_492 : ar_got_492);
                            float ar_x0_493 = _eh_f[69];
                            float ar_x1_493 = _eh_f[85];
                            float ar_send_493 = ((ar_bit_123 == 1) ? ar_x0_493 : ar_x1_493);
                            float ar_got_493 = __shfl_xor_sync(0xFFFFFFFF, ar_send_493, 4);
                            _eh_f[69] = ((ar_bit_123 == 1) ? ar_got_493 : ar_x0_493);
                            _eh_f[85] = ((ar_bit_123 == 1) ? ar_x1_493 : ar_got_493);
                            float ar_x0_494 = _eh_f[73];
                            float ar_x1_494 = _eh_f[89];
                            float ar_send_494 = ((ar_bit_123 == 1) ? ar_x0_494 : ar_x1_494);
                            float ar_got_494 = __shfl_xor_sync(0xFFFFFFFF, ar_send_494, 4);
                            _eh_f[73] = ((ar_bit_123 == 1) ? ar_got_494 : ar_x0_494);
                            _eh_f[89] = ((ar_bit_123 == 1) ? ar_x1_494 : ar_got_494);
                            float ar_x0_495 = _eh_f[77];
                            float ar_x1_495 = _eh_f[93];
                            float ar_send_495 = ((ar_bit_123 == 1) ? ar_x0_495 : ar_x1_495);
                            float ar_got_495 = __shfl_xor_sync(0xFFFFFFFF, ar_send_495, 4);
                            _eh_f[77] = ((ar_bit_123 == 1) ? ar_got_495 : ar_x0_495);
                            _eh_f[93] = ((ar_bit_123 == 1) ? ar_x1_495 : ar_got_495);
                            int ar_bit_124 = lane >> 1 & 1;
                            float ar_x0_496 = _eh_f[65];
                            float ar_x1_496 = _eh_f[73];
                            float ar_send_496 = ((ar_bit_124 == 1) ? ar_x0_496 : ar_x1_496);
                            float ar_got_496 = __shfl_xor_sync(0xFFFFFFFF, ar_send_496, 2);
                            _eh_f[65] = ((ar_bit_124 == 1) ? ar_got_496 : ar_x0_496);
                            _eh_f[73] = ((ar_bit_124 == 1) ? ar_x1_496 : ar_got_496);
                            float ar_x0_497 = _eh_f[69];
                            float ar_x1_497 = _eh_f[77];
                            float ar_send_497 = ((ar_bit_124 == 1) ? ar_x0_497 : ar_x1_497);
                            float ar_got_497 = __shfl_xor_sync(0xFFFFFFFF, ar_send_497, 2);
                            _eh_f[69] = ((ar_bit_124 == 1) ? ar_got_497 : ar_x0_497);
                            _eh_f[77] = ((ar_bit_124 == 1) ? ar_x1_497 : ar_got_497);
                            float ar_x0_498 = _eh_f[81];
                            float ar_x1_498 = _eh_f[89];
                            float ar_send_498 = ((ar_bit_124 == 1) ? ar_x0_498 : ar_x1_498);
                            float ar_got_498 = __shfl_xor_sync(0xFFFFFFFF, ar_send_498, 2);
                            _eh_f[81] = ((ar_bit_124 == 1) ? ar_got_498 : ar_x0_498);
                            _eh_f[89] = ((ar_bit_124 == 1) ? ar_x1_498 : ar_got_498);
                            float ar_x0_499 = _eh_f[85];
                            float ar_x1_499 = _eh_f[93];
                            float ar_send_499 = ((ar_bit_124 == 1) ? ar_x0_499 : ar_x1_499);
                            float ar_got_499 = __shfl_xor_sync(0xFFFFFFFF, ar_send_499, 2);
                            _eh_f[85] = ((ar_bit_124 == 1) ? ar_got_499 : ar_x0_499);
                            _eh_f[93] = ((ar_bit_124 == 1) ? ar_x1_499 : ar_got_499);
                            int ar_bit_125 = lane & 1;
                            float ar_x0_500 = _eh_f[65];
                            float ar_x1_500 = _eh_f[69];
                            float ar_send_500 = ((ar_bit_125 == 1) ? ar_x0_500 : ar_x1_500);
                            float ar_got_500 = __shfl_xor_sync(0xFFFFFFFF, ar_send_500, 1);
                            _eh_f[65] = ((ar_bit_125 == 1) ? ar_got_500 : ar_x0_500);
                            _eh_f[69] = ((ar_bit_125 == 1) ? ar_x1_500 : ar_got_500);
                            float ar_x0_501 = _eh_f[73];
                            float ar_x1_501 = _eh_f[77];
                            float ar_send_501 = ((ar_bit_125 == 1) ? ar_x0_501 : ar_x1_501);
                            float ar_got_501 = __shfl_xor_sync(0xFFFFFFFF, ar_send_501, 1);
                            _eh_f[73] = ((ar_bit_125 == 1) ? ar_got_501 : ar_x0_501);
                            _eh_f[77] = ((ar_bit_125 == 1) ? ar_x1_501 : ar_got_501);
                            float ar_x0_502 = _eh_f[81];
                            float ar_x1_502 = _eh_f[85];
                            float ar_send_502 = ((ar_bit_125 == 1) ? ar_x0_502 : ar_x1_502);
                            float ar_got_502 = __shfl_xor_sync(0xFFFFFFFF, ar_send_502, 1);
                            _eh_f[81] = ((ar_bit_125 == 1) ? ar_got_502 : ar_x0_502);
                            _eh_f[85] = ((ar_bit_125 == 1) ? ar_x1_502 : ar_got_502);
                            float ar_x0_503 = _eh_f[89];
                            float ar_x1_503 = _eh_f[93];
                            float ar_send_503 = ((ar_bit_125 == 1) ? ar_x0_503 : ar_x1_503);
                            float ar_got_503 = __shfl_xor_sync(0xFFFFFFFF, ar_send_503, 1);
                            _eh_f[89] = ((ar_bit_125 == 1) ? ar_got_503 : ar_x0_503);
                            _eh_f[93] = ((ar_bit_125 == 1) ? ar_x1_503 : ar_got_503);
                            int ar_bit_126 = lane >> 2 & 1;
                            float ar_x0_504 = _eh_f[66];
                            float ar_x1_504 = _eh_f[82];
                            float ar_send_504 = ((ar_bit_126 == 1) ? ar_x0_504 : ar_x1_504);
                            float ar_got_504 = __shfl_xor_sync(0xFFFFFFFF, ar_send_504, 4);
                            _eh_f[66] = ((ar_bit_126 == 1) ? ar_got_504 : ar_x0_504);
                            _eh_f[82] = ((ar_bit_126 == 1) ? ar_x1_504 : ar_got_504);
                            float ar_x0_505 = _eh_f[70];
                            float ar_x1_505 = _eh_f[86];
                            float ar_send_505 = ((ar_bit_126 == 1) ? ar_x0_505 : ar_x1_505);
                            float ar_got_505 = __shfl_xor_sync(0xFFFFFFFF, ar_send_505, 4);
                            _eh_f[70] = ((ar_bit_126 == 1) ? ar_got_505 : ar_x0_505);
                            _eh_f[86] = ((ar_bit_126 == 1) ? ar_x1_505 : ar_got_505);
                            float ar_x0_506 = _eh_f[74];
                            float ar_x1_506 = _eh_f[90];
                            float ar_send_506 = ((ar_bit_126 == 1) ? ar_x0_506 : ar_x1_506);
                            float ar_got_506 = __shfl_xor_sync(0xFFFFFFFF, ar_send_506, 4);
                            _eh_f[74] = ((ar_bit_126 == 1) ? ar_got_506 : ar_x0_506);
                            _eh_f[90] = ((ar_bit_126 == 1) ? ar_x1_506 : ar_got_506);
                            float ar_x0_507 = _eh_f[78];
                            float ar_x1_507 = _eh_f[94];
                            float ar_send_507 = ((ar_bit_126 == 1) ? ar_x0_507 : ar_x1_507);
                            float ar_got_507 = __shfl_xor_sync(0xFFFFFFFF, ar_send_507, 4);
                            _eh_f[78] = ((ar_bit_126 == 1) ? ar_got_507 : ar_x0_507);
                            _eh_f[94] = ((ar_bit_126 == 1) ? ar_x1_507 : ar_got_507);
                            int ar_bit_127 = lane >> 1 & 1;
                            float ar_x0_508 = _eh_f[66];
                            float ar_x1_508 = _eh_f[74];
                            float ar_send_508 = ((ar_bit_127 == 1) ? ar_x0_508 : ar_x1_508);
                            float ar_got_508 = __shfl_xor_sync(0xFFFFFFFF, ar_send_508, 2);
                            _eh_f[66] = ((ar_bit_127 == 1) ? ar_got_508 : ar_x0_508);
                            _eh_f[74] = ((ar_bit_127 == 1) ? ar_x1_508 : ar_got_508);
                            float ar_x0_509 = _eh_f[70];
                            float ar_x1_509 = _eh_f[78];
                            float ar_send_509 = ((ar_bit_127 == 1) ? ar_x0_509 : ar_x1_509);
                            float ar_got_509 = __shfl_xor_sync(0xFFFFFFFF, ar_send_509, 2);
                            _eh_f[70] = ((ar_bit_127 == 1) ? ar_got_509 : ar_x0_509);
                            _eh_f[78] = ((ar_bit_127 == 1) ? ar_x1_509 : ar_got_509);
                            float ar_x0_510 = _eh_f[82];
                            float ar_x1_510 = _eh_f[90];
                            float ar_send_510 = ((ar_bit_127 == 1) ? ar_x0_510 : ar_x1_510);
                            float ar_got_510 = __shfl_xor_sync(0xFFFFFFFF, ar_send_510, 2);
                            _eh_f[82] = ((ar_bit_127 == 1) ? ar_got_510 : ar_x0_510);
                            _eh_f[90] = ((ar_bit_127 == 1) ? ar_x1_510 : ar_got_510);
                            float ar_x0_511 = _eh_f[86];
                            float ar_x1_511 = _eh_f[94];
                            float ar_send_511 = ((ar_bit_127 == 1) ? ar_x0_511 : ar_x1_511);
                            float ar_got_511 = __shfl_xor_sync(0xFFFFFFFF, ar_send_511, 2);
                            _eh_f[86] = ((ar_bit_127 == 1) ? ar_got_511 : ar_x0_511);
                            _eh_f[94] = ((ar_bit_127 == 1) ? ar_x1_511 : ar_got_511);
                            int ar_bit_128 = lane & 1;
                            float ar_x0_512 = _eh_f[66];
                            float ar_x1_512 = _eh_f[70];
                            float ar_send_512 = ((ar_bit_128 == 1) ? ar_x0_512 : ar_x1_512);
                            float ar_got_512 = __shfl_xor_sync(0xFFFFFFFF, ar_send_512, 1);
                            _eh_f[66] = ((ar_bit_128 == 1) ? ar_got_512 : ar_x0_512);
                            _eh_f[70] = ((ar_bit_128 == 1) ? ar_x1_512 : ar_got_512);
                            float ar_x0_513 = _eh_f[74];
                            float ar_x1_513 = _eh_f[78];
                            float ar_send_513 = ((ar_bit_128 == 1) ? ar_x0_513 : ar_x1_513);
                            float ar_got_513 = __shfl_xor_sync(0xFFFFFFFF, ar_send_513, 1);
                            _eh_f[74] = ((ar_bit_128 == 1) ? ar_got_513 : ar_x0_513);
                            _eh_f[78] = ((ar_bit_128 == 1) ? ar_x1_513 : ar_got_513);
                            float ar_x0_514 = _eh_f[82];
                            float ar_x1_514 = _eh_f[86];
                            float ar_send_514 = ((ar_bit_128 == 1) ? ar_x0_514 : ar_x1_514);
                            float ar_got_514 = __shfl_xor_sync(0xFFFFFFFF, ar_send_514, 1);
                            _eh_f[82] = ((ar_bit_128 == 1) ? ar_got_514 : ar_x0_514);
                            _eh_f[86] = ((ar_bit_128 == 1) ? ar_x1_514 : ar_got_514);
                            float ar_x0_515 = _eh_f[90];
                            float ar_x1_515 = _eh_f[94];
                            float ar_send_515 = ((ar_bit_128 == 1) ? ar_x0_515 : ar_x1_515);
                            float ar_got_515 = __shfl_xor_sync(0xFFFFFFFF, ar_send_515, 1);
                            _eh_f[90] = ((ar_bit_128 == 1) ? ar_got_515 : ar_x0_515);
                            _eh_f[94] = ((ar_bit_128 == 1) ? ar_x1_515 : ar_got_515);
                            int ar_bit_129 = lane >> 2 & 1;
                            float ar_x0_516 = _eh_f[67];
                            float ar_x1_516 = _eh_f[83];
                            float ar_send_516 = ((ar_bit_129 == 1) ? ar_x0_516 : ar_x1_516);
                            float ar_got_516 = __shfl_xor_sync(0xFFFFFFFF, ar_send_516, 4);
                            _eh_f[67] = ((ar_bit_129 == 1) ? ar_got_516 : ar_x0_516);
                            _eh_f[83] = ((ar_bit_129 == 1) ? ar_x1_516 : ar_got_516);
                            float ar_x0_517 = _eh_f[71];
                            float ar_x1_517 = _eh_f[87];
                            float ar_send_517 = ((ar_bit_129 == 1) ? ar_x0_517 : ar_x1_517);
                            float ar_got_517 = __shfl_xor_sync(0xFFFFFFFF, ar_send_517, 4);
                            _eh_f[71] = ((ar_bit_129 == 1) ? ar_got_517 : ar_x0_517);
                            _eh_f[87] = ((ar_bit_129 == 1) ? ar_x1_517 : ar_got_517);
                            float ar_x0_518 = _eh_f[75];
                            float ar_x1_518 = _eh_f[91];
                            float ar_send_518 = ((ar_bit_129 == 1) ? ar_x0_518 : ar_x1_518);
                            float ar_got_518 = __shfl_xor_sync(0xFFFFFFFF, ar_send_518, 4);
                            _eh_f[75] = ((ar_bit_129 == 1) ? ar_got_518 : ar_x0_518);
                            _eh_f[91] = ((ar_bit_129 == 1) ? ar_x1_518 : ar_got_518);
                            float ar_x0_519 = _eh_f[79];
                            float ar_x1_519 = _eh_f[95];
                            float ar_send_519 = ((ar_bit_129 == 1) ? ar_x0_519 : ar_x1_519);
                            float ar_got_519 = __shfl_xor_sync(0xFFFFFFFF, ar_send_519, 4);
                            _eh_f[79] = ((ar_bit_129 == 1) ? ar_got_519 : ar_x0_519);
                            _eh_f[95] = ((ar_bit_129 == 1) ? ar_x1_519 : ar_got_519);
                            int ar_bit_130 = lane >> 1 & 1;
                            float ar_x0_520 = _eh_f[67];
                            float ar_x1_520 = _eh_f[75];
                            float ar_send_520 = ((ar_bit_130 == 1) ? ar_x0_520 : ar_x1_520);
                            float ar_got_520 = __shfl_xor_sync(0xFFFFFFFF, ar_send_520, 2);
                            _eh_f[67] = ((ar_bit_130 == 1) ? ar_got_520 : ar_x0_520);
                            _eh_f[75] = ((ar_bit_130 == 1) ? ar_x1_520 : ar_got_520);
                            float ar_x0_521 = _eh_f[71];
                            float ar_x1_521 = _eh_f[79];
                            float ar_send_521 = ((ar_bit_130 == 1) ? ar_x0_521 : ar_x1_521);
                            float ar_got_521 = __shfl_xor_sync(0xFFFFFFFF, ar_send_521, 2);
                            _eh_f[71] = ((ar_bit_130 == 1) ? ar_got_521 : ar_x0_521);
                            _eh_f[79] = ((ar_bit_130 == 1) ? ar_x1_521 : ar_got_521);
                            float ar_x0_522 = _eh_f[83];
                            float ar_x1_522 = _eh_f[91];
                            float ar_send_522 = ((ar_bit_130 == 1) ? ar_x0_522 : ar_x1_522);
                            float ar_got_522 = __shfl_xor_sync(0xFFFFFFFF, ar_send_522, 2);
                            _eh_f[83] = ((ar_bit_130 == 1) ? ar_got_522 : ar_x0_522);
                            _eh_f[91] = ((ar_bit_130 == 1) ? ar_x1_522 : ar_got_522);
                            float ar_x0_523 = _eh_f[87];
                            float ar_x1_523 = _eh_f[95];
                            float ar_send_523 = ((ar_bit_130 == 1) ? ar_x0_523 : ar_x1_523);
                            float ar_got_523 = __shfl_xor_sync(0xFFFFFFFF, ar_send_523, 2);
                            _eh_f[87] = ((ar_bit_130 == 1) ? ar_got_523 : ar_x0_523);
                            _eh_f[95] = ((ar_bit_130 == 1) ? ar_x1_523 : ar_got_523);
                            int ar_bit_131 = lane & 1;
                            float ar_x0_524 = _eh_f[67];
                            float ar_x1_524 = _eh_f[71];
                            float ar_send_524 = ((ar_bit_131 == 1) ? ar_x0_524 : ar_x1_524);
                            float ar_got_524 = __shfl_xor_sync(0xFFFFFFFF, ar_send_524, 1);
                            _eh_f[67] = ((ar_bit_131 == 1) ? ar_got_524 : ar_x0_524);
                            _eh_f[71] = ((ar_bit_131 == 1) ? ar_x1_524 : ar_got_524);
                            float ar_x0_525 = _eh_f[75];
                            float ar_x1_525 = _eh_f[79];
                            float ar_send_525 = ((ar_bit_131 == 1) ? ar_x0_525 : ar_x1_525);
                            float ar_got_525 = __shfl_xor_sync(0xFFFFFFFF, ar_send_525, 1);
                            _eh_f[75] = ((ar_bit_131 == 1) ? ar_got_525 : ar_x0_525);
                            _eh_f[79] = ((ar_bit_131 == 1) ? ar_x1_525 : ar_got_525);
                            float ar_x0_526 = _eh_f[83];
                            float ar_x1_526 = _eh_f[87];
                            float ar_send_526 = ((ar_bit_131 == 1) ? ar_x0_526 : ar_x1_526);
                            float ar_got_526 = __shfl_xor_sync(0xFFFFFFFF, ar_send_526, 1);
                            _eh_f[83] = ((ar_bit_131 == 1) ? ar_got_526 : ar_x0_526);
                            _eh_f[87] = ((ar_bit_131 == 1) ? ar_x1_526 : ar_got_526);
                            float ar_x0_527 = _eh_f[91];
                            float ar_x1_527 = _eh_f[95];
                            float ar_send_527 = ((ar_bit_131 == 1) ? ar_x0_527 : ar_x1_527);
                            float ar_got_527 = __shfl_xor_sync(0xFFFFFFFF, ar_send_527, 1);
                            _eh_f[91] = ((ar_bit_131 == 1) ? ar_got_527 : ar_x0_527);
                            _eh_f[95] = ((ar_bit_131 == 1) ? ar_x1_527 : ar_got_527);
                            int ar_ri_10 = eh_out + (lane & 7) * (4 - ldc) + 64;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_10])), "f"(_eh_f[64]), "f"(_eh_f[65]), "f"(_eh_f[66]), "f"(_eh_f[67]) : "memory");
                            ar_ri_10 = ar_ri_10 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_10])), "f"(_eh_f[68]), "f"(_eh_f[69]), "f"(_eh_f[70]), "f"(_eh_f[71]) : "memory");
                            ar_ri_10 = ar_ri_10 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_10])), "f"(_eh_f[72]), "f"(_eh_f[73]), "f"(_eh_f[74]), "f"(_eh_f[75]) : "memory");
                            ar_ri_10 = ar_ri_10 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_10])), "f"(_eh_f[76]), "f"(_eh_f[77]), "f"(_eh_f[78]), "f"(_eh_f[79]) : "memory");
                            ar_ri_10 = ar_ri_10 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_10])), "f"(_eh_f[80]), "f"(_eh_f[81]), "f"(_eh_f[82]), "f"(_eh_f[83]) : "memory");
                            ar_ri_10 = ar_ri_10 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_10])), "f"(_eh_f[84]), "f"(_eh_f[85]), "f"(_eh_f[86]), "f"(_eh_f[87]) : "memory");
                            ar_ri_10 = ar_ri_10 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_10])), "f"(_eh_f[88]), "f"(_eh_f[89]), "f"(_eh_f[90]), "f"(_eh_f[91]) : "memory");
                            ar_ri_10 = ar_ri_10 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_10])), "f"(_eh_f[92]), "f"(_eh_f[93]), "f"(_eh_f[94]), "f"(_eh_f[95]) : "memory");
                            int ar_bit_132 = lane >> 2 & 1;
                            float ar_x0_528 = _eh_f[96];
                            float ar_x1_528 = _eh_f[112];
                            float ar_send_528 = ((ar_bit_132 == 1) ? ar_x0_528 : ar_x1_528);
                            float ar_got_528 = __shfl_xor_sync(0xFFFFFFFF, ar_send_528, 4);
                            _eh_f[96] = ((ar_bit_132 == 1) ? ar_got_528 : ar_x0_528);
                            _eh_f[112] = ((ar_bit_132 == 1) ? ar_x1_528 : ar_got_528);
                            float ar_x0_529 = _eh_f[100];
                            float ar_x1_529 = _eh_f[116];
                            float ar_send_529 = ((ar_bit_132 == 1) ? ar_x0_529 : ar_x1_529);
                            float ar_got_529 = __shfl_xor_sync(0xFFFFFFFF, ar_send_529, 4);
                            _eh_f[100] = ((ar_bit_132 == 1) ? ar_got_529 : ar_x0_529);
                            _eh_f[116] = ((ar_bit_132 == 1) ? ar_x1_529 : ar_got_529);
                            float ar_x0_530 = _eh_f[104];
                            float ar_x1_530 = _eh_f[120];
                            float ar_send_530 = ((ar_bit_132 == 1) ? ar_x0_530 : ar_x1_530);
                            float ar_got_530 = __shfl_xor_sync(0xFFFFFFFF, ar_send_530, 4);
                            _eh_f[104] = ((ar_bit_132 == 1) ? ar_got_530 : ar_x0_530);
                            _eh_f[120] = ((ar_bit_132 == 1) ? ar_x1_530 : ar_got_530);
                            float ar_x0_531 = _eh_f[108];
                            float ar_x1_531 = _eh_f[124];
                            float ar_send_531 = ((ar_bit_132 == 1) ? ar_x0_531 : ar_x1_531);
                            float ar_got_531 = __shfl_xor_sync(0xFFFFFFFF, ar_send_531, 4);
                            _eh_f[108] = ((ar_bit_132 == 1) ? ar_got_531 : ar_x0_531);
                            _eh_f[124] = ((ar_bit_132 == 1) ? ar_x1_531 : ar_got_531);
                            int ar_bit_133 = lane >> 1 & 1;
                            float ar_x0_532 = _eh_f[96];
                            float ar_x1_532 = _eh_f[104];
                            float ar_send_532 = ((ar_bit_133 == 1) ? ar_x0_532 : ar_x1_532);
                            float ar_got_532 = __shfl_xor_sync(0xFFFFFFFF, ar_send_532, 2);
                            _eh_f[96] = ((ar_bit_133 == 1) ? ar_got_532 : ar_x0_532);
                            _eh_f[104] = ((ar_bit_133 == 1) ? ar_x1_532 : ar_got_532);
                            float ar_x0_533 = _eh_f[100];
                            float ar_x1_533 = _eh_f[108];
                            float ar_send_533 = ((ar_bit_133 == 1) ? ar_x0_533 : ar_x1_533);
                            float ar_got_533 = __shfl_xor_sync(0xFFFFFFFF, ar_send_533, 2);
                            _eh_f[100] = ((ar_bit_133 == 1) ? ar_got_533 : ar_x0_533);
                            _eh_f[108] = ((ar_bit_133 == 1) ? ar_x1_533 : ar_got_533);
                            float ar_x0_534 = _eh_f[112];
                            float ar_x1_534 = _eh_f[120];
                            float ar_send_534 = ((ar_bit_133 == 1) ? ar_x0_534 : ar_x1_534);
                            float ar_got_534 = __shfl_xor_sync(0xFFFFFFFF, ar_send_534, 2);
                            _eh_f[112] = ((ar_bit_133 == 1) ? ar_got_534 : ar_x0_534);
                            _eh_f[120] = ((ar_bit_133 == 1) ? ar_x1_534 : ar_got_534);
                            float ar_x0_535 = _eh_f[116];
                            float ar_x1_535 = _eh_f[124];
                            float ar_send_535 = ((ar_bit_133 == 1) ? ar_x0_535 : ar_x1_535);
                            float ar_got_535 = __shfl_xor_sync(0xFFFFFFFF, ar_send_535, 2);
                            _eh_f[116] = ((ar_bit_133 == 1) ? ar_got_535 : ar_x0_535);
                            _eh_f[124] = ((ar_bit_133 == 1) ? ar_x1_535 : ar_got_535);
                            int ar_bit_134 = lane & 1;
                            float ar_x0_536 = _eh_f[96];
                            float ar_x1_536 = _eh_f[100];
                            float ar_send_536 = ((ar_bit_134 == 1) ? ar_x0_536 : ar_x1_536);
                            float ar_got_536 = __shfl_xor_sync(0xFFFFFFFF, ar_send_536, 1);
                            _eh_f[96] = ((ar_bit_134 == 1) ? ar_got_536 : ar_x0_536);
                            _eh_f[100] = ((ar_bit_134 == 1) ? ar_x1_536 : ar_got_536);
                            float ar_x0_537 = _eh_f[104];
                            float ar_x1_537 = _eh_f[108];
                            float ar_send_537 = ((ar_bit_134 == 1) ? ar_x0_537 : ar_x1_537);
                            float ar_got_537 = __shfl_xor_sync(0xFFFFFFFF, ar_send_537, 1);
                            _eh_f[104] = ((ar_bit_134 == 1) ? ar_got_537 : ar_x0_537);
                            _eh_f[108] = ((ar_bit_134 == 1) ? ar_x1_537 : ar_got_537);
                            float ar_x0_538 = _eh_f[112];
                            float ar_x1_538 = _eh_f[116];
                            float ar_send_538 = ((ar_bit_134 == 1) ? ar_x0_538 : ar_x1_538);
                            float ar_got_538 = __shfl_xor_sync(0xFFFFFFFF, ar_send_538, 1);
                            _eh_f[112] = ((ar_bit_134 == 1) ? ar_got_538 : ar_x0_538);
                            _eh_f[116] = ((ar_bit_134 == 1) ? ar_x1_538 : ar_got_538);
                            float ar_x0_539 = _eh_f[120];
                            float ar_x1_539 = _eh_f[124];
                            float ar_send_539 = ((ar_bit_134 == 1) ? ar_x0_539 : ar_x1_539);
                            float ar_got_539 = __shfl_xor_sync(0xFFFFFFFF, ar_send_539, 1);
                            _eh_f[120] = ((ar_bit_134 == 1) ? ar_got_539 : ar_x0_539);
                            _eh_f[124] = ((ar_bit_134 == 1) ? ar_x1_539 : ar_got_539);
                            int ar_bit_135 = lane >> 2 & 1;
                            float ar_x0_540 = _eh_f[97];
                            float ar_x1_540 = _eh_f[113];
                            float ar_send_540 = ((ar_bit_135 == 1) ? ar_x0_540 : ar_x1_540);
                            float ar_got_540 = __shfl_xor_sync(0xFFFFFFFF, ar_send_540, 4);
                            _eh_f[97] = ((ar_bit_135 == 1) ? ar_got_540 : ar_x0_540);
                            _eh_f[113] = ((ar_bit_135 == 1) ? ar_x1_540 : ar_got_540);
                            float ar_x0_541 = _eh_f[101];
                            float ar_x1_541 = _eh_f[117];
                            float ar_send_541 = ((ar_bit_135 == 1) ? ar_x0_541 : ar_x1_541);
                            float ar_got_541 = __shfl_xor_sync(0xFFFFFFFF, ar_send_541, 4);
                            _eh_f[101] = ((ar_bit_135 == 1) ? ar_got_541 : ar_x0_541);
                            _eh_f[117] = ((ar_bit_135 == 1) ? ar_x1_541 : ar_got_541);
                            float ar_x0_542 = _eh_f[105];
                            float ar_x1_542 = _eh_f[121];
                            float ar_send_542 = ((ar_bit_135 == 1) ? ar_x0_542 : ar_x1_542);
                            float ar_got_542 = __shfl_xor_sync(0xFFFFFFFF, ar_send_542, 4);
                            _eh_f[105] = ((ar_bit_135 == 1) ? ar_got_542 : ar_x0_542);
                            _eh_f[121] = ((ar_bit_135 == 1) ? ar_x1_542 : ar_got_542);
                            float ar_x0_543 = _eh_f[109];
                            float ar_x1_543 = _eh_f[125];
                            float ar_send_543 = ((ar_bit_135 == 1) ? ar_x0_543 : ar_x1_543);
                            float ar_got_543 = __shfl_xor_sync(0xFFFFFFFF, ar_send_543, 4);
                            _eh_f[109] = ((ar_bit_135 == 1) ? ar_got_543 : ar_x0_543);
                            _eh_f[125] = ((ar_bit_135 == 1) ? ar_x1_543 : ar_got_543);
                            int ar_bit_136 = lane >> 1 & 1;
                            float ar_x0_544 = _eh_f[97];
                            float ar_x1_544 = _eh_f[105];
                            float ar_send_544 = ((ar_bit_136 == 1) ? ar_x0_544 : ar_x1_544);
                            float ar_got_544 = __shfl_xor_sync(0xFFFFFFFF, ar_send_544, 2);
                            _eh_f[97] = ((ar_bit_136 == 1) ? ar_got_544 : ar_x0_544);
                            _eh_f[105] = ((ar_bit_136 == 1) ? ar_x1_544 : ar_got_544);
                            float ar_x0_545 = _eh_f[101];
                            float ar_x1_545 = _eh_f[109];
                            float ar_send_545 = ((ar_bit_136 == 1) ? ar_x0_545 : ar_x1_545);
                            float ar_got_545 = __shfl_xor_sync(0xFFFFFFFF, ar_send_545, 2);
                            _eh_f[101] = ((ar_bit_136 == 1) ? ar_got_545 : ar_x0_545);
                            _eh_f[109] = ((ar_bit_136 == 1) ? ar_x1_545 : ar_got_545);
                            float ar_x0_546 = _eh_f[113];
                            float ar_x1_546 = _eh_f[121];
                            float ar_send_546 = ((ar_bit_136 == 1) ? ar_x0_546 : ar_x1_546);
                            float ar_got_546 = __shfl_xor_sync(0xFFFFFFFF, ar_send_546, 2);
                            _eh_f[113] = ((ar_bit_136 == 1) ? ar_got_546 : ar_x0_546);
                            _eh_f[121] = ((ar_bit_136 == 1) ? ar_x1_546 : ar_got_546);
                            float ar_x0_547 = _eh_f[117];
                            float ar_x1_547 = _eh_f[125];
                            float ar_send_547 = ((ar_bit_136 == 1) ? ar_x0_547 : ar_x1_547);
                            float ar_got_547 = __shfl_xor_sync(0xFFFFFFFF, ar_send_547, 2);
                            _eh_f[117] = ((ar_bit_136 == 1) ? ar_got_547 : ar_x0_547);
                            _eh_f[125] = ((ar_bit_136 == 1) ? ar_x1_547 : ar_got_547);
                            int ar_bit_137 = lane & 1;
                            float ar_x0_548 = _eh_f[97];
                            float ar_x1_548 = _eh_f[101];
                            float ar_send_548 = ((ar_bit_137 == 1) ? ar_x0_548 : ar_x1_548);
                            float ar_got_548 = __shfl_xor_sync(0xFFFFFFFF, ar_send_548, 1);
                            _eh_f[97] = ((ar_bit_137 == 1) ? ar_got_548 : ar_x0_548);
                            _eh_f[101] = ((ar_bit_137 == 1) ? ar_x1_548 : ar_got_548);
                            float ar_x0_549 = _eh_f[105];
                            float ar_x1_549 = _eh_f[109];
                            float ar_send_549 = ((ar_bit_137 == 1) ? ar_x0_549 : ar_x1_549);
                            float ar_got_549 = __shfl_xor_sync(0xFFFFFFFF, ar_send_549, 1);
                            _eh_f[105] = ((ar_bit_137 == 1) ? ar_got_549 : ar_x0_549);
                            _eh_f[109] = ((ar_bit_137 == 1) ? ar_x1_549 : ar_got_549);
                            float ar_x0_550 = _eh_f[113];
                            float ar_x1_550 = _eh_f[117];
                            float ar_send_550 = ((ar_bit_137 == 1) ? ar_x0_550 : ar_x1_550);
                            float ar_got_550 = __shfl_xor_sync(0xFFFFFFFF, ar_send_550, 1);
                            _eh_f[113] = ((ar_bit_137 == 1) ? ar_got_550 : ar_x0_550);
                            _eh_f[117] = ((ar_bit_137 == 1) ? ar_x1_550 : ar_got_550);
                            float ar_x0_551 = _eh_f[121];
                            float ar_x1_551 = _eh_f[125];
                            float ar_send_551 = ((ar_bit_137 == 1) ? ar_x0_551 : ar_x1_551);
                            float ar_got_551 = __shfl_xor_sync(0xFFFFFFFF, ar_send_551, 1);
                            _eh_f[121] = ((ar_bit_137 == 1) ? ar_got_551 : ar_x0_551);
                            _eh_f[125] = ((ar_bit_137 == 1) ? ar_x1_551 : ar_got_551);
                            int ar_bit_138 = lane >> 2 & 1;
                            float ar_x0_552 = _eh_f[98];
                            float ar_x1_552 = _eh_f[114];
                            float ar_send_552 = ((ar_bit_138 == 1) ? ar_x0_552 : ar_x1_552);
                            float ar_got_552 = __shfl_xor_sync(0xFFFFFFFF, ar_send_552, 4);
                            _eh_f[98] = ((ar_bit_138 == 1) ? ar_got_552 : ar_x0_552);
                            _eh_f[114] = ((ar_bit_138 == 1) ? ar_x1_552 : ar_got_552);
                            float ar_x0_553 = _eh_f[102];
                            float ar_x1_553 = _eh_f[118];
                            float ar_send_553 = ((ar_bit_138 == 1) ? ar_x0_553 : ar_x1_553);
                            float ar_got_553 = __shfl_xor_sync(0xFFFFFFFF, ar_send_553, 4);
                            _eh_f[102] = ((ar_bit_138 == 1) ? ar_got_553 : ar_x0_553);
                            _eh_f[118] = ((ar_bit_138 == 1) ? ar_x1_553 : ar_got_553);
                            float ar_x0_554 = _eh_f[106];
                            float ar_x1_554 = _eh_f[122];
                            float ar_send_554 = ((ar_bit_138 == 1) ? ar_x0_554 : ar_x1_554);
                            float ar_got_554 = __shfl_xor_sync(0xFFFFFFFF, ar_send_554, 4);
                            _eh_f[106] = ((ar_bit_138 == 1) ? ar_got_554 : ar_x0_554);
                            _eh_f[122] = ((ar_bit_138 == 1) ? ar_x1_554 : ar_got_554);
                            float ar_x0_555 = _eh_f[110];
                            float ar_x1_555 = _eh_f[126];
                            float ar_send_555 = ((ar_bit_138 == 1) ? ar_x0_555 : ar_x1_555);
                            float ar_got_555 = __shfl_xor_sync(0xFFFFFFFF, ar_send_555, 4);
                            _eh_f[110] = ((ar_bit_138 == 1) ? ar_got_555 : ar_x0_555);
                            _eh_f[126] = ((ar_bit_138 == 1) ? ar_x1_555 : ar_got_555);
                            int ar_bit_139 = lane >> 1 & 1;
                            float ar_x0_556 = _eh_f[98];
                            float ar_x1_556 = _eh_f[106];
                            float ar_send_556 = ((ar_bit_139 == 1) ? ar_x0_556 : ar_x1_556);
                            float ar_got_556 = __shfl_xor_sync(0xFFFFFFFF, ar_send_556, 2);
                            _eh_f[98] = ((ar_bit_139 == 1) ? ar_got_556 : ar_x0_556);
                            _eh_f[106] = ((ar_bit_139 == 1) ? ar_x1_556 : ar_got_556);
                            float ar_x0_557 = _eh_f[102];
                            float ar_x1_557 = _eh_f[110];
                            float ar_send_557 = ((ar_bit_139 == 1) ? ar_x0_557 : ar_x1_557);
                            float ar_got_557 = __shfl_xor_sync(0xFFFFFFFF, ar_send_557, 2);
                            _eh_f[102] = ((ar_bit_139 == 1) ? ar_got_557 : ar_x0_557);
                            _eh_f[110] = ((ar_bit_139 == 1) ? ar_x1_557 : ar_got_557);
                            float ar_x0_558 = _eh_f[114];
                            float ar_x1_558 = _eh_f[122];
                            float ar_send_558 = ((ar_bit_139 == 1) ? ar_x0_558 : ar_x1_558);
                            float ar_got_558 = __shfl_xor_sync(0xFFFFFFFF, ar_send_558, 2);
                            _eh_f[114] = ((ar_bit_139 == 1) ? ar_got_558 : ar_x0_558);
                            _eh_f[122] = ((ar_bit_139 == 1) ? ar_x1_558 : ar_got_558);
                            float ar_x0_559 = _eh_f[118];
                            float ar_x1_559 = _eh_f[126];
                            float ar_send_559 = ((ar_bit_139 == 1) ? ar_x0_559 : ar_x1_559);
                            float ar_got_559 = __shfl_xor_sync(0xFFFFFFFF, ar_send_559, 2);
                            _eh_f[118] = ((ar_bit_139 == 1) ? ar_got_559 : ar_x0_559);
                            _eh_f[126] = ((ar_bit_139 == 1) ? ar_x1_559 : ar_got_559);
                            int ar_bit_140 = lane & 1;
                            float ar_x0_560 = _eh_f[98];
                            float ar_x1_560 = _eh_f[102];
                            float ar_send_560 = ((ar_bit_140 == 1) ? ar_x0_560 : ar_x1_560);
                            float ar_got_560 = __shfl_xor_sync(0xFFFFFFFF, ar_send_560, 1);
                            _eh_f[98] = ((ar_bit_140 == 1) ? ar_got_560 : ar_x0_560);
                            _eh_f[102] = ((ar_bit_140 == 1) ? ar_x1_560 : ar_got_560);
                            float ar_x0_561 = _eh_f[106];
                            float ar_x1_561 = _eh_f[110];
                            float ar_send_561 = ((ar_bit_140 == 1) ? ar_x0_561 : ar_x1_561);
                            float ar_got_561 = __shfl_xor_sync(0xFFFFFFFF, ar_send_561, 1);
                            _eh_f[106] = ((ar_bit_140 == 1) ? ar_got_561 : ar_x0_561);
                            _eh_f[110] = ((ar_bit_140 == 1) ? ar_x1_561 : ar_got_561);
                            float ar_x0_562 = _eh_f[114];
                            float ar_x1_562 = _eh_f[118];
                            float ar_send_562 = ((ar_bit_140 == 1) ? ar_x0_562 : ar_x1_562);
                            float ar_got_562 = __shfl_xor_sync(0xFFFFFFFF, ar_send_562, 1);
                            _eh_f[114] = ((ar_bit_140 == 1) ? ar_got_562 : ar_x0_562);
                            _eh_f[118] = ((ar_bit_140 == 1) ? ar_x1_562 : ar_got_562);
                            float ar_x0_563 = _eh_f[122];
                            float ar_x1_563 = _eh_f[126];
                            float ar_send_563 = ((ar_bit_140 == 1) ? ar_x0_563 : ar_x1_563);
                            float ar_got_563 = __shfl_xor_sync(0xFFFFFFFF, ar_send_563, 1);
                            _eh_f[122] = ((ar_bit_140 == 1) ? ar_got_563 : ar_x0_563);
                            _eh_f[126] = ((ar_bit_140 == 1) ? ar_x1_563 : ar_got_563);
                            int ar_bit_141 = lane >> 2 & 1;
                            float ar_x0_564 = _eh_f[99];
                            float ar_x1_564 = _eh_f[115];
                            float ar_send_564 = ((ar_bit_141 == 1) ? ar_x0_564 : ar_x1_564);
                            float ar_got_564 = __shfl_xor_sync(0xFFFFFFFF, ar_send_564, 4);
                            _eh_f[99] = ((ar_bit_141 == 1) ? ar_got_564 : ar_x0_564);
                            _eh_f[115] = ((ar_bit_141 == 1) ? ar_x1_564 : ar_got_564);
                            float ar_x0_565 = _eh_f[103];
                            float ar_x1_565 = _eh_f[119];
                            float ar_send_565 = ((ar_bit_141 == 1) ? ar_x0_565 : ar_x1_565);
                            float ar_got_565 = __shfl_xor_sync(0xFFFFFFFF, ar_send_565, 4);
                            _eh_f[103] = ((ar_bit_141 == 1) ? ar_got_565 : ar_x0_565);
                            _eh_f[119] = ((ar_bit_141 == 1) ? ar_x1_565 : ar_got_565);
                            float ar_x0_566 = _eh_f[107];
                            float ar_x1_566 = _eh_f[123];
                            float ar_send_566 = ((ar_bit_141 == 1) ? ar_x0_566 : ar_x1_566);
                            float ar_got_566 = __shfl_xor_sync(0xFFFFFFFF, ar_send_566, 4);
                            _eh_f[107] = ((ar_bit_141 == 1) ? ar_got_566 : ar_x0_566);
                            _eh_f[123] = ((ar_bit_141 == 1) ? ar_x1_566 : ar_got_566);
                            float ar_x0_567 = _eh_f[111];
                            float ar_x1_567 = _eh_f[127];
                            float ar_send_567 = ((ar_bit_141 == 1) ? ar_x0_567 : ar_x1_567);
                            float ar_got_567 = __shfl_xor_sync(0xFFFFFFFF, ar_send_567, 4);
                            _eh_f[111] = ((ar_bit_141 == 1) ? ar_got_567 : ar_x0_567);
                            _eh_f[127] = ((ar_bit_141 == 1) ? ar_x1_567 : ar_got_567);
                            int ar_bit_142 = lane >> 1 & 1;
                            float ar_x0_568 = _eh_f[99];
                            float ar_x1_568 = _eh_f[107];
                            float ar_send_568 = ((ar_bit_142 == 1) ? ar_x0_568 : ar_x1_568);
                            float ar_got_568 = __shfl_xor_sync(0xFFFFFFFF, ar_send_568, 2);
                            _eh_f[99] = ((ar_bit_142 == 1) ? ar_got_568 : ar_x0_568);
                            _eh_f[107] = ((ar_bit_142 == 1) ? ar_x1_568 : ar_got_568);
                            float ar_x0_569 = _eh_f[103];
                            float ar_x1_569 = _eh_f[111];
                            float ar_send_569 = ((ar_bit_142 == 1) ? ar_x0_569 : ar_x1_569);
                            float ar_got_569 = __shfl_xor_sync(0xFFFFFFFF, ar_send_569, 2);
                            _eh_f[103] = ((ar_bit_142 == 1) ? ar_got_569 : ar_x0_569);
                            _eh_f[111] = ((ar_bit_142 == 1) ? ar_x1_569 : ar_got_569);
                            float ar_x0_570 = _eh_f[115];
                            float ar_x1_570 = _eh_f[123];
                            float ar_send_570 = ((ar_bit_142 == 1) ? ar_x0_570 : ar_x1_570);
                            float ar_got_570 = __shfl_xor_sync(0xFFFFFFFF, ar_send_570, 2);
                            _eh_f[115] = ((ar_bit_142 == 1) ? ar_got_570 : ar_x0_570);
                            _eh_f[123] = ((ar_bit_142 == 1) ? ar_x1_570 : ar_got_570);
                            float ar_x0_571 = _eh_f[119];
                            float ar_x1_571 = _eh_f[127];
                            float ar_send_571 = ((ar_bit_142 == 1) ? ar_x0_571 : ar_x1_571);
                            float ar_got_571 = __shfl_xor_sync(0xFFFFFFFF, ar_send_571, 2);
                            _eh_f[119] = ((ar_bit_142 == 1) ? ar_got_571 : ar_x0_571);
                            _eh_f[127] = ((ar_bit_142 == 1) ? ar_x1_571 : ar_got_571);
                            int ar_bit_143 = lane & 1;
                            float ar_x0_572 = _eh_f[99];
                            float ar_x1_572 = _eh_f[103];
                            float ar_send_572 = ((ar_bit_143 == 1) ? ar_x0_572 : ar_x1_572);
                            float ar_got_572 = __shfl_xor_sync(0xFFFFFFFF, ar_send_572, 1);
                            _eh_f[99] = ((ar_bit_143 == 1) ? ar_got_572 : ar_x0_572);
                            _eh_f[103] = ((ar_bit_143 == 1) ? ar_x1_572 : ar_got_572);
                            float ar_x0_573 = _eh_f[107];
                            float ar_x1_573 = _eh_f[111];
                            float ar_send_573 = ((ar_bit_143 == 1) ? ar_x0_573 : ar_x1_573);
                            float ar_got_573 = __shfl_xor_sync(0xFFFFFFFF, ar_send_573, 1);
                            _eh_f[107] = ((ar_bit_143 == 1) ? ar_got_573 : ar_x0_573);
                            _eh_f[111] = ((ar_bit_143 == 1) ? ar_x1_573 : ar_got_573);
                            float ar_x0_574 = _eh_f[115];
                            float ar_x1_574 = _eh_f[119];
                            float ar_send_574 = ((ar_bit_143 == 1) ? ar_x0_574 : ar_x1_574);
                            float ar_got_574 = __shfl_xor_sync(0xFFFFFFFF, ar_send_574, 1);
                            _eh_f[115] = ((ar_bit_143 == 1) ? ar_got_574 : ar_x0_574);
                            _eh_f[119] = ((ar_bit_143 == 1) ? ar_x1_574 : ar_got_574);
                            float ar_x0_575 = _eh_f[123];
                            float ar_x1_575 = _eh_f[127];
                            float ar_send_575 = ((ar_bit_143 == 1) ? ar_x0_575 : ar_x1_575);
                            float ar_got_575 = __shfl_xor_sync(0xFFFFFFFF, ar_send_575, 1);
                            _eh_f[123] = ((ar_bit_143 == 1) ? ar_got_575 : ar_x0_575);
                            _eh_f[127] = ((ar_bit_143 == 1) ? ar_x1_575 : ar_got_575);
                            int ar_ri_11 = eh_out + (lane & 7) * (4 - ldc) + 96;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_11])), "f"(_eh_f[96]), "f"(_eh_f[97]), "f"(_eh_f[98]), "f"(_eh_f[99]) : "memory");
                            ar_ri_11 = ar_ri_11 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_11])), "f"(_eh_f[100]), "f"(_eh_f[101]), "f"(_eh_f[102]), "f"(_eh_f[103]) : "memory");
                            ar_ri_11 = ar_ri_11 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_11])), "f"(_eh_f[104]), "f"(_eh_f[105]), "f"(_eh_f[106]), "f"(_eh_f[107]) : "memory");
                            ar_ri_11 = ar_ri_11 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_11])), "f"(_eh_f[108]), "f"(_eh_f[109]), "f"(_eh_f[110]), "f"(_eh_f[111]) : "memory");
                            ar_ri_11 = ar_ri_11 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_11])), "f"(_eh_f[112]), "f"(_eh_f[113]), "f"(_eh_f[114]), "f"(_eh_f[115]) : "memory");
                            ar_ri_11 = ar_ri_11 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_11])), "f"(_eh_f[116]), "f"(_eh_f[117]), "f"(_eh_f[118]), "f"(_eh_f[119]) : "memory");
                            ar_ri_11 = ar_ri_11 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_11])), "f"(_eh_f[120]), "f"(_eh_f[121]), "f"(_eh_f[122]), "f"(_eh_f[123]) : "memory");
                            ar_ri_11 = ar_ri_11 + ldc;
                            asm volatile("red.global.add.noftz.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&C[ar_ri_11])), "f"(_eh_f[124]), "f"(_eh_f[125]), "f"(_eh_f[126]), "f"(_eh_f[127]) : "memory");
                        }
                    }
                    if (whole_eh == 0) {
                        int eh_part = ((int)j_eh * splits_eh + s_eh) * (CLUSTER_M * BLOCK_N) + (cta_rank * BLOCK_M + epi_tid) * BLOCK_N + epi_half * 128;
                        {
                            unsigned _stv8_0_0 = __float_as_uint(_eh_f[0 + 0]);
                            unsigned _stv8_0_1 = __float_as_uint(_eh_f[0 + 1]);
                            unsigned _stv8_0_2 = __float_as_uint(_eh_f[0 + 2]);
                            unsigned _stv8_0_3 = __float_as_uint(_eh_f[0 + 3]);
                            unsigned _stv8_0_4 = __float_as_uint(_eh_f[0 + 4]);
                            unsigned _stv8_0_5 = __float_as_uint(_eh_f[0 + 5]);
                            unsigned _stv8_0_6 = __float_as_uint(_eh_f[0 + 6]);
                            unsigned _stv8_0_7 = __float_as_uint(_eh_f[0 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + eh_part + (0))), "r"(_stv8_0_0), "r"(_stv8_0_1), "r"(_stv8_0_2), "r"(_stv8_0_3), "r"(_stv8_0_4), "r"(_stv8_0_5), "r"(_stv8_0_6), "r"(_stv8_0_7) : "memory");
                        }
                        {
                            unsigned _stv8_1_0 = __float_as_uint(_eh_f[8 + 0]);
                            unsigned _stv8_1_1 = __float_as_uint(_eh_f[8 + 1]);
                            unsigned _stv8_1_2 = __float_as_uint(_eh_f[8 + 2]);
                            unsigned _stv8_1_3 = __float_as_uint(_eh_f[8 + 3]);
                            unsigned _stv8_1_4 = __float_as_uint(_eh_f[8 + 4]);
                            unsigned _stv8_1_5 = __float_as_uint(_eh_f[8 + 5]);
                            unsigned _stv8_1_6 = __float_as_uint(_eh_f[8 + 6]);
                            unsigned _stv8_1_7 = __float_as_uint(_eh_f[8 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 8) + (0))), "r"(_stv8_1_0), "r"(_stv8_1_1), "r"(_stv8_1_2), "r"(_stv8_1_3), "r"(_stv8_1_4), "r"(_stv8_1_5), "r"(_stv8_1_6), "r"(_stv8_1_7) : "memory");
                        }
                        {
                            unsigned _stv8_2_0 = __float_as_uint(_eh_f[16 + 0]);
                            unsigned _stv8_2_1 = __float_as_uint(_eh_f[16 + 1]);
                            unsigned _stv8_2_2 = __float_as_uint(_eh_f[16 + 2]);
                            unsigned _stv8_2_3 = __float_as_uint(_eh_f[16 + 3]);
                            unsigned _stv8_2_4 = __float_as_uint(_eh_f[16 + 4]);
                            unsigned _stv8_2_5 = __float_as_uint(_eh_f[16 + 5]);
                            unsigned _stv8_2_6 = __float_as_uint(_eh_f[16 + 6]);
                            unsigned _stv8_2_7 = __float_as_uint(_eh_f[16 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 16) + (0))), "r"(_stv8_2_0), "r"(_stv8_2_1), "r"(_stv8_2_2), "r"(_stv8_2_3), "r"(_stv8_2_4), "r"(_stv8_2_5), "r"(_stv8_2_6), "r"(_stv8_2_7) : "memory");
                        }
                        {
                            unsigned _stv8_3_0 = __float_as_uint(_eh_f[24 + 0]);
                            unsigned _stv8_3_1 = __float_as_uint(_eh_f[24 + 1]);
                            unsigned _stv8_3_2 = __float_as_uint(_eh_f[24 + 2]);
                            unsigned _stv8_3_3 = __float_as_uint(_eh_f[24 + 3]);
                            unsigned _stv8_3_4 = __float_as_uint(_eh_f[24 + 4]);
                            unsigned _stv8_3_5 = __float_as_uint(_eh_f[24 + 5]);
                            unsigned _stv8_3_6 = __float_as_uint(_eh_f[24 + 6]);
                            unsigned _stv8_3_7 = __float_as_uint(_eh_f[24 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 24) + (0))), "r"(_stv8_3_0), "r"(_stv8_3_1), "r"(_stv8_3_2), "r"(_stv8_3_3), "r"(_stv8_3_4), "r"(_stv8_3_5), "r"(_stv8_3_6), "r"(_stv8_3_7) : "memory");
                        }
                        {
                            unsigned _stv8_4_0 = __float_as_uint(_eh_f[32 + 0]);
                            unsigned _stv8_4_1 = __float_as_uint(_eh_f[32 + 1]);
                            unsigned _stv8_4_2 = __float_as_uint(_eh_f[32 + 2]);
                            unsigned _stv8_4_3 = __float_as_uint(_eh_f[32 + 3]);
                            unsigned _stv8_4_4 = __float_as_uint(_eh_f[32 + 4]);
                            unsigned _stv8_4_5 = __float_as_uint(_eh_f[32 + 5]);
                            unsigned _stv8_4_6 = __float_as_uint(_eh_f[32 + 6]);
                            unsigned _stv8_4_7 = __float_as_uint(_eh_f[32 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 32) + (0))), "r"(_stv8_4_0), "r"(_stv8_4_1), "r"(_stv8_4_2), "r"(_stv8_4_3), "r"(_stv8_4_4), "r"(_stv8_4_5), "r"(_stv8_4_6), "r"(_stv8_4_7) : "memory");
                        }
                        {
                            unsigned _stv8_5_0 = __float_as_uint(_eh_f[40 + 0]);
                            unsigned _stv8_5_1 = __float_as_uint(_eh_f[40 + 1]);
                            unsigned _stv8_5_2 = __float_as_uint(_eh_f[40 + 2]);
                            unsigned _stv8_5_3 = __float_as_uint(_eh_f[40 + 3]);
                            unsigned _stv8_5_4 = __float_as_uint(_eh_f[40 + 4]);
                            unsigned _stv8_5_5 = __float_as_uint(_eh_f[40 + 5]);
                            unsigned _stv8_5_6 = __float_as_uint(_eh_f[40 + 6]);
                            unsigned _stv8_5_7 = __float_as_uint(_eh_f[40 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 40) + (0))), "r"(_stv8_5_0), "r"(_stv8_5_1), "r"(_stv8_5_2), "r"(_stv8_5_3), "r"(_stv8_5_4), "r"(_stv8_5_5), "r"(_stv8_5_6), "r"(_stv8_5_7) : "memory");
                        }
                        {
                            unsigned _stv8_6_0 = __float_as_uint(_eh_f[48 + 0]);
                            unsigned _stv8_6_1 = __float_as_uint(_eh_f[48 + 1]);
                            unsigned _stv8_6_2 = __float_as_uint(_eh_f[48 + 2]);
                            unsigned _stv8_6_3 = __float_as_uint(_eh_f[48 + 3]);
                            unsigned _stv8_6_4 = __float_as_uint(_eh_f[48 + 4]);
                            unsigned _stv8_6_5 = __float_as_uint(_eh_f[48 + 5]);
                            unsigned _stv8_6_6 = __float_as_uint(_eh_f[48 + 6]);
                            unsigned _stv8_6_7 = __float_as_uint(_eh_f[48 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 48) + (0))), "r"(_stv8_6_0), "r"(_stv8_6_1), "r"(_stv8_6_2), "r"(_stv8_6_3), "r"(_stv8_6_4), "r"(_stv8_6_5), "r"(_stv8_6_6), "r"(_stv8_6_7) : "memory");
                        }
                        {
                            unsigned _stv8_7_0 = __float_as_uint(_eh_f[56 + 0]);
                            unsigned _stv8_7_1 = __float_as_uint(_eh_f[56 + 1]);
                            unsigned _stv8_7_2 = __float_as_uint(_eh_f[56 + 2]);
                            unsigned _stv8_7_3 = __float_as_uint(_eh_f[56 + 3]);
                            unsigned _stv8_7_4 = __float_as_uint(_eh_f[56 + 4]);
                            unsigned _stv8_7_5 = __float_as_uint(_eh_f[56 + 5]);
                            unsigned _stv8_7_6 = __float_as_uint(_eh_f[56 + 6]);
                            unsigned _stv8_7_7 = __float_as_uint(_eh_f[56 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 56) + (0))), "r"(_stv8_7_0), "r"(_stv8_7_1), "r"(_stv8_7_2), "r"(_stv8_7_3), "r"(_stv8_7_4), "r"(_stv8_7_5), "r"(_stv8_7_6), "r"(_stv8_7_7) : "memory");
                        }
                        {
                            unsigned _stv8_8_0 = __float_as_uint(_eh_f[64 + 0]);
                            unsigned _stv8_8_1 = __float_as_uint(_eh_f[64 + 1]);
                            unsigned _stv8_8_2 = __float_as_uint(_eh_f[64 + 2]);
                            unsigned _stv8_8_3 = __float_as_uint(_eh_f[64 + 3]);
                            unsigned _stv8_8_4 = __float_as_uint(_eh_f[64 + 4]);
                            unsigned _stv8_8_5 = __float_as_uint(_eh_f[64 + 5]);
                            unsigned _stv8_8_6 = __float_as_uint(_eh_f[64 + 6]);
                            unsigned _stv8_8_7 = __float_as_uint(_eh_f[64 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 64) + (0))), "r"(_stv8_8_0), "r"(_stv8_8_1), "r"(_stv8_8_2), "r"(_stv8_8_3), "r"(_stv8_8_4), "r"(_stv8_8_5), "r"(_stv8_8_6), "r"(_stv8_8_7) : "memory");
                        }
                        {
                            unsigned _stv8_9_0 = __float_as_uint(_eh_f[72 + 0]);
                            unsigned _stv8_9_1 = __float_as_uint(_eh_f[72 + 1]);
                            unsigned _stv8_9_2 = __float_as_uint(_eh_f[72 + 2]);
                            unsigned _stv8_9_3 = __float_as_uint(_eh_f[72 + 3]);
                            unsigned _stv8_9_4 = __float_as_uint(_eh_f[72 + 4]);
                            unsigned _stv8_9_5 = __float_as_uint(_eh_f[72 + 5]);
                            unsigned _stv8_9_6 = __float_as_uint(_eh_f[72 + 6]);
                            unsigned _stv8_9_7 = __float_as_uint(_eh_f[72 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 72) + (0))), "r"(_stv8_9_0), "r"(_stv8_9_1), "r"(_stv8_9_2), "r"(_stv8_9_3), "r"(_stv8_9_4), "r"(_stv8_9_5), "r"(_stv8_9_6), "r"(_stv8_9_7) : "memory");
                        }
                        {
                            unsigned _stv8_10_0 = __float_as_uint(_eh_f[80 + 0]);
                            unsigned _stv8_10_1 = __float_as_uint(_eh_f[80 + 1]);
                            unsigned _stv8_10_2 = __float_as_uint(_eh_f[80 + 2]);
                            unsigned _stv8_10_3 = __float_as_uint(_eh_f[80 + 3]);
                            unsigned _stv8_10_4 = __float_as_uint(_eh_f[80 + 4]);
                            unsigned _stv8_10_5 = __float_as_uint(_eh_f[80 + 5]);
                            unsigned _stv8_10_6 = __float_as_uint(_eh_f[80 + 6]);
                            unsigned _stv8_10_7 = __float_as_uint(_eh_f[80 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 80) + (0))), "r"(_stv8_10_0), "r"(_stv8_10_1), "r"(_stv8_10_2), "r"(_stv8_10_3), "r"(_stv8_10_4), "r"(_stv8_10_5), "r"(_stv8_10_6), "r"(_stv8_10_7) : "memory");
                        }
                        {
                            unsigned _stv8_11_0 = __float_as_uint(_eh_f[88 + 0]);
                            unsigned _stv8_11_1 = __float_as_uint(_eh_f[88 + 1]);
                            unsigned _stv8_11_2 = __float_as_uint(_eh_f[88 + 2]);
                            unsigned _stv8_11_3 = __float_as_uint(_eh_f[88 + 3]);
                            unsigned _stv8_11_4 = __float_as_uint(_eh_f[88 + 4]);
                            unsigned _stv8_11_5 = __float_as_uint(_eh_f[88 + 5]);
                            unsigned _stv8_11_6 = __float_as_uint(_eh_f[88 + 6]);
                            unsigned _stv8_11_7 = __float_as_uint(_eh_f[88 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 88) + (0))), "r"(_stv8_11_0), "r"(_stv8_11_1), "r"(_stv8_11_2), "r"(_stv8_11_3), "r"(_stv8_11_4), "r"(_stv8_11_5), "r"(_stv8_11_6), "r"(_stv8_11_7) : "memory");
                        }
                        {
                            unsigned _stv8_12_0 = __float_as_uint(_eh_f[96 + 0]);
                            unsigned _stv8_12_1 = __float_as_uint(_eh_f[96 + 1]);
                            unsigned _stv8_12_2 = __float_as_uint(_eh_f[96 + 2]);
                            unsigned _stv8_12_3 = __float_as_uint(_eh_f[96 + 3]);
                            unsigned _stv8_12_4 = __float_as_uint(_eh_f[96 + 4]);
                            unsigned _stv8_12_5 = __float_as_uint(_eh_f[96 + 5]);
                            unsigned _stv8_12_6 = __float_as_uint(_eh_f[96 + 6]);
                            unsigned _stv8_12_7 = __float_as_uint(_eh_f[96 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 96) + (0))), "r"(_stv8_12_0), "r"(_stv8_12_1), "r"(_stv8_12_2), "r"(_stv8_12_3), "r"(_stv8_12_4), "r"(_stv8_12_5), "r"(_stv8_12_6), "r"(_stv8_12_7) : "memory");
                        }
                        {
                            unsigned _stv8_13_0 = __float_as_uint(_eh_f[104 + 0]);
                            unsigned _stv8_13_1 = __float_as_uint(_eh_f[104 + 1]);
                            unsigned _stv8_13_2 = __float_as_uint(_eh_f[104 + 2]);
                            unsigned _stv8_13_3 = __float_as_uint(_eh_f[104 + 3]);
                            unsigned _stv8_13_4 = __float_as_uint(_eh_f[104 + 4]);
                            unsigned _stv8_13_5 = __float_as_uint(_eh_f[104 + 5]);
                            unsigned _stv8_13_6 = __float_as_uint(_eh_f[104 + 6]);
                            unsigned _stv8_13_7 = __float_as_uint(_eh_f[104 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 104) + (0))), "r"(_stv8_13_0), "r"(_stv8_13_1), "r"(_stv8_13_2), "r"(_stv8_13_3), "r"(_stv8_13_4), "r"(_stv8_13_5), "r"(_stv8_13_6), "r"(_stv8_13_7) : "memory");
                        }
                        {
                            unsigned _stv8_14_0 = __float_as_uint(_eh_f[112 + 0]);
                            unsigned _stv8_14_1 = __float_as_uint(_eh_f[112 + 1]);
                            unsigned _stv8_14_2 = __float_as_uint(_eh_f[112 + 2]);
                            unsigned _stv8_14_3 = __float_as_uint(_eh_f[112 + 3]);
                            unsigned _stv8_14_4 = __float_as_uint(_eh_f[112 + 4]);
                            unsigned _stv8_14_5 = __float_as_uint(_eh_f[112 + 5]);
                            unsigned _stv8_14_6 = __float_as_uint(_eh_f[112 + 6]);
                            unsigned _stv8_14_7 = __float_as_uint(_eh_f[112 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 112) + (0))), "r"(_stv8_14_0), "r"(_stv8_14_1), "r"(_stv8_14_2), "r"(_stv8_14_3), "r"(_stv8_14_4), "r"(_stv8_14_5), "r"(_stv8_14_6), "r"(_stv8_14_7) : "memory");
                        }
                        {
                            unsigned _stv8_15_0 = __float_as_uint(_eh_f[120 + 0]);
                            unsigned _stv8_15_1 = __float_as_uint(_eh_f[120 + 1]);
                            unsigned _stv8_15_2 = __float_as_uint(_eh_f[120 + 2]);
                            unsigned _stv8_15_3 = __float_as_uint(_eh_f[120 + 3]);
                            unsigned _stv8_15_4 = __float_as_uint(_eh_f[120 + 4]);
                            unsigned _stv8_15_5 = __float_as_uint(_eh_f[120 + 5]);
                            unsigned _stv8_15_6 = __float_as_uint(_eh_f[120 + 6]);
                            unsigned _stv8_15_7 = __float_as_uint(_eh_f[120 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(partials + (eh_part + 120) + (0))), "r"(_stv8_15_0), "r"(_stv8_15_1), "r"(_stv8_15_2), "r"(_stv8_15_3), "r"(_stv8_15_4), "r"(_stv8_15_5), "r"(_stv8_15_6), "r"(_stv8_15_7) : "memory");
                        }
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
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
