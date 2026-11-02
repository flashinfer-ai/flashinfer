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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 264
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFB_OFFSET 256
#define TMEM_TMEM_SFA_OFFSET 260
#define NUM_TMA_PIPE_STAGES 11
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_EPI_TMA_PIPE_STAGES 3
#define SMEM_EPI_STAGING_OFF 1024
#define SMEM_EPI_STAGING_STAGE_BYTES 4096
#define SMEM_EPI_STAGING_STRIDE 4096
#define SMEM_SMEM_B_OFF 13312
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 19456
#define SMEM_SMEM_A_OFF 29696
#define SMEM_SMEM_A_STAGE_BYTES 2048
#define SMEM_SMEM_A_STRIDE 19456
#define SMEM_SMEM_SFA_OFF 31744
#define SMEM_SMEM_SFA_STAGE_BYTES 512
#define SMEM_SMEM_SFA_STRIDE 19456
#define SMEM_SMEM_SFB_OFF 32256
#define SMEM_SMEM_SFB_STAGE_BYTES 512
#define SMEM_SMEM_SFB_STRIDE 19456
#define SMEM_TOTAL 227328
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



__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
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


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
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
kernel_cake_grouped_fp8_gemm_005d45335e5b2a81f511(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap A32, const __grid_constant__ CUtensorMap A64, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C_tma, const __grid_constant__ CUtensorMap SFA_packed, const __grid_constant__ CUtensorMap SFB_packed, int* __restrict__ m_indices, unsigned int shape_m, unsigned int shape_n, unsigned int k_tiles, unsigned int sf_cols)
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
    #define sf_full_addr (mbar_base + 88)
    #define tma_free_addr (mbar_base + 176)
    #define scale_ready_addr (mbar_base + 264)
    #define mainloop_done_addr (mbar_base + 352)
    #define epilogue_done_addr (mbar_base + 368)
    #define tmem_dealloc_bar_addr (mbar_base + 384)

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
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 13312);
    const int smem_b_addr = smem + 13312;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 29696);
    const int smem_a_addr = smem + 29696;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + 31744);
    const int smem_sfa_addr = smem + 31744;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 32256);
    const int smem_sfb_addr = smem + 32256;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A32))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A64))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA_packed))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB_packed))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C_tma))) : "memory");

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 49 barriers)
    // Mbarriers at smem_raw[0..392)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 11 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            // sf_full: 11 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // tma_free: 11 barriers, init_count=33
            mbarrier_init(smem + 176, 33);
            mbarrier_init(smem + 184, 33);
            mbarrier_init(smem + 192, 33);
            mbarrier_init(smem + 200, 33);
            mbarrier_init(smem + 208, 33);
            mbarrier_init(smem + 216, 33);
            mbarrier_init(smem + 224, 33);
            mbarrier_init(smem + 232, 33);
            mbarrier_init(smem + 240, 33);
            mbarrier_init(smem + 248, 33);
            mbarrier_init(smem + 256, 33);
            // scale_ready: 11 barriers, init_count=64
            mbarrier_init(smem + 264, 64);
            mbarrier_init(smem + 272, 64);
            mbarrier_init(smem + 280, 64);
            mbarrier_init(smem + 288, 64);
            mbarrier_init(smem + 296, 64);
            mbarrier_init(smem + 304, 64);
            mbarrier_init(smem + 312, 64);
            mbarrier_init(smem + 320, 64);
            mbarrier_init(smem + 328, 64);
            mbarrier_init(smem + 336, 64);
            mbarrier_init(smem + 344, 64);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            // epilogue_done: 2 barriers, init_count=8
            mbarrier_init(smem + 368, 8);
            mbarrier_init(smem + 376, 8);
            // tmem_dealloc_bar: 1 barriers, init_count=32
            mbarrier_init(smem + 384, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 264 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 392);
    if (warp == 2) {
        int _tmem_hold = smem + 392;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfb = taddr + 256;
    const int tmem_tmem_sfa = taddr + 260;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int num_workers = (unsigned int)num_clusters;
            unsigned int worker_idx = (unsigned int)cluster_id;
            unsigned int m_blocks = (shape_m + 32 - 1) / 32;
            unsigned int total_pairs = m_blocks * (shape_n / 256);
            unsigned int _phase_tma_free = 1;
            #pragma unroll 1
            for (unsigned int pair = worker_idx; pair < total_pairs; pair += num_workers) {
                unsigned int grid_np = shape_n / 256;
                unsigned int m_blocks_0 = (shape_m + 32 - 1) / 32;
                unsigned int pairs_per_l2 = m_blocks_0 * 8;
                unsigned int n_l2_limit = (unsigned int)8;
                unsigned int l2_group = pair / pairs_per_l2;
                unsigned int first_np = l2_group * 8;
                unsigned int in_l2 = pair % pairs_per_l2;
                unsigned int remaining_np = grid_np - first_np;
                unsigned int n_in_l2 = ((remaining_np > n_l2_limit) ? n_l2_limit : remaining_np);
                unsigned int m_block = in_l2 / n_in_l2;
                unsigned int n_pair = first_np + in_l2 % n_in_l2;
                unsigned int off_m = m_block * 32;
                unsigned int off_n = (n_pair * 2 + (unsigned int)cta_rank) * 128;
                unsigned int last_row = shape_m - 1;
                int e0 = m_indices[off_m];
                unsigned int next_pair = pair + num_workers;
                if (next_pair < total_pairs) {
                    unsigned int grid_np_0 = shape_n / 256;
                    unsigned int m_blocks_1 = (shape_m + 32 - 1) / 32;
                    unsigned int pairs_per_l2_2 = m_blocks_1 * 8;
                    unsigned int n_l2_limit_3 = (unsigned int)8;
                    unsigned int l2_group_4 = next_pair / pairs_per_l2_2;
                    unsigned int first_np_5 = l2_group_4 * 8;
                    unsigned int in_l2_6 = next_pair % pairs_per_l2_2;
                    unsigned int remaining_np_7 = grid_np_0 - first_np_5;
                    unsigned int n_in_l2_8 = ((remaining_np_7 > n_l2_limit_3) ? n_l2_limit_3 : remaining_np_7);
                    unsigned int m_block_9 = in_l2_6 / n_in_l2_8;
                    unsigned int n_pair_10 = first_np_5 + in_l2_6 % n_in_l2_8;
                    unsigned int off_m_11 = m_block_9 * 32;
                    unsigned int off_n_12 = (n_pair_10 * 2 + (unsigned int)cta_rank) * 128;
                    if (off_m_11 < shape_m) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(m_indices + off_m_11))); }
                }
                unsigned int nsub = (unsigned int)0;
                unsigned int runs = (unsigned int)0;
                if (e0 >= 0) {
                    nsub = 1;
                    runs = 1;
                }
                unsigned int half = nsub * 16;
                unsigned int tok_row0 = off_m + (unsigned int)0 + (unsigned int)cta_rank * half;
                unsigned int sf_row0 = off_m + (unsigned int)0;
                unsigned int sfb_outer0 = (unsigned int)e0 * sf_cols;
                #pragma unroll 1
                for (unsigned int _run = 0; _run < runs; _run++) {
                    #pragma unroll 1
                    for (unsigned int iter_k = 0; iter_k < k_tiles; iter_k++) {
                        mbarrier_wait(tma_free_addr + (load_stage) * 8, _phase_tma_free);
                        if (elect_sync()) {
                            unsigned int sf_bytes = (unsigned int)0;
                            if (iter_k % 4 == 0) {
                                tma_2d_gmem2smem(smem_sfa_addr + load_stage * 19456, (&SFA_packed), sf_row0, iter_k / 4, sf_full_addr + (load_stage) * 8);
                                tma_2d_gmem2smem(smem_sfb_addr + load_stage * 19456, (&SFB_packed), off_n, sfb_outer0 + iter_k / 4, sf_full_addr + (load_stage) * 8);
                                sf_bytes = 1024;
                            }
                            mbarrier_arrive_expect_tx(sf_full_addr + (load_stage) * 8, sf_bytes);
                            tma_4d_gmem2smem_cta2(smem_b_addr + load_stage * 19456, (&B), 0, off_n, iter_k, (unsigned int)e0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 19456, (&A), 0, tok_row0, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(18432)) : "memory");
                        }
                        load_stage += 1;
                        if (load_stage == 11) { load_stage = 0; _phase_tma_free ^= 1; }
                    }
                }
            }
            #pragma unroll
            for (int _tail = 0; _tail < 11; _tail++) {
                mbarrier_wait(tma_free_addr + (load_stage) * 8, _phase_tma_free);
                load_stage += 1;
                if (load_stage == 11) { load_stage = 0; _phase_tma_free ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int tma_stage = 0;
            unsigned int epi_stage = 0;
            unsigned int num_workers_1 = (unsigned int)num_clusters;
            unsigned int worker_idx_1 = (unsigned int)cluster_id;
            unsigned int m_blocks_2 = (shape_m + 32 - 1) / 32;
            unsigned int total_pairs_1 = m_blocks_2 * (shape_n / 256);
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (unsigned int pair_1 = worker_idx_1; pair_1 < total_pairs_1; pair_1 += num_workers_1) {
                unsigned int grid_np_1 = shape_n / 256;
                unsigned int m_blocks_0_1 = (shape_m + 32 - 1) / 32;
                unsigned int pairs_per_l2_1 = m_blocks_0_1 * 8;
                unsigned int n_l2_limit_1 = (unsigned int)8;
                unsigned int l2_group_1 = pair_1 / pairs_per_l2_1;
                unsigned int first_np_1 = l2_group_1 * 8;
                unsigned int in_l2_1 = pair_1 % pairs_per_l2_1;
                unsigned int remaining_np_1 = grid_np_1 - first_np_1;
                unsigned int n_in_l2_1 = ((remaining_np_1 > n_l2_limit_1) ? n_l2_limit_1 : remaining_np_1);
                unsigned int m_block_1 = in_l2_1 / n_in_l2_1;
                unsigned int n_pair_1 = first_np_1 + in_l2_1 % n_in_l2_1;
                unsigned int off_m_1 = m_block_1 * 32;
                unsigned int off_n_1 = (n_pair_1 * 2 + (unsigned int)cta_rank) * 128;
                unsigned int last_row_1 = shape_m - 1;
                int e0_1 = m_indices[off_m_1];
                unsigned int nsub_1 = (unsigned int)0;
                unsigned int runs_1 = (unsigned int)0;
                if (e0_1 >= 0) {
                    nsub_1 = 1;
                    runs_1 = 1;
                }
                #pragma unroll 1
                for (unsigned int _run_1 = 0; _run_1 < runs_1; _run_1++) {
                    unsigned int mma_subs = ((nsub_1 > 1) ? nsub_1 : 1);
                    unsigned int runtime_mma_n = mma_subs * 32;
                    if (cta_rank == 0) {
                        mbarrier_wait(epilogue_done_addr + (epi_stage) * 8, _phase_epilogue_done);
                    }
                    #pragma unroll 4
                    for (unsigned int iter_k_1 = 0; iter_k_1 < k_tiles; iter_k_1++) {
                        if (cta_rank == 0) {
                            mbarrier_wait(tma_full_addr + (tma_stage) * 8, _phase_tma_full);
                            mbarrier_wait(scale_ready_addr + (tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                            if (elect_sync()) {
                                if (iter_k_1 % 4 == 0) {
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (tma_stage) * 1216)));
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (tma_stage) * 1216)));
                                }
                                int _smem_desc_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (tma_stage) * 1216;
                                int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (tma_stage) * 1216;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf8_bs_cta2((tmem_accum + (epi_stage * 128)), a_desc + 0, b_desc + 0,
                                        (((0x10880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n) >> 3) << 17)) | ((((uint32_t)(iter_k_1) % 4U)) << 29) | ((((uint32_t)(iter_k_1) % 4U)) << 4)), tmem_tmem_sfb, tmem_tmem_sfa, ((init_flag) ? 0 : 1));
                                    tcgen05_mma_mxf8_bs_cta2((tmem_accum + (epi_stage * 128)), a_desc + 2, b_desc + 2,
                                        (((0x10880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n) >> 3) << 17)) | ((((uint32_t)(iter_k_1) % 4U)) << 29) | ((((uint32_t)(iter_k_1) % 4U)) << 4)), tmem_tmem_sfb, tmem_tmem_sfa, 1);
                                    tcgen05_mma_mxf8_bs_cta2((tmem_accum + (epi_stage * 128)), a_desc + 4, b_desc + 4,
                                        (((0x10880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n) >> 3) << 17)) | ((((uint32_t)(iter_k_1) % 4U)) << 29) | ((((uint32_t)(iter_k_1) % 4U)) << 4)), tmem_tmem_sfb, tmem_tmem_sfa, 1);
                                    tcgen05_mma_mxf8_bs_cta2((tmem_accum + (epi_stage * 128)), a_desc + 6, b_desc + 6,
                                        (((0x10880000U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(runtime_mma_n) >> 3) << 17)) | ((((uint32_t)(iter_k_1) % 4U)) << 29) | ((((uint32_t)(iter_k_1) % 4U)) << 4)), tmem_tmem_sfb, tmem_tmem_sfa, 1);
                                }
                            }
                            elect_commit_cg2_multicast(tma_free_addr + (tma_stage) * 8, (uint16_t)(3));
                        }
                        tma_stage += 1;
                        if (tma_stage == 11) { tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    if (cta_rank == 0) {
                        elect_commit_cg2_multicast(mainloop_done_addr + (epi_stage) * 8, (uint16_t)(3));
                    }
                    epi_stage += 1;
                    if (epi_stage == 2) { epi_stage = 0; _phase_epilogue_done ^= 1; }
                }
            }
            #pragma unroll
            for (int _tail_1 = 0; _tail_1 < 11; _tail_1++) {
                tma_stage += 1;
                if (tma_stage == 11) { tma_stage = 0; _phase_tma_full ^= 1; }
            }
            #pragma unroll
            for (int _drain = 0; _drain < 2; _drain++) {
                if (cta_rank == 0) {
                    mbarrier_wait(epilogue_done_addr + (epi_stage) * 8, _phase_epilogue_done);
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_epilogue_done ^= 1; }
            }
            int dealloc_peer_rank = cta_rank ^ 1;
            if (cta_rank != 0) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_bar_addr), "r"(dealloc_peer_rank) : "memory");
            }
            mbarrier_wait(tmem_dealloc_bar_addr, 0);
            if (cta_rank == 0) {
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_bar_addr), "r"(dealloc_peer_rank) : "memory");
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: transpose ----
    if (warp == 2) {
        { // transpose_main
            unsigned int transpose_stage = 0;
            unsigned int num_workers_2 = (unsigned int)num_clusters;
            unsigned int worker_idx_2 = (unsigned int)cluster_id;
            unsigned int m_blocks_3 = (shape_m + 32 - 1) / 32;
            unsigned int total_pairs_2 = m_blocks_3 * (shape_n / 256);
            unsigned int _phase_sf_full = 0;
            #pragma unroll 1
            for (unsigned int pair_2 = worker_idx_2; pair_2 < total_pairs_2; pair_2 += num_workers_2) {
                unsigned int grid_np_2 = shape_n / 256;
                unsigned int m_blocks_0_2 = (shape_m + 32 - 1) / 32;
                unsigned int pairs_per_l2_3 = m_blocks_0_2 * 8;
                unsigned int n_l2_limit_2 = (unsigned int)8;
                unsigned int l2_group_2 = pair_2 / pairs_per_l2_3;
                unsigned int first_np_2 = l2_group_2 * 8;
                unsigned int in_l2_2 = pair_2 % pairs_per_l2_3;
                unsigned int remaining_np_2 = grid_np_2 - first_np_2;
                unsigned int n_in_l2_2 = ((remaining_np_2 > n_l2_limit_2) ? n_l2_limit_2 : remaining_np_2);
                unsigned int m_block_2 = in_l2_2 / n_in_l2_2;
                unsigned int n_pair_2 = first_np_2 + in_l2_2 % n_in_l2_2;
                unsigned int off_m_2 = m_block_2 * 32;
                unsigned int off_n_2 = (n_pair_2 * 2 + (unsigned int)cta_rank) * 128;
                unsigned int last_row_2 = shape_m - 1;
                int e0_2 = m_indices[off_m_2];
                unsigned int nsub_2 = (unsigned int)0;
                unsigned int runs_2 = (unsigned int)0;
                if (e0_2 >= 0) {
                    nsub_2 = 1;
                    runs_2 = 1;
                }
                #pragma unroll 1
                for (unsigned int _run_2 = 0; _run_2 < runs_2; _run_2++) {
                    #pragma unroll 1
                    for (unsigned int iter_k_2 = 0; iter_k_2 < k_tiles; iter_k_2++) {
                        mbarrier_wait(sf_full_addr + (transpose_stage) * 8, _phase_sf_full);
                        if (iter_k_2 % 4 == 0) {
                            unsigned int _sf_v[4];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[0])) : "r"(smem_sfa_addr + transpose_stage * 19456 + (unsigned int)(((lane >> 3 ^ 0) * 32 + lane) * 4)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[1])) : "r"(smem_sfa_addr + transpose_stage * 19456 + (unsigned int)(((lane >> 3 ^ 1) * 32 + lane) * 4)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[2])) : "r"(smem_sfa_addr + transpose_stage * 19456 + (unsigned int)(((lane >> 3 ^ 2) * 32 + lane) * 4)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v[3])) : "r"(smem_sfa_addr + transpose_stage * 19456 + (unsigned int)(((lane >> 3 ^ 3) * 32 + lane) * 4)));
                            __syncwarp();
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfa_addr + transpose_stage * 19456 + (unsigned int)((lane * 4 + (lane >> 3 ^ 0)) * 4)), "r"(_sf_v[0]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfa_addr + transpose_stage * 19456 + (unsigned int)((lane * 4 + (lane >> 3 ^ 1)) * 4)), "r"(_sf_v[1]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfa_addr + transpose_stage * 19456 + (unsigned int)((lane * 4 + (lane >> 3 ^ 2)) * 4)), "r"(_sf_v[2]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfa_addr + transpose_stage * 19456 + (unsigned int)((lane * 4 + (lane >> 3 ^ 3)) * 4)), "r"(_sf_v[3]));
                            unsigned int _sf_v_0[4];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[0])) : "r"(smem_sfb_addr + transpose_stage * 19456 + (unsigned int)(((lane >> 3 ^ 0) * 32 + lane) * 4)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[1])) : "r"(smem_sfb_addr + transpose_stage * 19456 + (unsigned int)(((lane >> 3 ^ 1) * 32 + lane) * 4)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[2])) : "r"(smem_sfb_addr + transpose_stage * 19456 + (unsigned int)(((lane >> 3 ^ 2) * 32 + lane) * 4)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&_sf_v_0[3])) : "r"(smem_sfb_addr + transpose_stage * 19456 + (unsigned int)(((lane >> 3 ^ 3) * 32 + lane) * 4)));
                            __syncwarp();
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfb_addr + transpose_stage * 19456 + (unsigned int)((lane * 4 + (lane >> 3 ^ 0)) * 4)), "r"(_sf_v_0[0]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfb_addr + transpose_stage * 19456 + (unsigned int)((lane * 4 + (lane >> 3 ^ 1)) * 4)), "r"(_sf_v_0[1]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfb_addr + transpose_stage * 19456 + (unsigned int)((lane * 4 + (lane >> 3 ^ 2)) * 4)), "r"(_sf_v_0[2]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_sfb_addr + transpose_stage * 19456 + (unsigned int)((lane * 4 + (lane >> 3 ^ 3)) * 4)), "r"(_sf_v_0[3]));
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        }
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((scale_ready_addr + (transpose_stage) * 8) & 0xFEFFFFFF) : "memory");
                        mbarrier_arrive(tma_free_addr + (transpose_stage) * 8);
                        transpose_stage += 1;
                        if (transpose_stage == 11) { transpose_stage = 0; _phase_sf_full ^= 1; }
                    }
                }
            }
            #pragma unroll
            for (int _tail_2 = 0; _tail_2 < 11; _tail_2++) {
                transpose_stage += 1;
                if (transpose_stage == 11) { transpose_stage = 0; _phase_sf_full ^= 1; }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            unsigned int epi_stage_1 = 0;
            unsigned int tma_stage_1 = 0;
            const int epi_warp = warp % 4;
            unsigned int num_workers_3 = (unsigned int)num_clusters;
            unsigned int worker_idx_3 = (unsigned int)cluster_id;
            unsigned int m_blocks_4 = (shape_m + 32 - 1) / 32;
            unsigned int total_pairs_3 = m_blocks_4 * (shape_n / 256);
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int pair_3 = worker_idx_3; pair_3 < total_pairs_3; pair_3 += num_workers_3) {
                unsigned int grid_np_3 = shape_n / 256;
                unsigned int m_blocks_0_3 = (shape_m + 32 - 1) / 32;
                unsigned int pairs_per_l2_4 = m_blocks_0_3 * 8;
                unsigned int n_l2_limit_4 = (unsigned int)8;
                unsigned int l2_group_3 = pair_3 / pairs_per_l2_4;
                unsigned int first_np_3 = l2_group_3 * 8;
                unsigned int in_l2_3 = pair_3 % pairs_per_l2_4;
                unsigned int remaining_np_3 = grid_np_3 - first_np_3;
                unsigned int n_in_l2_3 = ((remaining_np_3 > n_l2_limit_4) ? n_l2_limit_4 : remaining_np_3);
                unsigned int m_block_3 = in_l2_3 / n_in_l2_3;
                unsigned int n_pair_3 = first_np_3 + in_l2_3 % n_in_l2_3;
                unsigned int off_m_3 = m_block_3 * 32;
                unsigned int off_n_3 = (n_pair_3 * 2 + (unsigned int)cta_rank) * 128;
                unsigned int last_row_3 = shape_m - 1;
                int e0_3 = m_indices[off_m_3];
                unsigned int nsub_3 = (unsigned int)0;
                unsigned int runs_3 = (unsigned int)0;
                if (e0_3 >= 0) {
                    nsub_3 = 1;
                    runs_3 = 1;
                }
                #pragma unroll 1
                for (unsigned int _run_3 = 0; _run_3 < runs_3; _run_3++) {
                    mbarrier_wait(mainloop_done_addr + (epi_stage_1) * 8, _phase_mainloop_done);
                    unsigned int num_strips = nsub_3 * 2;
                    unsigned int row_base = off_m_3 + (unsigned int)0;
                    int tmem_base = taddr + (unsigned int)((int)epi_stage_1 * 128) + (unsigned int)(epi_warp * 32 << 16);
                    #pragma unroll 1
                    for (unsigned int strip = 0; strip < num_strips; strip++) {
                        int col_strip = (int)strip * 16;
                        if (epi_warp == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 15, 128;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
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
                                : "r"(tmem_addr + 1048576));
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
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 15, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                tma_store_2d((&C_tma), off_n_3, row_base + (unsigned int)col_strip, stage_base);
                                tma_store_2d((&C_tma), off_n_3 + 64, row_base + (unsigned int)col_strip, stage_base + 2048);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        __syncwarp();
                        tma_stage_1 += 1;
                        if (tma_stage_1 == 3) { tma_stage_1 = 0; }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((epilogue_done_addr + (epi_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    epi_stage_1 += 1;
                    if (epi_stage_1 == 2) { epi_stage_1 = 0; _phase_mainloop_done ^= 1; }
                }
            }
            if (epi_warp == 0) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
        }
    }

    // Cleanup
}

} // extern "C"
