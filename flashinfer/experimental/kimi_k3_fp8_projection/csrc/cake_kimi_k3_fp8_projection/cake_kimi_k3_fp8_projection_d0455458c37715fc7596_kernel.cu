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
#define SMEM_SMEM_RED_STAGE_BYTES 19968
#define SMEM_SMEM_RED_STRIDE 19968
#define SMEM_SMEM_STG_OFF 193024
#define SMEM_SMEM_STG_STAGE_BYTES 13312
#define SMEM_SMEM_STG_STRIDE 13312
#define SMEM_TOTAL 206336
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



__device__ __forceinline__ void tcgen05_mma_mxf8_bs_leader(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "setp.ne.b32 leader, %7, 0;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr), "r"(enable_input_d), "r"(leader) : "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


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





__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}



__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_leader(
    int taddr, uint64_t s_desc, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n\t"
        "}\n"
        :: "r"(taddr), "l"(s_desc), "r"(leader) : "memory");
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



extern "C" {

__global__ __launch_bounds__(192) __cluster_dims__(3,1,1) void
kernel_cake_kimi_k3_fp8_projection_d0455458c37715fc7596(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap SFW, const __grid_constant__ CUtensorMap SFX, __nv_bfloat16* __restrict__ out, float* __restrict__ partials, unsigned int* __restrict__ counters, int M, int n_tiles, int n_valid, int ldo, int num_k_iters, int sf_k_tiles, int split, int tok_per_cta, int total_work, int store_vec, __nv_bfloat16* __restrict__ x, int K, const __grid_constant__ CUtensorMap XB)
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
    #define red_full_addr (mbar_base + 80)
    #define peers_free_addr (mbar_base + 88)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 3;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 3;
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
    float* smem_stg = reinterpret_cast<float*>(smem_raw + 193024);
    const int smem_stg_addr = smem + 193024;

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
            // peers_free: 1 barriers, init_count=2
            mbarrier_init(smem + 88, 2);
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
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
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
                mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                #pragma unroll 1
                for (int i_1 = 0; i_1 < k_count_1; i_1++) {
                    mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((i_1 == 0) ? 1 : 0);
                    uint32_t _elected_0 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfw, make_sf_cp_desc_lo_sbo128((((smem_sfw0_addr) >> 4) + (mma_stage) * 2688)), _elected_0);
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfx, make_sf_cp_desc_lo_sbo128((((smem_sfx0_addr) >> 4) + (mma_stage) * 2688)), _elected_0);
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfw + 4, make_sf_cp_desc_lo_sbo128((((smem_sfw1_addr) >> 4) + (mma_stage) * 2688)), _elected_0);
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfx + 4, make_sf_cp_desc_lo_sbo128((((smem_sfx1_addr) >> 4) + (mma_stage) * 2688)), _elected_0);
                    int _smem_desc_lo_1 = (((smem_w_addr) >> 4) & 0x3FFF) + (mma_stage) * 2688;
                    int _smem_desc_lo_2 = (((smem_x_addr) >> 4) & 0x3FFF) + (mma_stage) * 2688;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 32)), a_desc + 0, b_desc + 0,
                            0x8880000U, tmem_tmem_sfw, tmem_tmem_sfx, ((init_flag) ? 0 : 1), _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 32)), a_desc + 2, b_desc + 2,
                            0x28880010U, tmem_tmem_sfw, tmem_tmem_sfx, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 32)), a_desc + 4, b_desc + 4,
                            0x48880020U, tmem_tmem_sfw, tmem_tmem_sfx, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 32)), a_desc + 6, b_desc + 6,
                            0x68880030U, tmem_tmem_sfw, tmem_tmem_sfx, 1, _elected_0);
                    }
                    int init2 = 0;
                    int _smem_desc_lo_3 = (((smem_w_addr + 16384) >> 4) & 0x3FFF) + (mma_stage) * 2688;
                    int _smem_desc_lo_4 = (((smem_x_addr + 4096) >> 4) & 0x3FFF) + (mma_stage) * 2688;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 32)), a_desc + 0, b_desc + 0,
                            0x8880000U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, ((init2) ? 0 : 1), _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 32)), a_desc + 2, b_desc + 2,
                            0x28880010U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 32)), a_desc + 4, b_desc + 4,
                            0x48880020U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 32)), a_desc + 6, b_desc + 6,
                            0x68880030U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1, _elected_0);
                    }
                    elect_commit(mma_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 4) { mma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(mainloop_done_addr);
                _phase_epilogue_done ^= 1;
            }
            mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
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
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
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
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(epilogue_done_addr);
                }
                _phase_mainloop_done ^= 1;
                unsigned int epi_base = smem_epi_addr + (unsigned int)(epi_group * 16384);
                int my_rank = cta_rank;
                int _min_0 = ((32) < ((my_rank + 1) * 11) ? (32) : ((my_rank + 1) * 11));
                int rows_mine = _min_0 - my_rank * 11;
                unsigned int line_b = (unsigned int)(lane_row * 52);
                unsigned int slot_b = (unsigned int)(my_rank * 6656);
                mbarrier_wait_cluster_hint(peers_free_addr, _phase_peers_free, 10000000);
                if (tid == 64) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (epi_group == 0) {
                    if (0 == my_rank) {
                        {
                            uint32_t _addr_1 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_1), "f"(_tmem_load_0[0]) : "memory");
                        }
                        {
                            uint32_t _addr_2 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_2), "f"(_tmem_load_0[1]) : "memory");
                        }
                        {
                            uint32_t _addr_3 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_3), "f"(_tmem_load_0[2]) : "memory");
                        }
                        {
                            uint32_t _addr_4 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_4), "f"(_tmem_load_0[3]) : "memory");
                        }
                        {
                            uint32_t _addr_5 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_5), "f"(_tmem_load_0[4]) : "memory");
                        }
                        {
                            uint32_t _addr_6 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_6), "f"(_tmem_load_0[5]) : "memory");
                        }
                        {
                            uint32_t _addr_7 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_7), "f"(_tmem_load_0[6]) : "memory");
                        }
                        {
                            uint32_t _addr_8 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_8), "f"(_tmem_load_0[7]) : "memory");
                        }
                        {
                            uint32_t _addr_9 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_9), "f"(_tmem_load_0[8]) : "memory");
                        }
                        {
                            uint32_t _addr_10 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_10), "f"(_tmem_load_0[9]) : "memory");
                        }
                        {
                            uint32_t _addr_11 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(_tmem_load_0[10]) : "memory");
                        }
                    } else {
                        int _max_13 = ((0) > (-my_rank) ? (0) : (-my_rank));
                        int _min_12 = ((1) < (_max_13) ? (1) : (_max_13));
                        int o_idx = -_min_12;
                        unsigned int stg_b = (unsigned int)(o_idx * 6656) + line_b;
                        {
                            uint32_t _addr_14 = static_cast<uint32_t>(smem_stg_addr + stg_b);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(_tmem_load_0[0]) : "memory");
                        }
                        {
                            uint32_t _addr_15 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(_tmem_load_0[1]) : "memory");
                        }
                        {
                            uint32_t _addr_16 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(_tmem_load_0[2]) : "memory");
                        }
                        {
                            uint32_t _addr_17 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(_tmem_load_0[3]) : "memory");
                        }
                        {
                            uint32_t _addr_18 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_18), "f"(_tmem_load_0[4]) : "memory");
                        }
                        {
                            uint32_t _addr_19 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(_tmem_load_0[5]) : "memory");
                        }
                        {
                            uint32_t _addr_20 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_20), "f"(_tmem_load_0[6]) : "memory");
                        }
                        {
                            uint32_t _addr_21 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_21), "f"(_tmem_load_0[7]) : "memory");
                        }
                        {
                            uint32_t _addr_22 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_22), "f"(_tmem_load_0[8]) : "memory");
                        }
                        {
                            uint32_t _addr_23 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(_tmem_load_0[9]) : "memory");
                        }
                        {
                            uint32_t _addr_24 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_24), "f"(_tmem_load_0[10]) : "memory");
                        }
                    }
                    if (1 == my_rank) {
                        {
                            uint32_t _addr_25 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_25), "f"(_tmem_load_0[11]) : "memory");
                        }
                        {
                            uint32_t _addr_26 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_26), "f"(_tmem_load_0[12]) : "memory");
                        }
                        {
                            uint32_t _addr_27 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_27), "f"(_tmem_load_0[13]) : "memory");
                        }
                        {
                            uint32_t _addr_28 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_28), "f"(_tmem_load_0[14]) : "memory");
                        }
                        {
                            uint32_t _addr_29 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_29), "f"(_tmem_load_0[15]) : "memory");
                        }
                        {
                            uint32_t _addr_30 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_30), "f"(_tmem_load_0[16]) : "memory");
                        }
                        {
                            uint32_t _addr_31 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_31), "f"(_tmem_load_0[17]) : "memory");
                        }
                        {
                            uint32_t _addr_32 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_32), "f"(_tmem_load_0[18]) : "memory");
                        }
                        {
                            uint32_t _addr_33 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_33), "f"(_tmem_load_0[19]) : "memory");
                        }
                        {
                            uint32_t _addr_34 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_34), "f"(_tmem_load_0[20]) : "memory");
                        }
                        {
                            uint32_t _addr_35 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_35), "f"(_tmem_load_0[21]) : "memory");
                        }
                    } else {
                        int _max_37 = ((0) > (1 - my_rank) ? (0) : (1 - my_rank));
                        int _min_36 = ((1) < (_max_37) ? (1) : (_max_37));
                        int o_idx_1 = 1 - _min_36;
                        unsigned int stg_b_1 = (unsigned int)(o_idx_1 * 6656) + line_b;
                        {
                            uint32_t _addr_38 = static_cast<uint32_t>(smem_stg_addr + stg_b_1);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_38), "f"(_tmem_load_0[11]) : "memory");
                        }
                        {
                            uint32_t _addr_39 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_39), "f"(_tmem_load_0[12]) : "memory");
                        }
                        {
                            uint32_t _addr_40 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_40), "f"(_tmem_load_0[13]) : "memory");
                        }
                        {
                            uint32_t _addr_41 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_41), "f"(_tmem_load_0[14]) : "memory");
                        }
                        {
                            uint32_t _addr_42 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_42), "f"(_tmem_load_0[15]) : "memory");
                        }
                        {
                            uint32_t _addr_43 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_43), "f"(_tmem_load_0[16]) : "memory");
                        }
                        {
                            uint32_t _addr_44 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_44), "f"(_tmem_load_0[17]) : "memory");
                        }
                        {
                            uint32_t _addr_45 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_45), "f"(_tmem_load_0[18]) : "memory");
                        }
                        {
                            uint32_t _addr_46 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_46), "f"(_tmem_load_0[19]) : "memory");
                        }
                        {
                            uint32_t _addr_47 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_47), "f"(_tmem_load_0[20]) : "memory");
                        }
                        {
                            uint32_t _addr_48 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_48), "f"(_tmem_load_0[21]) : "memory");
                        }
                    }
                    if (2 == my_rank) {
                        {
                            uint32_t _addr_49 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_49), "f"(_tmem_load_0[22]) : "memory");
                        }
                        {
                            uint32_t _addr_50 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_50), "f"(_tmem_load_0[23]) : "memory");
                        }
                        {
                            uint32_t _addr_51 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_51), "f"(_tmem_load_0[24]) : "memory");
                        }
                        {
                            uint32_t _addr_52 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_52), "f"(_tmem_load_0[25]) : "memory");
                        }
                        {
                            uint32_t _addr_53 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_53), "f"(_tmem_load_0[26]) : "memory");
                        }
                        {
                            uint32_t _addr_54 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_54), "f"(_tmem_load_0[27]) : "memory");
                        }
                        {
                            uint32_t _addr_55 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_55), "f"(_tmem_load_0[28]) : "memory");
                        }
                        {
                            uint32_t _addr_56 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_56), "f"(_tmem_load_0[29]) : "memory");
                        }
                        {
                            uint32_t _addr_57 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_57), "f"(_tmem_load_0[30]) : "memory");
                        }
                        {
                            uint32_t _addr_58 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_58), "f"(_tmem_load_0[31]) : "memory");
                        }
                    } else {
                        int _max_60 = ((0) > (2 - my_rank) ? (0) : (2 - my_rank));
                        int _min_59 = ((1) < (_max_60) ? (1) : (_max_60));
                        int o_idx_2 = 2 - _min_59;
                        unsigned int stg_b_2 = (unsigned int)(o_idx_2 * 6656) + line_b;
                        {
                            uint32_t _addr_61 = static_cast<uint32_t>(smem_stg_addr + stg_b_2);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_61), "f"(_tmem_load_0[22]) : "memory");
                        }
                        {
                            uint32_t _addr_62 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_62), "f"(_tmem_load_0[23]) : "memory");
                        }
                        {
                            uint32_t _addr_63 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_63), "f"(_tmem_load_0[24]) : "memory");
                        }
                        {
                            uint32_t _addr_64 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_64), "f"(_tmem_load_0[25]) : "memory");
                        }
                        {
                            uint32_t _addr_65 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_65), "f"(_tmem_load_0[26]) : "memory");
                        }
                        {
                            uint32_t _addr_66 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_66), "f"(_tmem_load_0[27]) : "memory");
                        }
                        {
                            uint32_t _addr_67 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_67), "f"(_tmem_load_0[28]) : "memory");
                        }
                        {
                            uint32_t _addr_68 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_68), "f"(_tmem_load_0[29]) : "memory");
                        }
                        {
                            uint32_t _addr_69 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_69), "f"(_tmem_load_0[30]) : "memory");
                        }
                        {
                            uint32_t _addr_70 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_70), "f"(_tmem_load_0[31]) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                int _min_71 = ((12) < (rows_mine) ? (12) : (rows_mine));
                int rows_round = ((0) > (_min_71) ? (0) : (_min_71));
                if (tid == 64) {
                    if (0 != my_rank) {
                        int _max_73 = ((0) > (-my_rank) ? (0) : (-my_rank));
                        int _min_72 = ((1) < (_max_73) ? (1) : (_max_73));
                        int o_idx_c = -_min_72;
                        uint32_t _mbar_peer_74;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_74) : "r"(red_full_addr), "r"(0));
                        uint32_t _cluster_mapa_75;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_75) : "r"(smem_red_addr + slot_b), "r"(0));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_cluster_mapa_75), "r"(smem_stg_addr + (unsigned int)(o_idx_c * 6656)), "r"((uint32_t)(6656)), "r"(_mbar_peer_74)
                            : "memory");
                    }
                    if (1 != my_rank) {
                        int _max_77 = ((0) > (1 - my_rank) ? (0) : (1 - my_rank));
                        int _min_76 = ((1) < (_max_77) ? (1) : (_max_77));
                        int o_idx_c_1 = 1 - _min_76;
                        uint32_t _mbar_peer_78;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_78) : "r"(red_full_addr), "r"(1));
                        uint32_t _cluster_mapa_79;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_79) : "r"(smem_red_addr + slot_b), "r"(1));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_cluster_mapa_79), "r"(smem_stg_addr + (unsigned int)(o_idx_c_1 * 6656)), "r"((uint32_t)(6656)), "r"(_mbar_peer_78)
                            : "memory");
                    }
                    if (2 != my_rank) {
                        int _max_81 = ((0) > (2 - my_rank) ? (0) : (2 - my_rank));
                        int _min_80 = ((1) < (_max_81) ? (1) : (_max_81));
                        int o_idx_c_2 = 2 - _min_80;
                        uint32_t _mbar_peer_82;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_82) : "r"(red_full_addr), "r"(2));
                        uint32_t _cluster_mapa_83;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_83) : "r"(smem_red_addr + slot_b), "r"(2));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_cluster_mapa_83), "r"(smem_stg_addr + (unsigned int)(o_idx_c_2 * 6656)), "r"((uint32_t)(6656)), "r"(_mbar_peer_82)
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    int _min_84 = ((1) < (rows_round) ? (1) : (rows_round));
                    mbarrier_arrive_expect_tx(red_full_addr, _min_84 * 13312);
                }
                mbarrier_wait_cluster_hint(red_full_addr, _phase_red_full, 10000000);
                if (epi_group == 0) {
                    if (0 == my_rank) {
                        float total = 0.0f;
                        #pragma unroll
                        for (int src = 0; src < 3; src++) {
                            total = total + smem_red[src * 1664 + lane_row * 13];
                        }
                        int tok_c = tok0;
                        if (feature < n_valid && tok_c < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total;
                        }
                        float total_0 = 0.0f;
                        #pragma unroll
                        for (int src_1 = 0; src_1 < 3; src_1++) {
                            total_0 = total_0 + smem_red[src_1 * 1664 + lane_row * 13 + 1];
                        }
                        int tok_c_1 = tok0 + 1;
                        if (feature < n_valid && tok_c_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_0;
                        }
                        float total_2 = 0.0f;
                        #pragma unroll
                        for (int src_2 = 0; src_2 < 3; src_2++) {
                            total_2 = total_2 + smem_red[src_2 * 1664 + lane_row * 13 + 2];
                        }
                        int tok_c_3 = tok0 + 2;
                        if (feature < n_valid && tok_c_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_2;
                        }
                        float total_4 = 0.0f;
                        #pragma unroll
                        for (int src_3 = 0; src_3 < 3; src_3++) {
                            total_4 = total_4 + smem_red[src_3 * 1664 + lane_row * 13 + 3];
                        }
                        int tok_c_5 = tok0 + 3;
                        if (feature < n_valid && tok_c_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_5 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_4;
                        }
                        float total_6 = 0.0f;
                        #pragma unroll
                        for (int src_4 = 0; src_4 < 3; src_4++) {
                            total_6 = total_6 + smem_red[src_4 * 1664 + lane_row * 13 + 4];
                        }
                        int tok_c_7 = tok0 + 4;
                        if (feature < n_valid && tok_c_7 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_7 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_6;
                        }
                        float total_8 = 0.0f;
                        #pragma unroll
                        for (int src_5 = 0; src_5 < 3; src_5++) {
                            total_8 = total_8 + smem_red[src_5 * 1664 + lane_row * 13 + 5];
                        }
                        int tok_c_9 = tok0 + 5;
                        if (feature < n_valid && tok_c_9 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_9 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_8;
                        }
                        float total_10 = 0.0f;
                        #pragma unroll
                        for (int src_6 = 0; src_6 < 3; src_6++) {
                            total_10 = total_10 + smem_red[src_6 * 1664 + lane_row * 13 + 6];
                        }
                        int tok_c_11 = tok0 + 6;
                        if (feature < n_valid && tok_c_11 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_11 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_10;
                        }
                        float total_12 = 0.0f;
                        #pragma unroll
                        for (int src_7 = 0; src_7 < 3; src_7++) {
                            total_12 = total_12 + smem_red[src_7 * 1664 + lane_row * 13 + 7];
                        }
                        int tok_c_13 = tok0 + 7;
                        if (feature < n_valid && tok_c_13 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_13 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_12;
                        }
                        float total_14 = 0.0f;
                        #pragma unroll
                        for (int src_8 = 0; src_8 < 3; src_8++) {
                            total_14 = total_14 + smem_red[src_8 * 1664 + lane_row * 13 + 8];
                        }
                        int tok_c_15 = tok0 + 8;
                        if (feature < n_valid && tok_c_15 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_15 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_14;
                        }
                        float total_16 = 0.0f;
                        #pragma unroll
                        for (int src_9 = 0; src_9 < 3; src_9++) {
                            total_16 = total_16 + smem_red[src_9 * 1664 + lane_row * 13 + 9];
                        }
                        int tok_c_17 = tok0 + 9;
                        if (feature < n_valid && tok_c_17 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_17 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_16;
                        }
                        float total_18 = 0.0f;
                        #pragma unroll
                        for (int src_10 = 0; src_10 < 3; src_10++) {
                            total_18 = total_18 + smem_red[src_10 * 1664 + lane_row * 13 + 10];
                        }
                        int tok_c_19 = tok0 + 10;
                        if (feature < n_valid && tok_c_19 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_19 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_18;
                        }
                    }
                    if (1 == my_rank) {
                        float total_1 = 0.0f;
                        #pragma unroll
                        for (int src_11 = 0; src_11 < 3; src_11++) {
                            total_1 = total_1 + smem_red[src_11 * 1664 + lane_row * 13];
                        }
                        int tok_c_2 = tok0 + 11;
                        if (feature < n_valid && tok_c_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1;
                        }
                        float total_0_1 = 0.0f;
                        #pragma unroll
                        for (int src_12 = 0; src_12 < 3; src_12++) {
                            total_0_1 = total_0_1 + smem_red[src_12 * 1664 + lane_row * 13 + 1];
                        }
                        int tok_c_1_1 = tok0 + 12;
                        if (feature < n_valid && tok_c_1_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_1_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_0_1;
                        }
                        float total_2_1 = 0.0f;
                        #pragma unroll
                        for (int src_13 = 0; src_13 < 3; src_13++) {
                            total_2_1 = total_2_1 + smem_red[src_13 * 1664 + lane_row * 13 + 2];
                        }
                        int tok_c_3_1 = tok0 + 13;
                        if (feature < n_valid && tok_c_3_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_3_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_2_1;
                        }
                        float total_4_1 = 0.0f;
                        #pragma unroll
                        for (int src_14 = 0; src_14 < 3; src_14++) {
                            total_4_1 = total_4_1 + smem_red[src_14 * 1664 + lane_row * 13 + 3];
                        }
                        int tok_c_5_1 = tok0 + 14;
                        if (feature < n_valid && tok_c_5_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_5_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_4_1;
                        }
                        float total_6_1 = 0.0f;
                        #pragma unroll
                        for (int src_15 = 0; src_15 < 3; src_15++) {
                            total_6_1 = total_6_1 + smem_red[src_15 * 1664 + lane_row * 13 + 4];
                        }
                        int tok_c_7_1 = tok0 + 15;
                        if (feature < n_valid && tok_c_7_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_7_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_6_1;
                        }
                        float total_8_1 = 0.0f;
                        #pragma unroll
                        for (int src_16 = 0; src_16 < 3; src_16++) {
                            total_8_1 = total_8_1 + smem_red[src_16 * 1664 + lane_row * 13 + 5];
                        }
                        int tok_c_9_1 = tok0 + 16;
                        if (feature < n_valid && tok_c_9_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_9_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_8_1;
                        }
                        float total_10_1 = 0.0f;
                        #pragma unroll
                        for (int src_17 = 0; src_17 < 3; src_17++) {
                            total_10_1 = total_10_1 + smem_red[src_17 * 1664 + lane_row * 13 + 6];
                        }
                        int tok_c_11_1 = tok0 + 17;
                        if (feature < n_valid && tok_c_11_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_11_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_10_1;
                        }
                        float total_12_1 = 0.0f;
                        #pragma unroll
                        for (int src_18 = 0; src_18 < 3; src_18++) {
                            total_12_1 = total_12_1 + smem_red[src_18 * 1664 + lane_row * 13 + 7];
                        }
                        int tok_c_13_1 = tok0 + 18;
                        if (feature < n_valid && tok_c_13_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_13_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_12_1;
                        }
                        float total_14_1 = 0.0f;
                        #pragma unroll
                        for (int src_19 = 0; src_19 < 3; src_19++) {
                            total_14_1 = total_14_1 + smem_red[src_19 * 1664 + lane_row * 13 + 8];
                        }
                        int tok_c_15_1 = tok0 + 19;
                        if (feature < n_valid && tok_c_15_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_15_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_14_1;
                        }
                        float total_16_1 = 0.0f;
                        #pragma unroll
                        for (int src_20 = 0; src_20 < 3; src_20++) {
                            total_16_1 = total_16_1 + smem_red[src_20 * 1664 + lane_row * 13 + 9];
                        }
                        int tok_c_17_1 = tok0 + 20;
                        if (feature < n_valid && tok_c_17_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_17_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_16_1;
                        }
                        float total_18_1 = 0.0f;
                        #pragma unroll
                        for (int src_21 = 0; src_21 < 3; src_21++) {
                            total_18_1 = total_18_1 + smem_red[src_21 * 1664 + lane_row * 13 + 10];
                        }
                        int tok_c_19_1 = tok0 + 21;
                        if (feature < n_valid && tok_c_19_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_19_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_18_1;
                        }
                    }
                    if (2 == my_rank) {
                        float total_3 = 0.0f;
                        #pragma unroll
                        for (int src_22 = 0; src_22 < 3; src_22++) {
                            total_3 = total_3 + smem_red[src_22 * 1664 + lane_row * 13];
                        }
                        int tok_c_4 = tok0 + 22;
                        if (feature < n_valid && tok_c_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3;
                        }
                        float total_0_2 = 0.0f;
                        #pragma unroll
                        for (int src_23 = 0; src_23 < 3; src_23++) {
                            total_0_2 = total_0_2 + smem_red[src_23 * 1664 + lane_row * 13 + 1];
                        }
                        int tok_c_1_2 = tok0 + 23;
                        if (feature < n_valid && tok_c_1_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_1_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_0_2;
                        }
                        float total_2_2 = 0.0f;
                        #pragma unroll
                        for (int src_24 = 0; src_24 < 3; src_24++) {
                            total_2_2 = total_2_2 + smem_red[src_24 * 1664 + lane_row * 13 + 2];
                        }
                        int tok_c_3_2 = tok0 + 24;
                        if (feature < n_valid && tok_c_3_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_3_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_2_2;
                        }
                        float total_4_2 = 0.0f;
                        #pragma unroll
                        for (int src_25 = 0; src_25 < 3; src_25++) {
                            total_4_2 = total_4_2 + smem_red[src_25 * 1664 + lane_row * 13 + 3];
                        }
                        int tok_c_5_2 = tok0 + 25;
                        if (feature < n_valid && tok_c_5_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_5_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_4_2;
                        }
                        float total_6_2 = 0.0f;
                        #pragma unroll
                        for (int src_26 = 0; src_26 < 3; src_26++) {
                            total_6_2 = total_6_2 + smem_red[src_26 * 1664 + lane_row * 13 + 4];
                        }
                        int tok_c_7_2 = tok0 + 26;
                        if (feature < n_valid && tok_c_7_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_7_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_6_2;
                        }
                        float total_8_2 = 0.0f;
                        #pragma unroll
                        for (int src_27 = 0; src_27 < 3; src_27++) {
                            total_8_2 = total_8_2 + smem_red[src_27 * 1664 + lane_row * 13 + 5];
                        }
                        int tok_c_9_2 = tok0 + 27;
                        if (feature < n_valid && tok_c_9_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_9_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_8_2;
                        }
                        float total_10_2 = 0.0f;
                        #pragma unroll
                        for (int src_28 = 0; src_28 < 3; src_28++) {
                            total_10_2 = total_10_2 + smem_red[src_28 * 1664 + lane_row * 13 + 6];
                        }
                        int tok_c_11_2 = tok0 + 28;
                        if (feature < n_valid && tok_c_11_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_11_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_10_2;
                        }
                        float total_12_2 = 0.0f;
                        #pragma unroll
                        for (int src_29 = 0; src_29 < 3; src_29++) {
                            total_12_2 = total_12_2 + smem_red[src_29 * 1664 + lane_row * 13 + 7];
                        }
                        int tok_c_13_2 = tok0 + 29;
                        if (feature < n_valid && tok_c_13_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_13_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_12_2;
                        }
                        float total_14_2 = 0.0f;
                        #pragma unroll
                        for (int src_30 = 0; src_30 < 3; src_30++) {
                            total_14_2 = total_14_2 + smem_red[src_30 * 1664 + lane_row * 13 + 8];
                        }
                        int tok_c_15_2 = tok0 + 30;
                        if (feature < n_valid && tok_c_15_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_15_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_14_2;
                        }
                        float total_16_2 = 0.0f;
                        #pragma unroll
                        for (int src_31 = 0; src_31 < 3; src_31++) {
                            total_16_2 = total_16_2 + smem_red[src_31 * 1664 + lane_row * 13 + 9];
                        }
                        int tok_c_17_2 = tok0 + 31;
                        if (feature < n_valid && tok_c_17_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_17_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_16_2;
                        }
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (n_items_e > it_e + 1) {
                    if (tid == 64) {
                        #pragma unroll
                        for (int p = 0; p < 3; p++) {
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

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Cleanup
}

} // extern "C"
