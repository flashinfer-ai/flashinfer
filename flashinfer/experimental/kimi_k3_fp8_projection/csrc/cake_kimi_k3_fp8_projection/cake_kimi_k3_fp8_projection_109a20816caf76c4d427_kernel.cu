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
#define TMEM_NCOLS 80
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFW_OFFSET 64
#define TMEM_TMEM_SFX_OFFSET 72
#define NUM_TMA_PIPE_STAGES 3
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_RED_PIPE_STAGES 1
#define SMEM_SMEM_W_OFF 1024
#define SMEM_SMEM_W_STAGE_BYTES 32768
#define SMEM_SMEM_W_STRIDE 51200
#define SMEM_SMEM_SFW0_OFF 50176
#define SMEM_SMEM_SFW0_STAGE_BYTES 512
#define SMEM_SMEM_SFW0_STRIDE 51200
#define SMEM_SMEM_SFW1_OFF 50688
#define SMEM_SMEM_SFW1_STAGE_BYTES 512
#define SMEM_SMEM_SFW1_STRIDE 51200
#define SMEM_SMEM_X_OFF 33792
#define SMEM_SMEM_X_STAGE_BYTES 16384
#define SMEM_SMEM_X_STRIDE 51200
#define SMEM_SMEM_SFX_ALL_OFF 51200
#define SMEM_SMEM_SFX_ALL_STAGE_BYTES 1024
#define SMEM_SMEM_SFX_ALL_STRIDE 51200
#define SMEM_SMEM_SFX0_OFF 51200
#define SMEM_SMEM_SFX0_STAGE_BYTES 512
#define SMEM_SMEM_SFX0_STRIDE 51200
#define SMEM_SMEM_SFX1_OFF 51712
#define SMEM_SMEM_SFX1_STAGE_BYTES 512
#define SMEM_SMEM_SFX1_STRIDE 51200
#define SMEM_SMEM_EPI_OFF 154624
#define SMEM_SMEM_EPI_STAGE_BYTES 16384
#define SMEM_SMEM_EPI_STRIDE 16384
#define SMEM_SMEM_RED_OFF 154624
#define SMEM_SMEM_RED_STAGE_BYTES 36864
#define SMEM_SMEM_RED_STRIDE 36864
#define SMEM_SMEM_STG_OFF 191488
#define SMEM_SMEM_STG_STAGE_BYTES 32256
#define SMEM_SMEM_STG_STRIDE 32256
#define SMEM_TOTAL 223744
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

__global__ __launch_bounds__(192) __cluster_dims__(8,1,1) void
kernel_cake_kimi_k3_fp8_projection_109a20816caf76c4d427(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap SFW, const __grid_constant__ CUtensorMap SFX, __nv_bfloat16* __restrict__ out, float* __restrict__ partials, unsigned int* __restrict__ counters, int M, int n_tiles, int n_valid, int ldo, int num_k_iters, int sf_k_tiles, int split, int tok_per_cta, int total_work, int store_vec, __nv_bfloat16* __restrict__ x, int K, const __grid_constant__ CUtensorMap XB)
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
    #define mma_done_addr (mbar_base + 24)
    #define mainloop_done_addr (mbar_base + 48)
    #define epilogue_done_addr (mbar_base + 56)
    #define red_full_addr (mbar_base + 64)
    #define peers_free_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 8;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 8;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_w = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_w_addr = smem + 1024;
    uint8_t* smem_sfw0 = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int smem_sfw0_addr = smem + 50176;
    uint8_t* smem_sfw1 = reinterpret_cast<uint8_t*>(smem_raw + 50688);
    const int smem_sfw1_addr = smem + 50688;
    uint8_t* smem_x = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_x_addr = smem + 33792;
    uint8_t* smem_sfx_all = reinterpret_cast<uint8_t*>(smem_raw + 51200);
    const int smem_sfx_all_addr = smem + 51200;
    uint8_t* smem_sfx0 = reinterpret_cast<uint8_t*>(smem_raw + 51200);
    const int smem_sfx0_addr = smem + 51200;
    uint8_t* smem_sfx1 = reinterpret_cast<uint8_t*>(smem_raw + 51712);
    const int smem_sfx1_addr = smem + 51712;
    uint8_t* smem_epi = reinterpret_cast<uint8_t*>(smem_raw + 154624);
    const int smem_epi_addr = smem + 154624;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 154624);
    const int smem_red_addr = smem + 154624;
    float* smem_stg = reinterpret_cast<float*>(smem_raw + 191488);
    const int smem_stg_addr = smem + 191488;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // mma_done: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // epilogue_done: 1 barriers, init_count=4
            mbarrier_init(smem + 56, 4);
            // --- pipeline 'red_pipe' ---
            // red_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // peers_free: 1 barriers, init_count=7
            mbarrier_init(smem + 72, 7);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 80 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 0) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
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
    const int tmem_tmem_sfw = taddr + 64;
    const int tmem_tmem_sfx = taddr + 72;

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
                    int x_row = m_tile * 64;
                    int sfw_unit0 = n_tile / 2 * sf_k_tiles * 2 + n_tile % 2;
                    int sfx_unit0 = m_tile * sf_k_tiles;
                    #pragma unroll 1
                    for (int i = 0; i < k_count; i++) {
                        int iter_k = k_begin + i;
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k_group = iter_k * 2;
                        tma_3d_gmem2smem(smem_w_addr + load_stage * 51200, (&W), 0, 0, w_tile0 + k_group, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw0_addr + load_stage * 51200, (&SFW), 0, 0, sfw_unit0 + k_group * 2, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfw1_addr + load_stage * 51200, (&SFW), 0, 0, sfw_unit0 + k_group * 2 + 2, tma_full_addr + (load_stage) * 8);
                        if (it == 0 && i == 0) {
                            asm volatile("griddepcontrol.wait;" ::: "memory");
                        }
                        tma_3d_gmem2smem(smem_x_addr + load_stage * 51200, (&X), 0, x_row, k_group, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfx_all_addr + load_stage * 51200, (&SFX), 0, 0, sfx_unit0 + k_group, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 51200);
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_mma_done ^= 1; }
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
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfw, make_sf_cp_desc_lo_sbo128((((smem_sfw0_addr) >> 4) + (mma_stage) * 3200)), _elected_0);
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfx, make_sf_cp_desc_lo_sbo128((((smem_sfx0_addr) >> 4) + (mma_stage) * 3200)), _elected_0);
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfw + 4, make_sf_cp_desc_lo_sbo128((((smem_sfw1_addr) >> 4) + (mma_stage) * 3200)), _elected_0);
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfx + 4, make_sf_cp_desc_lo_sbo128((((smem_sfx1_addr) >> 4) + (mma_stage) * 3200)), _elected_0);
                    int _smem_desc_lo_1 = (((smem_w_addr) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                    int _smem_desc_lo_2 = (((smem_x_addr) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 0, b_desc + 0,
                            0x8900000U, tmem_tmem_sfw, tmem_tmem_sfx, ((init_flag) ? 0 : 1), _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 2, b_desc + 2,
                            0x28900010U, tmem_tmem_sfw, tmem_tmem_sfx, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 4, b_desc + 4,
                            0x48900020U, tmem_tmem_sfw, tmem_tmem_sfx, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 6, b_desc + 6,
                            0x68900030U, tmem_tmem_sfw, tmem_tmem_sfx, 1, _elected_0);
                    }
                    int init2 = 0;
                    int _smem_desc_lo_3 = (((smem_w_addr + 16384) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                    int _smem_desc_lo_4 = (((smem_x_addr + 8192) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 0, b_desc + 0,
                            0x8900000U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, ((init2) ? 0 : 1), _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 2, b_desc + 2,
                            0x28900010U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 4, b_desc + 4,
                            0x48900020U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (0 * 64)), a_desc + 6, b_desc + 6,
                            0x68900030U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1, _elected_0);
                    }
                    elect_commit(mma_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 3) { mma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(mainloop_done_addr);
                _phase_epilogue_done ^= 1;
            }
            mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int epi_group = (warp - 2) / 4;
            int tok_g0 = epi_group * 64;
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
                int tok0 = m_tile_e * 64;
                unsigned long long col = (unsigned long long)feature;
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_col_e = 0;
                float _tmem_load_0[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_col_e + (unsigned int)tok_g0 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(epilogue_done_addr);
                }
                _phase_mainloop_done ^= 1;
                unsigned int epi_base = smem_epi_addr + (unsigned int)(epi_group * 16384);
                int my_rank = cta_rank;
                int _min_0 = ((64) < ((my_rank + 1) * 8) ? (64) : ((my_rank + 1) * 8));
                int rows_mine = _min_0 - my_rank * 8;
                unsigned int line_b = (unsigned int)(lane_row * 36);
                unsigned int slot_b = (unsigned int)(my_rank * 4608);
                mbarrier_wait_cluster_hint(peers_free_addr, _phase_peers_free, 10000000);
                if (warp == 2) {
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
                    } else {
                        int _max_10 = ((0) > (-my_rank) ? (0) : (-my_rank));
                        int _min_9 = ((1) < (_max_10) ? (1) : (_max_10));
                        int o_idx = -_min_9;
                        unsigned int stg_b = (unsigned int)(o_idx * 4608) + line_b;
                        {
                            uint32_t _addr_11 = static_cast<uint32_t>(smem_stg_addr + stg_b);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_11), "f"(_tmem_load_0[0]) : "memory");
                        }
                        {
                            uint32_t _addr_12 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(_tmem_load_0[1]) : "memory");
                        }
                        {
                            uint32_t _addr_13 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(_tmem_load_0[2]) : "memory");
                        }
                        {
                            uint32_t _addr_14 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(_tmem_load_0[3]) : "memory");
                        }
                        {
                            uint32_t _addr_15 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(_tmem_load_0[4]) : "memory");
                        }
                        {
                            uint32_t _addr_16 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(_tmem_load_0[5]) : "memory");
                        }
                        {
                            uint32_t _addr_17 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(_tmem_load_0[6]) : "memory");
                        }
                        {
                            uint32_t _addr_18 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_18), "f"(_tmem_load_0[7]) : "memory");
                        }
                    }
                    if (1 == my_rank) {
                        {
                            uint32_t _addr_19 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(_tmem_load_0[8]) : "memory");
                        }
                        {
                            uint32_t _addr_20 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_20), "f"(_tmem_load_0[9]) : "memory");
                        }
                        {
                            uint32_t _addr_21 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_21), "f"(_tmem_load_0[10]) : "memory");
                        }
                        {
                            uint32_t _addr_22 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_22), "f"(_tmem_load_0[11]) : "memory");
                        }
                        {
                            uint32_t _addr_23 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(_tmem_load_0[12]) : "memory");
                        }
                        {
                            uint32_t _addr_24 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_24), "f"(_tmem_load_0[13]) : "memory");
                        }
                        {
                            uint32_t _addr_25 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_25), "f"(_tmem_load_0[14]) : "memory");
                        }
                        {
                            uint32_t _addr_26 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_26), "f"(_tmem_load_0[15]) : "memory");
                        }
                    } else {
                        int _max_28 = ((0) > (1 - my_rank) ? (0) : (1 - my_rank));
                        int _min_27 = ((1) < (_max_28) ? (1) : (_max_28));
                        int o_idx_1 = 1 - _min_27;
                        unsigned int stg_b_1 = (unsigned int)(o_idx_1 * 4608) + line_b;
                        {
                            uint32_t _addr_29 = static_cast<uint32_t>(smem_stg_addr + stg_b_1);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_29), "f"(_tmem_load_0[8]) : "memory");
                        }
                        {
                            uint32_t _addr_30 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_30), "f"(_tmem_load_0[9]) : "memory");
                        }
                        {
                            uint32_t _addr_31 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_31), "f"(_tmem_load_0[10]) : "memory");
                        }
                        {
                            uint32_t _addr_32 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_32), "f"(_tmem_load_0[11]) : "memory");
                        }
                        {
                            uint32_t _addr_33 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_33), "f"(_tmem_load_0[12]) : "memory");
                        }
                        {
                            uint32_t _addr_34 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_34), "f"(_tmem_load_0[13]) : "memory");
                        }
                        {
                            uint32_t _addr_35 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_35), "f"(_tmem_load_0[14]) : "memory");
                        }
                        {
                            uint32_t _addr_36 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_36), "f"(_tmem_load_0[15]) : "memory");
                        }
                    }
                    if (2 == my_rank) {
                        {
                            uint32_t _addr_37 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_37), "f"(_tmem_load_0[16]) : "memory");
                        }
                        {
                            uint32_t _addr_38 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_38), "f"(_tmem_load_0[17]) : "memory");
                        }
                        {
                            uint32_t _addr_39 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_39), "f"(_tmem_load_0[18]) : "memory");
                        }
                        {
                            uint32_t _addr_40 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_40), "f"(_tmem_load_0[19]) : "memory");
                        }
                        {
                            uint32_t _addr_41 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_41), "f"(_tmem_load_0[20]) : "memory");
                        }
                        {
                            uint32_t _addr_42 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_42), "f"(_tmem_load_0[21]) : "memory");
                        }
                        {
                            uint32_t _addr_43 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_43), "f"(_tmem_load_0[22]) : "memory");
                        }
                        {
                            uint32_t _addr_44 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_44), "f"(_tmem_load_0[23]) : "memory");
                        }
                    } else {
                        int _max_46 = ((0) > (2 - my_rank) ? (0) : (2 - my_rank));
                        int _min_45 = ((1) < (_max_46) ? (1) : (_max_46));
                        int o_idx_2 = 2 - _min_45;
                        unsigned int stg_b_2 = (unsigned int)(o_idx_2 * 4608) + line_b;
                        {
                            uint32_t _addr_47 = static_cast<uint32_t>(smem_stg_addr + stg_b_2);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_47), "f"(_tmem_load_0[16]) : "memory");
                        }
                        {
                            uint32_t _addr_48 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_48), "f"(_tmem_load_0[17]) : "memory");
                        }
                        {
                            uint32_t _addr_49 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_49), "f"(_tmem_load_0[18]) : "memory");
                        }
                        {
                            uint32_t _addr_50 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_50), "f"(_tmem_load_0[19]) : "memory");
                        }
                        {
                            uint32_t _addr_51 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_51), "f"(_tmem_load_0[20]) : "memory");
                        }
                        {
                            uint32_t _addr_52 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_52), "f"(_tmem_load_0[21]) : "memory");
                        }
                        {
                            uint32_t _addr_53 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_53), "f"(_tmem_load_0[22]) : "memory");
                        }
                        {
                            uint32_t _addr_54 = static_cast<uint32_t>(smem_stg_addr + (stg_b_2 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_54), "f"(_tmem_load_0[23]) : "memory");
                        }
                    }
                    if (3 == my_rank) {
                        {
                            uint32_t _addr_55 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_55), "f"(_tmem_load_0[24]) : "memory");
                        }
                        {
                            uint32_t _addr_56 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_56), "f"(_tmem_load_0[25]) : "memory");
                        }
                        {
                            uint32_t _addr_57 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_57), "f"(_tmem_load_0[26]) : "memory");
                        }
                        {
                            uint32_t _addr_58 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_58), "f"(_tmem_load_0[27]) : "memory");
                        }
                        {
                            uint32_t _addr_59 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_59), "f"(_tmem_load_0[28]) : "memory");
                        }
                        {
                            uint32_t _addr_60 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_60), "f"(_tmem_load_0[29]) : "memory");
                        }
                        {
                            uint32_t _addr_61 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_61), "f"(_tmem_load_0[30]) : "memory");
                        }
                        {
                            uint32_t _addr_62 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_62), "f"(_tmem_load_0[31]) : "memory");
                        }
                    } else {
                        int _max_64 = ((0) > (3 - my_rank) ? (0) : (3 - my_rank));
                        int _min_63 = ((1) < (_max_64) ? (1) : (_max_64));
                        int o_idx_3 = 3 - _min_63;
                        unsigned int stg_b_3 = (unsigned int)(o_idx_3 * 4608) + line_b;
                        {
                            uint32_t _addr_65 = static_cast<uint32_t>(smem_stg_addr + stg_b_3);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_65), "f"(_tmem_load_0[24]) : "memory");
                        }
                        {
                            uint32_t _addr_66 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_66), "f"(_tmem_load_0[25]) : "memory");
                        }
                        {
                            uint32_t _addr_67 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_67), "f"(_tmem_load_0[26]) : "memory");
                        }
                        {
                            uint32_t _addr_68 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_68), "f"(_tmem_load_0[27]) : "memory");
                        }
                        {
                            uint32_t _addr_69 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_69), "f"(_tmem_load_0[28]) : "memory");
                        }
                        {
                            uint32_t _addr_70 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_70), "f"(_tmem_load_0[29]) : "memory");
                        }
                        {
                            uint32_t _addr_71 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_71), "f"(_tmem_load_0[30]) : "memory");
                        }
                        {
                            uint32_t _addr_72 = static_cast<uint32_t>(smem_stg_addr + (stg_b_3 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_72), "f"(_tmem_load_0[31]) : "memory");
                        }
                    }
                    if (4 == my_rank) {
                        {
                            uint32_t _addr_73 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_73), "f"(_tmem_load_0[32]) : "memory");
                        }
                        {
                            uint32_t _addr_74 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_74), "f"(_tmem_load_0[33]) : "memory");
                        }
                        {
                            uint32_t _addr_75 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_75), "f"(_tmem_load_0[34]) : "memory");
                        }
                        {
                            uint32_t _addr_76 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_76), "f"(_tmem_load_0[35]) : "memory");
                        }
                        {
                            uint32_t _addr_77 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_77), "f"(_tmem_load_0[36]) : "memory");
                        }
                        {
                            uint32_t _addr_78 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_78), "f"(_tmem_load_0[37]) : "memory");
                        }
                        {
                            uint32_t _addr_79 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_79), "f"(_tmem_load_0[38]) : "memory");
                        }
                        {
                            uint32_t _addr_80 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_80), "f"(_tmem_load_0[39]) : "memory");
                        }
                    } else {
                        int _max_82 = ((0) > (4 - my_rank) ? (0) : (4 - my_rank));
                        int _min_81 = ((1) < (_max_82) ? (1) : (_max_82));
                        int o_idx_4 = 4 - _min_81;
                        unsigned int stg_b_4 = (unsigned int)(o_idx_4 * 4608) + line_b;
                        {
                            uint32_t _addr_83 = static_cast<uint32_t>(smem_stg_addr + stg_b_4);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_83), "f"(_tmem_load_0[32]) : "memory");
                        }
                        {
                            uint32_t _addr_84 = static_cast<uint32_t>(smem_stg_addr + (stg_b_4 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_84), "f"(_tmem_load_0[33]) : "memory");
                        }
                        {
                            uint32_t _addr_85 = static_cast<uint32_t>(smem_stg_addr + (stg_b_4 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_85), "f"(_tmem_load_0[34]) : "memory");
                        }
                        {
                            uint32_t _addr_86 = static_cast<uint32_t>(smem_stg_addr + (stg_b_4 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_86), "f"(_tmem_load_0[35]) : "memory");
                        }
                        {
                            uint32_t _addr_87 = static_cast<uint32_t>(smem_stg_addr + (stg_b_4 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_87), "f"(_tmem_load_0[36]) : "memory");
                        }
                        {
                            uint32_t _addr_88 = static_cast<uint32_t>(smem_stg_addr + (stg_b_4 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_88), "f"(_tmem_load_0[37]) : "memory");
                        }
                        {
                            uint32_t _addr_89 = static_cast<uint32_t>(smem_stg_addr + (stg_b_4 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_89), "f"(_tmem_load_0[38]) : "memory");
                        }
                        {
                            uint32_t _addr_90 = static_cast<uint32_t>(smem_stg_addr + (stg_b_4 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_90), "f"(_tmem_load_0[39]) : "memory");
                        }
                    }
                    if (5 == my_rank) {
                        {
                            uint32_t _addr_91 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_91), "f"(_tmem_load_0[40]) : "memory");
                        }
                        {
                            uint32_t _addr_92 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_92), "f"(_tmem_load_0[41]) : "memory");
                        }
                        {
                            uint32_t _addr_93 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_93), "f"(_tmem_load_0[42]) : "memory");
                        }
                        {
                            uint32_t _addr_94 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_94), "f"(_tmem_load_0[43]) : "memory");
                        }
                        {
                            uint32_t _addr_95 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_95), "f"(_tmem_load_0[44]) : "memory");
                        }
                        {
                            uint32_t _addr_96 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_96), "f"(_tmem_load_0[45]) : "memory");
                        }
                        {
                            uint32_t _addr_97 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_97), "f"(_tmem_load_0[46]) : "memory");
                        }
                        {
                            uint32_t _addr_98 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_98), "f"(_tmem_load_0[47]) : "memory");
                        }
                    } else {
                        int _max_100 = ((0) > (5 - my_rank) ? (0) : (5 - my_rank));
                        int _min_99 = ((1) < (_max_100) ? (1) : (_max_100));
                        int o_idx_5 = 5 - _min_99;
                        unsigned int stg_b_5 = (unsigned int)(o_idx_5 * 4608) + line_b;
                        {
                            uint32_t _addr_101 = static_cast<uint32_t>(smem_stg_addr + stg_b_5);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_101), "f"(_tmem_load_0[40]) : "memory");
                        }
                        {
                            uint32_t _addr_102 = static_cast<uint32_t>(smem_stg_addr + (stg_b_5 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_102), "f"(_tmem_load_0[41]) : "memory");
                        }
                        {
                            uint32_t _addr_103 = static_cast<uint32_t>(smem_stg_addr + (stg_b_5 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_103), "f"(_tmem_load_0[42]) : "memory");
                        }
                        {
                            uint32_t _addr_104 = static_cast<uint32_t>(smem_stg_addr + (stg_b_5 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_104), "f"(_tmem_load_0[43]) : "memory");
                        }
                        {
                            uint32_t _addr_105 = static_cast<uint32_t>(smem_stg_addr + (stg_b_5 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_105), "f"(_tmem_load_0[44]) : "memory");
                        }
                        {
                            uint32_t _addr_106 = static_cast<uint32_t>(smem_stg_addr + (stg_b_5 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_106), "f"(_tmem_load_0[45]) : "memory");
                        }
                        {
                            uint32_t _addr_107 = static_cast<uint32_t>(smem_stg_addr + (stg_b_5 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_107), "f"(_tmem_load_0[46]) : "memory");
                        }
                        {
                            uint32_t _addr_108 = static_cast<uint32_t>(smem_stg_addr + (stg_b_5 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_108), "f"(_tmem_load_0[47]) : "memory");
                        }
                    }
                    if (6 == my_rank) {
                        {
                            uint32_t _addr_109 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_109), "f"(_tmem_load_0[48]) : "memory");
                        }
                        {
                            uint32_t _addr_110 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_110), "f"(_tmem_load_0[49]) : "memory");
                        }
                        {
                            uint32_t _addr_111 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_111), "f"(_tmem_load_0[50]) : "memory");
                        }
                        {
                            uint32_t _addr_112 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_112), "f"(_tmem_load_0[51]) : "memory");
                        }
                        {
                            uint32_t _addr_113 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_113), "f"(_tmem_load_0[52]) : "memory");
                        }
                        {
                            uint32_t _addr_114 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_114), "f"(_tmem_load_0[53]) : "memory");
                        }
                        {
                            uint32_t _addr_115 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_115), "f"(_tmem_load_0[54]) : "memory");
                        }
                        {
                            uint32_t _addr_116 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_116), "f"(_tmem_load_0[55]) : "memory");
                        }
                    } else {
                        int _max_118 = ((0) > (6 - my_rank) ? (0) : (6 - my_rank));
                        int _min_117 = ((1) < (_max_118) ? (1) : (_max_118));
                        int o_idx_6 = 6 - _min_117;
                        unsigned int stg_b_6 = (unsigned int)(o_idx_6 * 4608) + line_b;
                        {
                            uint32_t _addr_119 = static_cast<uint32_t>(smem_stg_addr + stg_b_6);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_119), "f"(_tmem_load_0[48]) : "memory");
                        }
                        {
                            uint32_t _addr_120 = static_cast<uint32_t>(smem_stg_addr + (stg_b_6 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_120), "f"(_tmem_load_0[49]) : "memory");
                        }
                        {
                            uint32_t _addr_121 = static_cast<uint32_t>(smem_stg_addr + (stg_b_6 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_121), "f"(_tmem_load_0[50]) : "memory");
                        }
                        {
                            uint32_t _addr_122 = static_cast<uint32_t>(smem_stg_addr + (stg_b_6 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_122), "f"(_tmem_load_0[51]) : "memory");
                        }
                        {
                            uint32_t _addr_123 = static_cast<uint32_t>(smem_stg_addr + (stg_b_6 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_123), "f"(_tmem_load_0[52]) : "memory");
                        }
                        {
                            uint32_t _addr_124 = static_cast<uint32_t>(smem_stg_addr + (stg_b_6 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_124), "f"(_tmem_load_0[53]) : "memory");
                        }
                        {
                            uint32_t _addr_125 = static_cast<uint32_t>(smem_stg_addr + (stg_b_6 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_125), "f"(_tmem_load_0[54]) : "memory");
                        }
                        {
                            uint32_t _addr_126 = static_cast<uint32_t>(smem_stg_addr + (stg_b_6 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_126), "f"(_tmem_load_0[55]) : "memory");
                        }
                    }
                    if (7 == my_rank) {
                        {
                            uint32_t _addr_127 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_127), "f"(_tmem_load_0[56]) : "memory");
                        }
                        {
                            uint32_t _addr_128 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_128), "f"(_tmem_load_0[57]) : "memory");
                        }
                        {
                            uint32_t _addr_129 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_129), "f"(_tmem_load_0[58]) : "memory");
                        }
                        {
                            uint32_t _addr_130 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_130), "f"(_tmem_load_0[59]) : "memory");
                        }
                        {
                            uint32_t _addr_131 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_131), "f"(_tmem_load_0[60]) : "memory");
                        }
                        {
                            uint32_t _addr_132 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_132), "f"(_tmem_load_0[61]) : "memory");
                        }
                        {
                            uint32_t _addr_133 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_133), "f"(_tmem_load_0[62]) : "memory");
                        }
                        {
                            uint32_t _addr_134 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_134), "f"(_tmem_load_0[63]) : "memory");
                        }
                    } else {
                        int _max_136 = ((0) > (7 - my_rank) ? (0) : (7 - my_rank));
                        int _min_135 = ((1) < (_max_136) ? (1) : (_max_136));
                        int o_idx_7 = 7 - _min_135;
                        unsigned int stg_b_7 = (unsigned int)(o_idx_7 * 4608) + line_b;
                        {
                            uint32_t _addr_137 = static_cast<uint32_t>(smem_stg_addr + stg_b_7);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_137), "f"(_tmem_load_0[56]) : "memory");
                        }
                        {
                            uint32_t _addr_138 = static_cast<uint32_t>(smem_stg_addr + (stg_b_7 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_138), "f"(_tmem_load_0[57]) : "memory");
                        }
                        {
                            uint32_t _addr_139 = static_cast<uint32_t>(smem_stg_addr + (stg_b_7 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_139), "f"(_tmem_load_0[58]) : "memory");
                        }
                        {
                            uint32_t _addr_140 = static_cast<uint32_t>(smem_stg_addr + (stg_b_7 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_140), "f"(_tmem_load_0[59]) : "memory");
                        }
                        {
                            uint32_t _addr_141 = static_cast<uint32_t>(smem_stg_addr + (stg_b_7 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_141), "f"(_tmem_load_0[60]) : "memory");
                        }
                        {
                            uint32_t _addr_142 = static_cast<uint32_t>(smem_stg_addr + (stg_b_7 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_142), "f"(_tmem_load_0[61]) : "memory");
                        }
                        {
                            uint32_t _addr_143 = static_cast<uint32_t>(smem_stg_addr + (stg_b_7 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_143), "f"(_tmem_load_0[62]) : "memory");
                        }
                        {
                            uint32_t _addr_144 = static_cast<uint32_t>(smem_stg_addr + (stg_b_7 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_144), "f"(_tmem_load_0[63]) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                int _min_145 = ((8) < (rows_mine) ? (8) : (rows_mine));
                int rows_round = ((0) > (_min_145) ? (0) : (_min_145));
                if (warp == 2) {
                    int owner_l = lane;
                    int _min_147 = ((64) < ((owner_l + 1) * 8) ? (64) : ((owner_l + 1) * 8));
                    int _min_146 = ((8) < (_min_147 - owner_l * 8) ? (8) : (_min_147 - owner_l * 8));
                    int n_o_l = ((0) > (_min_146) ? (0) : (_min_146));
                    if (owner_l < 8 && owner_l != my_rank && n_o_l > 0) {
                        int _max_149 = ((0) > (owner_l - my_rank) ? (0) : (owner_l - my_rank));
                        int _min_148 = ((1) < (_max_149) ? (1) : (_max_149));
                        int o_idx_l = owner_l - _min_148;
                        uint32_t _mbar_peer_150;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_150) : "r"(red_full_addr), "r"(owner_l));
                        uint32_t _cluster_mapa_151;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_151) : "r"(smem_red_addr + slot_b), "r"(owner_l));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_cluster_mapa_151), "r"(smem_stg_addr + (unsigned int)(o_idx_l * 4608)), "r"((uint32_t)(4608)), "r"(_mbar_peer_150)
                            : "memory");
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    if (lane == 0) {
                        int _min_152 = ((1) < (rows_round) ? (1) : (rows_round));
                        mbarrier_arrive_expect_tx(red_full_addr, _min_152 * 32256);
                    }
                }
                mbarrier_wait_cluster_hint(red_full_addr, _phase_red_full, 10000000);
                if (epi_group == 0) {
                    if (0 == my_rank) {
                        float total_1_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_0 = 0; src_1_0 < 8; src_1_0++) {
                            total_1_0 = total_1_0 + smem_red[src_1_0 * 1152 + lane_row * 9];
                        }
                        int tok_c_2_0 = tok0;
                        if (feature < n_valid && tok_c_2_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_0;
                        }
                        float total_3_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_1 = 0; src_1_1 < 8; src_1_1++) {
                            total_3_0 = total_3_0 + smem_red[src_1_1 * 1152 + lane_row * 9 + 1];
                        }
                        int tok_c_4_0 = tok0 + 1;
                        if (feature < n_valid && tok_c_4_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_0;
                        }
                        float total_5_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_2 = 0; src_1_2 < 8; src_1_2++) {
                            total_5_0 = total_5_0 + smem_red[src_1_2 * 1152 + lane_row * 9 + 2];
                        }
                        int tok_c_6_0 = tok0 + 2;
                        if (feature < n_valid && tok_c_6_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_0;
                        }
                        float total_7_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_3 = 0; src_1_3 < 8; src_1_3++) {
                            total_7_0 = total_7_0 + smem_red[src_1_3 * 1152 + lane_row * 9 + 3];
                        }
                        int tok_c_8_0 = tok0 + 3;
                        if (feature < n_valid && tok_c_8_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_0;
                        }
                        float total_9_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_4 = 0; src_1_4 < 8; src_1_4++) {
                            total_9_0 = total_9_0 + smem_red[src_1_4 * 1152 + lane_row * 9 + 4];
                        }
                        int tok_c_10_0 = tok0 + 4;
                        if (feature < n_valid && tok_c_10_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_0;
                        }
                        float total_11_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_5 = 0; src_1_5 < 8; src_1_5++) {
                            total_11_0 = total_11_0 + smem_red[src_1_5 * 1152 + lane_row * 9 + 5];
                        }
                        int tok_c_12_0 = tok0 + 5;
                        if (feature < n_valid && tok_c_12_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_0;
                        }
                        float total_13_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_6 = 0; src_1_6 < 8; src_1_6++) {
                            total_13_0 = total_13_0 + smem_red[src_1_6 * 1152 + lane_row * 9 + 6];
                        }
                        int tok_c_14_0 = tok0 + 6;
                        if (feature < n_valid && tok_c_14_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_0;
                        }
                        float total_14_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_7 = 0; src_1_7 < 8; src_1_7++) {
                            total_14_0 = total_14_0 + smem_red[src_1_7 * 1152 + lane_row * 9 + 7];
                        }
                        int tok_c_15_0 = tok0 + 7;
                        if (feature < n_valid && tok_c_15_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_15_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_14_0;
                        }
                    }
                    if (1 == my_rank) {
                        float total_1_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_8 = 0; src_1_8 < 8; src_1_8++) {
                            total_1_1 = total_1_1 + smem_red[src_1_8 * 1152 + lane_row * 9];
                        }
                        int tok_c_2_1 = tok0 + 8;
                        if (feature < n_valid && tok_c_2_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_1;
                        }
                        float total_3_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_9 = 0; src_1_9 < 8; src_1_9++) {
                            total_3_1 = total_3_1 + smem_red[src_1_9 * 1152 + lane_row * 9 + 1];
                        }
                        int tok_c_4_1 = tok0 + 9;
                        if (feature < n_valid && tok_c_4_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_1;
                        }
                        float total_5_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_10 = 0; src_1_10 < 8; src_1_10++) {
                            total_5_1 = total_5_1 + smem_red[src_1_10 * 1152 + lane_row * 9 + 2];
                        }
                        int tok_c_6_1 = tok0 + 10;
                        if (feature < n_valid && tok_c_6_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_1;
                        }
                        float total_7_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_11 = 0; src_1_11 < 8; src_1_11++) {
                            total_7_1 = total_7_1 + smem_red[src_1_11 * 1152 + lane_row * 9 + 3];
                        }
                        int tok_c_8_1 = tok0 + 11;
                        if (feature < n_valid && tok_c_8_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_1;
                        }
                        float total_9_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_12 = 0; src_1_12 < 8; src_1_12++) {
                            total_9_1 = total_9_1 + smem_red[src_1_12 * 1152 + lane_row * 9 + 4];
                        }
                        int tok_c_10_1 = tok0 + 12;
                        if (feature < n_valid && tok_c_10_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_1;
                        }
                        float total_11_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_13 = 0; src_1_13 < 8; src_1_13++) {
                            total_11_1 = total_11_1 + smem_red[src_1_13 * 1152 + lane_row * 9 + 5];
                        }
                        int tok_c_12_1 = tok0 + 13;
                        if (feature < n_valid && tok_c_12_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_1;
                        }
                        float total_13_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_14 = 0; src_1_14 < 8; src_1_14++) {
                            total_13_1 = total_13_1 + smem_red[src_1_14 * 1152 + lane_row * 9 + 6];
                        }
                        int tok_c_14_1 = tok0 + 14;
                        if (feature < n_valid && tok_c_14_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_1;
                        }
                        float total_14_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_15 = 0; src_1_15 < 8; src_1_15++) {
                            total_14_1 = total_14_1 + smem_red[src_1_15 * 1152 + lane_row * 9 + 7];
                        }
                        int tok_c_15_1 = tok0 + 15;
                        if (feature < n_valid && tok_c_15_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_15_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_14_1;
                        }
                    }
                    if (2 == my_rank) {
                        float total_1_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_16 = 0; src_1_16 < 8; src_1_16++) {
                            total_1_2 = total_1_2 + smem_red[src_1_16 * 1152 + lane_row * 9];
                        }
                        int tok_c_2_2 = tok0 + 16;
                        if (feature < n_valid && tok_c_2_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_2;
                        }
                        float total_3_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_17 = 0; src_1_17 < 8; src_1_17++) {
                            total_3_2 = total_3_2 + smem_red[src_1_17 * 1152 + lane_row * 9 + 1];
                        }
                        int tok_c_4_2 = tok0 + 17;
                        if (feature < n_valid && tok_c_4_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_2;
                        }
                        float total_5_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_18 = 0; src_1_18 < 8; src_1_18++) {
                            total_5_2 = total_5_2 + smem_red[src_1_18 * 1152 + lane_row * 9 + 2];
                        }
                        int tok_c_6_2 = tok0 + 18;
                        if (feature < n_valid && tok_c_6_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_2;
                        }
                        float total_7_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_19 = 0; src_1_19 < 8; src_1_19++) {
                            total_7_2 = total_7_2 + smem_red[src_1_19 * 1152 + lane_row * 9 + 3];
                        }
                        int tok_c_8_2 = tok0 + 19;
                        if (feature < n_valid && tok_c_8_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_2;
                        }
                        float total_9_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_20 = 0; src_1_20 < 8; src_1_20++) {
                            total_9_2 = total_9_2 + smem_red[src_1_20 * 1152 + lane_row * 9 + 4];
                        }
                        int tok_c_10_2 = tok0 + 20;
                        if (feature < n_valid && tok_c_10_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_2;
                        }
                        float total_11_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_21 = 0; src_1_21 < 8; src_1_21++) {
                            total_11_2 = total_11_2 + smem_red[src_1_21 * 1152 + lane_row * 9 + 5];
                        }
                        int tok_c_12_2 = tok0 + 21;
                        if (feature < n_valid && tok_c_12_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_2;
                        }
                        float total_13_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_22 = 0; src_1_22 < 8; src_1_22++) {
                            total_13_2 = total_13_2 + smem_red[src_1_22 * 1152 + lane_row * 9 + 6];
                        }
                        int tok_c_14_2 = tok0 + 22;
                        if (feature < n_valid && tok_c_14_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_2;
                        }
                        float total_14_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_23 = 0; src_1_23 < 8; src_1_23++) {
                            total_14_2 = total_14_2 + smem_red[src_1_23 * 1152 + lane_row * 9 + 7];
                        }
                        int tok_c_15_2 = tok0 + 23;
                        if (feature < n_valid && tok_c_15_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_15_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_14_2;
                        }
                    }
                    if (3 == my_rank) {
                        float total_1_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_24 = 0; src_1_24 < 8; src_1_24++) {
                            total_1_3 = total_1_3 + smem_red[src_1_24 * 1152 + lane_row * 9];
                        }
                        int tok_c_2_3 = tok0 + 24;
                        if (feature < n_valid && tok_c_2_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_3;
                        }
                        float total_3_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_25 = 0; src_1_25 < 8; src_1_25++) {
                            total_3_3 = total_3_3 + smem_red[src_1_25 * 1152 + lane_row * 9 + 1];
                        }
                        int tok_c_4_3 = tok0 + 25;
                        if (feature < n_valid && tok_c_4_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_3;
                        }
                        float total_5_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_26 = 0; src_1_26 < 8; src_1_26++) {
                            total_5_3 = total_5_3 + smem_red[src_1_26 * 1152 + lane_row * 9 + 2];
                        }
                        int tok_c_6_3 = tok0 + 26;
                        if (feature < n_valid && tok_c_6_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_3;
                        }
                        float total_7_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_27 = 0; src_1_27 < 8; src_1_27++) {
                            total_7_3 = total_7_3 + smem_red[src_1_27 * 1152 + lane_row * 9 + 3];
                        }
                        int tok_c_8_3 = tok0 + 27;
                        if (feature < n_valid && tok_c_8_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_3;
                        }
                        float total_9_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_28 = 0; src_1_28 < 8; src_1_28++) {
                            total_9_3 = total_9_3 + smem_red[src_1_28 * 1152 + lane_row * 9 + 4];
                        }
                        int tok_c_10_3 = tok0 + 28;
                        if (feature < n_valid && tok_c_10_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_3;
                        }
                        float total_11_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_29 = 0; src_1_29 < 8; src_1_29++) {
                            total_11_3 = total_11_3 + smem_red[src_1_29 * 1152 + lane_row * 9 + 5];
                        }
                        int tok_c_12_3 = tok0 + 29;
                        if (feature < n_valid && tok_c_12_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_3;
                        }
                        float total_13_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_30 = 0; src_1_30 < 8; src_1_30++) {
                            total_13_3 = total_13_3 + smem_red[src_1_30 * 1152 + lane_row * 9 + 6];
                        }
                        int tok_c_14_3 = tok0 + 30;
                        if (feature < n_valid && tok_c_14_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_3;
                        }
                        float total_14_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_31 = 0; src_1_31 < 8; src_1_31++) {
                            total_14_3 = total_14_3 + smem_red[src_1_31 * 1152 + lane_row * 9 + 7];
                        }
                        int tok_c_15_3 = tok0 + 31;
                        if (feature < n_valid && tok_c_15_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_15_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_14_3;
                        }
                    }
                    if (4 == my_rank) {
                        float total_1_4 = 0.0f;
                        #pragma unroll
                        for (int src_1_32 = 0; src_1_32 < 8; src_1_32++) {
                            total_1_4 = total_1_4 + smem_red[src_1_32 * 1152 + lane_row * 9];
                        }
                        int tok_c_2_4 = tok0 + 32;
                        if (feature < n_valid && tok_c_2_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_4 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_4;
                        }
                        float total_3_4 = 0.0f;
                        #pragma unroll
                        for (int src_1_33 = 0; src_1_33 < 8; src_1_33++) {
                            total_3_4 = total_3_4 + smem_red[src_1_33 * 1152 + lane_row * 9 + 1];
                        }
                        int tok_c_4_4 = tok0 + 33;
                        if (feature < n_valid && tok_c_4_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_4 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_4;
                        }
                        float total_5_4 = 0.0f;
                        #pragma unroll
                        for (int src_1_34 = 0; src_1_34 < 8; src_1_34++) {
                            total_5_4 = total_5_4 + smem_red[src_1_34 * 1152 + lane_row * 9 + 2];
                        }
                        int tok_c_6_4 = tok0 + 34;
                        if (feature < n_valid && tok_c_6_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_4 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_4;
                        }
                        float total_7_4 = 0.0f;
                        #pragma unroll
                        for (int src_1_35 = 0; src_1_35 < 8; src_1_35++) {
                            total_7_4 = total_7_4 + smem_red[src_1_35 * 1152 + lane_row * 9 + 3];
                        }
                        int tok_c_8_4 = tok0 + 35;
                        if (feature < n_valid && tok_c_8_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_4 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_4;
                        }
                        float total_9_4 = 0.0f;
                        #pragma unroll
                        for (int src_1_36 = 0; src_1_36 < 8; src_1_36++) {
                            total_9_4 = total_9_4 + smem_red[src_1_36 * 1152 + lane_row * 9 + 4];
                        }
                        int tok_c_10_4 = tok0 + 36;
                        if (feature < n_valid && tok_c_10_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_4 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_4;
                        }
                        float total_11_4 = 0.0f;
                        #pragma unroll
                        for (int src_1_37 = 0; src_1_37 < 8; src_1_37++) {
                            total_11_4 = total_11_4 + smem_red[src_1_37 * 1152 + lane_row * 9 + 5];
                        }
                        int tok_c_12_4 = tok0 + 37;
                        if (feature < n_valid && tok_c_12_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_4 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_4;
                        }
                        float total_13_4 = 0.0f;
                        #pragma unroll
                        for (int src_1_38 = 0; src_1_38 < 8; src_1_38++) {
                            total_13_4 = total_13_4 + smem_red[src_1_38 * 1152 + lane_row * 9 + 6];
                        }
                        int tok_c_14_4 = tok0 + 38;
                        if (feature < n_valid && tok_c_14_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_4 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_4;
                        }
                        float total_14_4 = 0.0f;
                        #pragma unroll
                        for (int src_1_39 = 0; src_1_39 < 8; src_1_39++) {
                            total_14_4 = total_14_4 + smem_red[src_1_39 * 1152 + lane_row * 9 + 7];
                        }
                        int tok_c_15_4 = tok0 + 39;
                        if (feature < n_valid && tok_c_15_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_15_4 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_14_4;
                        }
                    }
                    if (5 == my_rank) {
                        float total_1_5 = 0.0f;
                        #pragma unroll
                        for (int src_1_40 = 0; src_1_40 < 8; src_1_40++) {
                            total_1_5 = total_1_5 + smem_red[src_1_40 * 1152 + lane_row * 9];
                        }
                        int tok_c_2_5 = tok0 + 40;
                        if (feature < n_valid && tok_c_2_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_5 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_5;
                        }
                        float total_3_5 = 0.0f;
                        #pragma unroll
                        for (int src_1_41 = 0; src_1_41 < 8; src_1_41++) {
                            total_3_5 = total_3_5 + smem_red[src_1_41 * 1152 + lane_row * 9 + 1];
                        }
                        int tok_c_4_5 = tok0 + 41;
                        if (feature < n_valid && tok_c_4_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_5 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_5;
                        }
                        float total_5_5 = 0.0f;
                        #pragma unroll
                        for (int src_1_42 = 0; src_1_42 < 8; src_1_42++) {
                            total_5_5 = total_5_5 + smem_red[src_1_42 * 1152 + lane_row * 9 + 2];
                        }
                        int tok_c_6_5 = tok0 + 42;
                        if (feature < n_valid && tok_c_6_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_5 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_5;
                        }
                        float total_7_5 = 0.0f;
                        #pragma unroll
                        for (int src_1_43 = 0; src_1_43 < 8; src_1_43++) {
                            total_7_5 = total_7_5 + smem_red[src_1_43 * 1152 + lane_row * 9 + 3];
                        }
                        int tok_c_8_5 = tok0 + 43;
                        if (feature < n_valid && tok_c_8_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_5 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_5;
                        }
                        float total_9_5 = 0.0f;
                        #pragma unroll
                        for (int src_1_44 = 0; src_1_44 < 8; src_1_44++) {
                            total_9_5 = total_9_5 + smem_red[src_1_44 * 1152 + lane_row * 9 + 4];
                        }
                        int tok_c_10_5 = tok0 + 44;
                        if (feature < n_valid && tok_c_10_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_5 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_5;
                        }
                        float total_11_5 = 0.0f;
                        #pragma unroll
                        for (int src_1_45 = 0; src_1_45 < 8; src_1_45++) {
                            total_11_5 = total_11_5 + smem_red[src_1_45 * 1152 + lane_row * 9 + 5];
                        }
                        int tok_c_12_5 = tok0 + 45;
                        if (feature < n_valid && tok_c_12_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_5 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_5;
                        }
                        float total_13_5 = 0.0f;
                        #pragma unroll
                        for (int src_1_46 = 0; src_1_46 < 8; src_1_46++) {
                            total_13_5 = total_13_5 + smem_red[src_1_46 * 1152 + lane_row * 9 + 6];
                        }
                        int tok_c_14_5 = tok0 + 46;
                        if (feature < n_valid && tok_c_14_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_5 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_5;
                        }
                        float total_14_5 = 0.0f;
                        #pragma unroll
                        for (int src_1_47 = 0; src_1_47 < 8; src_1_47++) {
                            total_14_5 = total_14_5 + smem_red[src_1_47 * 1152 + lane_row * 9 + 7];
                        }
                        int tok_c_15_5 = tok0 + 47;
                        if (feature < n_valid && tok_c_15_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_15_5 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_14_5;
                        }
                    }
                    if (6 == my_rank) {
                        float total_1_6 = 0.0f;
                        #pragma unroll
                        for (int src_1_48 = 0; src_1_48 < 8; src_1_48++) {
                            total_1_6 = total_1_6 + smem_red[src_1_48 * 1152 + lane_row * 9];
                        }
                        int tok_c_2_6 = tok0 + 48;
                        if (feature < n_valid && tok_c_2_6 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_6 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_6;
                        }
                        float total_3_6 = 0.0f;
                        #pragma unroll
                        for (int src_1_49 = 0; src_1_49 < 8; src_1_49++) {
                            total_3_6 = total_3_6 + smem_red[src_1_49 * 1152 + lane_row * 9 + 1];
                        }
                        int tok_c_4_6 = tok0 + 49;
                        if (feature < n_valid && tok_c_4_6 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_6 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_6;
                        }
                        float total_5_6 = 0.0f;
                        #pragma unroll
                        for (int src_1_50 = 0; src_1_50 < 8; src_1_50++) {
                            total_5_6 = total_5_6 + smem_red[src_1_50 * 1152 + lane_row * 9 + 2];
                        }
                        int tok_c_6_6 = tok0 + 50;
                        if (feature < n_valid && tok_c_6_6 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_6 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_6;
                        }
                        float total_7_6 = 0.0f;
                        #pragma unroll
                        for (int src_1_51 = 0; src_1_51 < 8; src_1_51++) {
                            total_7_6 = total_7_6 + smem_red[src_1_51 * 1152 + lane_row * 9 + 3];
                        }
                        int tok_c_8_6 = tok0 + 51;
                        if (feature < n_valid && tok_c_8_6 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_6 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_6;
                        }
                        float total_9_6 = 0.0f;
                        #pragma unroll
                        for (int src_1_52 = 0; src_1_52 < 8; src_1_52++) {
                            total_9_6 = total_9_6 + smem_red[src_1_52 * 1152 + lane_row * 9 + 4];
                        }
                        int tok_c_10_6 = tok0 + 52;
                        if (feature < n_valid && tok_c_10_6 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_6 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_6;
                        }
                        float total_11_6 = 0.0f;
                        #pragma unroll
                        for (int src_1_53 = 0; src_1_53 < 8; src_1_53++) {
                            total_11_6 = total_11_6 + smem_red[src_1_53 * 1152 + lane_row * 9 + 5];
                        }
                        int tok_c_12_6 = tok0 + 53;
                        if (feature < n_valid && tok_c_12_6 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_6 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_6;
                        }
                        float total_13_6 = 0.0f;
                        #pragma unroll
                        for (int src_1_54 = 0; src_1_54 < 8; src_1_54++) {
                            total_13_6 = total_13_6 + smem_red[src_1_54 * 1152 + lane_row * 9 + 6];
                        }
                        int tok_c_14_6 = tok0 + 54;
                        if (feature < n_valid && tok_c_14_6 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_6 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_6;
                        }
                        float total_14_6 = 0.0f;
                        #pragma unroll
                        for (int src_1_55 = 0; src_1_55 < 8; src_1_55++) {
                            total_14_6 = total_14_6 + smem_red[src_1_55 * 1152 + lane_row * 9 + 7];
                        }
                        int tok_c_15_6 = tok0 + 55;
                        if (feature < n_valid && tok_c_15_6 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_15_6 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_14_6;
                        }
                    }
                    if (7 == my_rank) {
                        float total_1_7 = 0.0f;
                        #pragma unroll
                        for (int src_1_56 = 0; src_1_56 < 8; src_1_56++) {
                            total_1_7 = total_1_7 + smem_red[src_1_56 * 1152 + lane_row * 9];
                        }
                        int tok_c_2_7 = tok0 + 56;
                        if (feature < n_valid && tok_c_2_7 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_7 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_7;
                        }
                        float total_3_7 = 0.0f;
                        #pragma unroll
                        for (int src_1_57 = 0; src_1_57 < 8; src_1_57++) {
                            total_3_7 = total_3_7 + smem_red[src_1_57 * 1152 + lane_row * 9 + 1];
                        }
                        int tok_c_4_7 = tok0 + 57;
                        if (feature < n_valid && tok_c_4_7 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_7 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_7;
                        }
                        float total_5_7 = 0.0f;
                        #pragma unroll
                        for (int src_1_58 = 0; src_1_58 < 8; src_1_58++) {
                            total_5_7 = total_5_7 + smem_red[src_1_58 * 1152 + lane_row * 9 + 2];
                        }
                        int tok_c_6_7 = tok0 + 58;
                        if (feature < n_valid && tok_c_6_7 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_7 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_7;
                        }
                        float total_7_7 = 0.0f;
                        #pragma unroll
                        for (int src_1_59 = 0; src_1_59 < 8; src_1_59++) {
                            total_7_7 = total_7_7 + smem_red[src_1_59 * 1152 + lane_row * 9 + 3];
                        }
                        int tok_c_8_7 = tok0 + 59;
                        if (feature < n_valid && tok_c_8_7 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_7 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_7;
                        }
                        float total_9_7 = 0.0f;
                        #pragma unroll
                        for (int src_1_60 = 0; src_1_60 < 8; src_1_60++) {
                            total_9_7 = total_9_7 + smem_red[src_1_60 * 1152 + lane_row * 9 + 4];
                        }
                        int tok_c_10_7 = tok0 + 60;
                        if (feature < n_valid && tok_c_10_7 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_7 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_7;
                        }
                        float total_11_7 = 0.0f;
                        #pragma unroll
                        for (int src_1_61 = 0; src_1_61 < 8; src_1_61++) {
                            total_11_7 = total_11_7 + smem_red[src_1_61 * 1152 + lane_row * 9 + 5];
                        }
                        int tok_c_12_7 = tok0 + 61;
                        if (feature < n_valid && tok_c_12_7 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_7 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_7;
                        }
                        float total_13_7 = 0.0f;
                        #pragma unroll
                        for (int src_1_62 = 0; src_1_62 < 8; src_1_62++) {
                            total_13_7 = total_13_7 + smem_red[src_1_62 * 1152 + lane_row * 9 + 6];
                        }
                        int tok_c_14_7 = tok0 + 62;
                        if (feature < n_valid && tok_c_14_7 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_7 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_7;
                        }
                        float total_14_7 = 0.0f;
                        #pragma unroll
                        for (int src_1_63 = 0; src_1_63 < 8; src_1_63++) {
                            total_14_7 = total_14_7 + smem_red[src_1_63 * 1152 + lane_row * 9 + 7];
                        }
                        int tok_c_15_7 = tok0 + 63;
                        if (feature < n_valid && tok_c_15_7 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_15_7 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_14_7;
                        }
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (n_items_e > it_e + 1) {
                    if (tid == 64) {
                        #pragma unroll
                        for (int p = 0; p < 8; p++) {
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
