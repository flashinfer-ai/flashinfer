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
#define TMEM_NCOLS 144
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFW_OFFSET 128
#define TMEM_TMEM_SFX_OFFSET 136
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAINLOOP_PIPE_STAGES 2
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
#define SMEM_SMEM_EPI_OFF 205824
#define SMEM_SMEM_EPI_STAGE_BYTES 16384
#define SMEM_SMEM_EPI_STRIDE 16384
#define SMEM_SMEM_RED_OFF 1024
#define SMEM_SMEM_RED_STAGE_BYTES 33792
#define SMEM_SMEM_RED_STRIDE 33792
#define SMEM_SMEM_STG_OFF 34816
#define SMEM_SMEM_STG_STAGE_BYTES 16896
#define SMEM_SMEM_STG_STRIDE 16896
#define SMEM_TOTAL 222208
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

__global__ __launch_bounds__(192) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_fp8_projection_e2e5a8209d576711ba08(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, const __grid_constant__ CUtensorMap SFW, const __grid_constant__ CUtensorMap SFX, __nv_bfloat16* __restrict__ out, float* __restrict__ partials, unsigned int* __restrict__ counters, int M, int n_tiles, int n_valid, int ldo, int num_k_iters, int sf_k_tiles, int split, int tok_per_cta, int total_work, int store_vec, __nv_bfloat16* __restrict__ x, int K, const __grid_constant__ CUtensorMap XB)
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
    #define epilogue_done_addr (mbar_base + 80)
    #define red_full_addr (mbar_base + 96)
    #define peers_free_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
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
    uint8_t* smem_epi = reinterpret_cast<uint8_t*>(smem_raw + 205824);
    const int smem_epi_addr = smem + 205824;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_red_addr = smem + 1024;
    float* smem_stg = reinterpret_cast<float*>(smem_raw + 34816);
    const int smem_stg_addr = smem + 34816;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 14 barriers)
    // Mbarriers at smem_raw[0..112)

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
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // epilogue_done: 2 barriers, init_count=4
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            // --- pipeline 'red_pipe' ---
            // red_full: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // peers_free: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 144 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 112);
    if (warp == 0) {
        int _tmem_hold = smem + 112;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
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
    const int tmem_tmem_sfw = taddr + 128;
    const int tmem_tmem_sfx = taddr + 136;

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
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_w_addr + load_stage * 51200), "l"((&W)), "r"(0), "r"(0), "r"(w_tile0 + k_group),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_sfw0_addr + load_stage * 51200), "l"((&SFW)), "r"(0), "r"(0), "r"(sfw_unit0 + k_group * 2),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_sfw1_addr + load_stage * 51200), "l"((&SFW)), "r"(0), "r"(0), "r"(sfw_unit0 + k_group * 2 + 2),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        if (it == 0 && i == 0) {
                            asm volatile("griddepcontrol.wait;" ::: "memory");
                        }
                        tma_3d_gmem2smem(smem_x_addr + load_stage * 51200, (&X), 0, x_row, k_group, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_sfx_all_addr + load_stage * 51200, (&SFX), 0, 0, sfx_unit0 + k_group, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 51200);
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

                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (acc_stage * 64)), a_desc + 0, b_desc + 0,
                            0x8900000U, tmem_tmem_sfw, tmem_tmem_sfx, ((init_flag) ? 0 : 1), _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (acc_stage * 64)), a_desc + 2, b_desc + 2,
                            0x28900010U, tmem_tmem_sfw, tmem_tmem_sfx, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (acc_stage * 64)), a_desc + 4, b_desc + 4,
                            0x48900020U, tmem_tmem_sfw, tmem_tmem_sfx, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (acc_stage * 64)), a_desc + 6, b_desc + 6,
                            0x68900030U, tmem_tmem_sfw, tmem_tmem_sfx, 1, _elected_0);
                    }
                    int init2 = 0;
                    int _smem_desc_lo_3 = (((smem_w_addr + 16384) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                    int _smem_desc_lo_4 = (((smem_x_addr + 8192) >> 4) & 0x3FFF) + (mma_stage) * 3200;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (acc_stage * 64)), a_desc + 0, b_desc + 0,
                            0x8900000U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, ((init2) ? 0 : 1), _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (acc_stage * 64)), a_desc + 2, b_desc + 2,
                            0x28900010U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (acc_stage * 64)), a_desc + 4, b_desc + 4,
                            0x48900020U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1, _elected_0);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (acc_stage * 64)), a_desc + 6, b_desc + 6,
                            0x68900030U, tmem_tmem_sfw + 4, tmem_tmem_sfx + 4, 1, _elected_0);
                    }
                    elect_commit(mma_done_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 4) { mma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(mainloop_done_addr + (acc_stage) * 8);
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_epilogue_done ^= 1; }
            }
            mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
            acc_stage += 1;
            if (acc_stage == 2) { acc_stage = 0; _phase_epilogue_done ^= 1; }
            mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
            __syncwarp();
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
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
            unsigned int acc_stage_e = 0;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_peers_free = 0;
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
                mbarrier_wait(mainloop_done_addr + (acc_stage_e) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (tid == 64) {
                    #pragma unroll
                    for (int p_a = 0; p_a < 2; p_a++) {
                        if (p_a != cta_rank) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(peers_free_addr), "r"(p_a) : "memory");
                        }
                    }
                }
                int acc_col_e = it_e % 2 * 64;
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
                    mbarrier_arrive(epilogue_done_addr + (acc_stage_e) * 8);
                }
                acc_stage_e += 1;
                if (acc_stage_e == 2) { acc_stage_e = 0; _phase_mainloop_done ^= 1; }
                unsigned int epi_base = smem_epi_addr + (unsigned int)(epi_group * 16384);
                int my_rank = cta_rank;
                int _min_0 = ((64) < ((my_rank + 1) * 32) ? (64) : ((my_rank + 1) * 32));
                int rows_mine = _min_0 - my_rank * 32;
                unsigned int line_b = (unsigned int)(lane_row * 132);
                unsigned int slot_b = (unsigned int)(my_rank * 16896);
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
                        {
                            uint32_t _addr_12 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 44));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_12), "f"(_tmem_load_0[11]) : "memory");
                        }
                        {
                            uint32_t _addr_13 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 48));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_13), "f"(_tmem_load_0[12]) : "memory");
                        }
                        {
                            uint32_t _addr_14 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 52));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_14), "f"(_tmem_load_0[13]) : "memory");
                        }
                        {
                            uint32_t _addr_15 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 56));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_15), "f"(_tmem_load_0[14]) : "memory");
                        }
                        {
                            uint32_t _addr_16 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 60));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_16), "f"(_tmem_load_0[15]) : "memory");
                        }
                        {
                            uint32_t _addr_17 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 64));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(_tmem_load_0[16]) : "memory");
                        }
                        {
                            uint32_t _addr_18 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 68));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_18), "f"(_tmem_load_0[17]) : "memory");
                        }
                        {
                            uint32_t _addr_19 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 72));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(_tmem_load_0[18]) : "memory");
                        }
                        {
                            uint32_t _addr_20 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 76));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_20), "f"(_tmem_load_0[19]) : "memory");
                        }
                        {
                            uint32_t _addr_21 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 80));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_21), "f"(_tmem_load_0[20]) : "memory");
                        }
                        {
                            uint32_t _addr_22 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 84));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_22), "f"(_tmem_load_0[21]) : "memory");
                        }
                        {
                            uint32_t _addr_23 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 88));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(_tmem_load_0[22]) : "memory");
                        }
                        {
                            uint32_t _addr_24 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 92));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_24), "f"(_tmem_load_0[23]) : "memory");
                        }
                        {
                            uint32_t _addr_25 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 96));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_25), "f"(_tmem_load_0[24]) : "memory");
                        }
                        {
                            uint32_t _addr_26 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 100));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_26), "f"(_tmem_load_0[25]) : "memory");
                        }
                        {
                            uint32_t _addr_27 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 104));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_27), "f"(_tmem_load_0[26]) : "memory");
                        }
                        {
                            uint32_t _addr_28 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 108));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_28), "f"(_tmem_load_0[27]) : "memory");
                        }
                        {
                            uint32_t _addr_29 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 112));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_29), "f"(_tmem_load_0[28]) : "memory");
                        }
                        {
                            uint32_t _addr_30 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 116));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_30), "f"(_tmem_load_0[29]) : "memory");
                        }
                        {
                            uint32_t _addr_31 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 120));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_31), "f"(_tmem_load_0[30]) : "memory");
                        }
                        {
                            uint32_t _addr_32 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 124));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_32), "f"(_tmem_load_0[31]) : "memory");
                        }
                    } else {
                        int _max_34 = ((0) > (-my_rank) ? (0) : (-my_rank));
                        int _min_33 = ((1) < (_max_34) ? (1) : (_max_34));
                        int o_idx = -_min_33;
                        unsigned int stg_b = (unsigned int)(o_idx * 16896) + line_b;
                        {
                            uint32_t _addr_35 = static_cast<uint32_t>(smem_stg_addr + stg_b);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_35), "f"(_tmem_load_0[0]) : "memory");
                        }
                        {
                            uint32_t _addr_36 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_36), "f"(_tmem_load_0[1]) : "memory");
                        }
                        {
                            uint32_t _addr_37 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_37), "f"(_tmem_load_0[2]) : "memory");
                        }
                        {
                            uint32_t _addr_38 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_38), "f"(_tmem_load_0[3]) : "memory");
                        }
                        {
                            uint32_t _addr_39 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_39), "f"(_tmem_load_0[4]) : "memory");
                        }
                        {
                            uint32_t _addr_40 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_40), "f"(_tmem_load_0[5]) : "memory");
                        }
                        {
                            uint32_t _addr_41 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_41), "f"(_tmem_load_0[6]) : "memory");
                        }
                        {
                            uint32_t _addr_42 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_42), "f"(_tmem_load_0[7]) : "memory");
                        }
                        {
                            uint32_t _addr_43 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_43), "f"(_tmem_load_0[8]) : "memory");
                        }
                        {
                            uint32_t _addr_44 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_44), "f"(_tmem_load_0[9]) : "memory");
                        }
                        {
                            uint32_t _addr_45 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_45), "f"(_tmem_load_0[10]) : "memory");
                        }
                        {
                            uint32_t _addr_46 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 44));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_46), "f"(_tmem_load_0[11]) : "memory");
                        }
                        {
                            uint32_t _addr_47 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 48));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_47), "f"(_tmem_load_0[12]) : "memory");
                        }
                        {
                            uint32_t _addr_48 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 52));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_48), "f"(_tmem_load_0[13]) : "memory");
                        }
                        {
                            uint32_t _addr_49 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 56));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_49), "f"(_tmem_load_0[14]) : "memory");
                        }
                        {
                            uint32_t _addr_50 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 60));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_50), "f"(_tmem_load_0[15]) : "memory");
                        }
                        {
                            uint32_t _addr_51 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 64));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_51), "f"(_tmem_load_0[16]) : "memory");
                        }
                        {
                            uint32_t _addr_52 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 68));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_52), "f"(_tmem_load_0[17]) : "memory");
                        }
                        {
                            uint32_t _addr_53 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 72));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_53), "f"(_tmem_load_0[18]) : "memory");
                        }
                        {
                            uint32_t _addr_54 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 76));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_54), "f"(_tmem_load_0[19]) : "memory");
                        }
                        {
                            uint32_t _addr_55 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 80));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_55), "f"(_tmem_load_0[20]) : "memory");
                        }
                        {
                            uint32_t _addr_56 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 84));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_56), "f"(_tmem_load_0[21]) : "memory");
                        }
                        {
                            uint32_t _addr_57 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 88));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_57), "f"(_tmem_load_0[22]) : "memory");
                        }
                        {
                            uint32_t _addr_58 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 92));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_58), "f"(_tmem_load_0[23]) : "memory");
                        }
                        {
                            uint32_t _addr_59 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 96));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_59), "f"(_tmem_load_0[24]) : "memory");
                        }
                        {
                            uint32_t _addr_60 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 100));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_60), "f"(_tmem_load_0[25]) : "memory");
                        }
                        {
                            uint32_t _addr_61 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 104));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_61), "f"(_tmem_load_0[26]) : "memory");
                        }
                        {
                            uint32_t _addr_62 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 108));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_62), "f"(_tmem_load_0[27]) : "memory");
                        }
                        {
                            uint32_t _addr_63 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 112));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_63), "f"(_tmem_load_0[28]) : "memory");
                        }
                        {
                            uint32_t _addr_64 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 116));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_64), "f"(_tmem_load_0[29]) : "memory");
                        }
                        {
                            uint32_t _addr_65 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 120));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_65), "f"(_tmem_load_0[30]) : "memory");
                        }
                        {
                            uint32_t _addr_66 = static_cast<uint32_t>(smem_stg_addr + (stg_b + 124));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_66), "f"(_tmem_load_0[31]) : "memory");
                        }
                    }
                    if (1 == my_rank) {
                        {
                            uint32_t _addr_67 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_67), "f"(_tmem_load_0[32]) : "memory");
                        }
                        {
                            uint32_t _addr_68 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_68), "f"(_tmem_load_0[33]) : "memory");
                        }
                        {
                            uint32_t _addr_69 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_69), "f"(_tmem_load_0[34]) : "memory");
                        }
                        {
                            uint32_t _addr_70 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_70), "f"(_tmem_load_0[35]) : "memory");
                        }
                        {
                            uint32_t _addr_71 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_71), "f"(_tmem_load_0[36]) : "memory");
                        }
                        {
                            uint32_t _addr_72 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_72), "f"(_tmem_load_0[37]) : "memory");
                        }
                        {
                            uint32_t _addr_73 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_73), "f"(_tmem_load_0[38]) : "memory");
                        }
                        {
                            uint32_t _addr_74 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_74), "f"(_tmem_load_0[39]) : "memory");
                        }
                        {
                            uint32_t _addr_75 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_75), "f"(_tmem_load_0[40]) : "memory");
                        }
                        {
                            uint32_t _addr_76 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_76), "f"(_tmem_load_0[41]) : "memory");
                        }
                        {
                            uint32_t _addr_77 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_77), "f"(_tmem_load_0[42]) : "memory");
                        }
                        {
                            uint32_t _addr_78 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 44));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_78), "f"(_tmem_load_0[43]) : "memory");
                        }
                        {
                            uint32_t _addr_79 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 48));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_79), "f"(_tmem_load_0[44]) : "memory");
                        }
                        {
                            uint32_t _addr_80 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 52));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_80), "f"(_tmem_load_0[45]) : "memory");
                        }
                        {
                            uint32_t _addr_81 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 56));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_81), "f"(_tmem_load_0[46]) : "memory");
                        }
                        {
                            uint32_t _addr_82 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 60));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_82), "f"(_tmem_load_0[47]) : "memory");
                        }
                        {
                            uint32_t _addr_83 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 64));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_83), "f"(_tmem_load_0[48]) : "memory");
                        }
                        {
                            uint32_t _addr_84 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 68));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_84), "f"(_tmem_load_0[49]) : "memory");
                        }
                        {
                            uint32_t _addr_85 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 72));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_85), "f"(_tmem_load_0[50]) : "memory");
                        }
                        {
                            uint32_t _addr_86 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 76));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_86), "f"(_tmem_load_0[51]) : "memory");
                        }
                        {
                            uint32_t _addr_87 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 80));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_87), "f"(_tmem_load_0[52]) : "memory");
                        }
                        {
                            uint32_t _addr_88 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 84));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_88), "f"(_tmem_load_0[53]) : "memory");
                        }
                        {
                            uint32_t _addr_89 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 88));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_89), "f"(_tmem_load_0[54]) : "memory");
                        }
                        {
                            uint32_t _addr_90 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 92));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_90), "f"(_tmem_load_0[55]) : "memory");
                        }
                        {
                            uint32_t _addr_91 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 96));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_91), "f"(_tmem_load_0[56]) : "memory");
                        }
                        {
                            uint32_t _addr_92 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 100));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_92), "f"(_tmem_load_0[57]) : "memory");
                        }
                        {
                            uint32_t _addr_93 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 104));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_93), "f"(_tmem_load_0[58]) : "memory");
                        }
                        {
                            uint32_t _addr_94 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 108));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_94), "f"(_tmem_load_0[59]) : "memory");
                        }
                        {
                            uint32_t _addr_95 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 112));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_95), "f"(_tmem_load_0[60]) : "memory");
                        }
                        {
                            uint32_t _addr_96 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 116));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_96), "f"(_tmem_load_0[61]) : "memory");
                        }
                        {
                            uint32_t _addr_97 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 120));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_97), "f"(_tmem_load_0[62]) : "memory");
                        }
                        {
                            uint32_t _addr_98 = static_cast<uint32_t>(smem_red_addr + (slot_b + line_b + 124));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_98), "f"(_tmem_load_0[63]) : "memory");
                        }
                    } else {
                        int _max_100 = ((0) > (1 - my_rank) ? (0) : (1 - my_rank));
                        int _min_99 = ((1) < (_max_100) ? (1) : (_max_100));
                        int o_idx_1 = 1 - _min_99;
                        unsigned int stg_b_1 = (unsigned int)(o_idx_1 * 16896) + line_b;
                        {
                            uint32_t _addr_101 = static_cast<uint32_t>(smem_stg_addr + stg_b_1);
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_101), "f"(_tmem_load_0[32]) : "memory");
                        }
                        {
                            uint32_t _addr_102 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 4));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_102), "f"(_tmem_load_0[33]) : "memory");
                        }
                        {
                            uint32_t _addr_103 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 8));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_103), "f"(_tmem_load_0[34]) : "memory");
                        }
                        {
                            uint32_t _addr_104 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 12));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_104), "f"(_tmem_load_0[35]) : "memory");
                        }
                        {
                            uint32_t _addr_105 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 16));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_105), "f"(_tmem_load_0[36]) : "memory");
                        }
                        {
                            uint32_t _addr_106 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 20));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_106), "f"(_tmem_load_0[37]) : "memory");
                        }
                        {
                            uint32_t _addr_107 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 24));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_107), "f"(_tmem_load_0[38]) : "memory");
                        }
                        {
                            uint32_t _addr_108 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 28));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_108), "f"(_tmem_load_0[39]) : "memory");
                        }
                        {
                            uint32_t _addr_109 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 32));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_109), "f"(_tmem_load_0[40]) : "memory");
                        }
                        {
                            uint32_t _addr_110 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 36));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_110), "f"(_tmem_load_0[41]) : "memory");
                        }
                        {
                            uint32_t _addr_111 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 40));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_111), "f"(_tmem_load_0[42]) : "memory");
                        }
                        {
                            uint32_t _addr_112 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 44));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_112), "f"(_tmem_load_0[43]) : "memory");
                        }
                        {
                            uint32_t _addr_113 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 48));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_113), "f"(_tmem_load_0[44]) : "memory");
                        }
                        {
                            uint32_t _addr_114 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 52));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_114), "f"(_tmem_load_0[45]) : "memory");
                        }
                        {
                            uint32_t _addr_115 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 56));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_115), "f"(_tmem_load_0[46]) : "memory");
                        }
                        {
                            uint32_t _addr_116 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 60));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_116), "f"(_tmem_load_0[47]) : "memory");
                        }
                        {
                            uint32_t _addr_117 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 64));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_117), "f"(_tmem_load_0[48]) : "memory");
                        }
                        {
                            uint32_t _addr_118 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 68));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_118), "f"(_tmem_load_0[49]) : "memory");
                        }
                        {
                            uint32_t _addr_119 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 72));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_119), "f"(_tmem_load_0[50]) : "memory");
                        }
                        {
                            uint32_t _addr_120 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 76));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_120), "f"(_tmem_load_0[51]) : "memory");
                        }
                        {
                            uint32_t _addr_121 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 80));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_121), "f"(_tmem_load_0[52]) : "memory");
                        }
                        {
                            uint32_t _addr_122 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 84));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_122), "f"(_tmem_load_0[53]) : "memory");
                        }
                        {
                            uint32_t _addr_123 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 88));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_123), "f"(_tmem_load_0[54]) : "memory");
                        }
                        {
                            uint32_t _addr_124 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 92));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_124), "f"(_tmem_load_0[55]) : "memory");
                        }
                        {
                            uint32_t _addr_125 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 96));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_125), "f"(_tmem_load_0[56]) : "memory");
                        }
                        {
                            uint32_t _addr_126 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 100));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_126), "f"(_tmem_load_0[57]) : "memory");
                        }
                        {
                            uint32_t _addr_127 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 104));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_127), "f"(_tmem_load_0[58]) : "memory");
                        }
                        {
                            uint32_t _addr_128 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 108));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_128), "f"(_tmem_load_0[59]) : "memory");
                        }
                        {
                            uint32_t _addr_129 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 112));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_129), "f"(_tmem_load_0[60]) : "memory");
                        }
                        {
                            uint32_t _addr_130 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 116));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_130), "f"(_tmem_load_0[61]) : "memory");
                        }
                        {
                            uint32_t _addr_131 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 120));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_131), "f"(_tmem_load_0[62]) : "memory");
                        }
                        {
                            uint32_t _addr_132 = static_cast<uint32_t>(smem_stg_addr + (stg_b_1 + 124));
                            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_132), "f"(_tmem_load_0[63]) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                int _min_133 = ((32) < (rows_mine) ? (32) : (rows_mine));
                int rows_round = ((0) > (_min_133) ? (0) : (_min_133));
                if (tid == 64) {
                    if (0 != my_rank) {
                        int _max_135 = ((0) > (-my_rank) ? (0) : (-my_rank));
                        int _min_134 = ((1) < (_max_135) ? (1) : (_max_135));
                        int o_idx_c = -_min_134;
                        uint32_t _mbar_peer_136;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_136) : "r"(red_full_addr), "r"(0));
                        uint32_t _cluster_mapa_137;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_137) : "r"(smem_red_addr + slot_b), "r"(0));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_cluster_mapa_137), "r"(smem_stg_addr + (unsigned int)(o_idx_c * 16896)), "r"((uint32_t)(16896)), "r"(_mbar_peer_136)
                            : "memory");
                    }
                    if (1 != my_rank) {
                        int _max_139 = ((0) > (1 - my_rank) ? (0) : (1 - my_rank));
                        int _min_138 = ((1) < (_max_139) ? (1) : (_max_139));
                        int o_idx_c_1 = 1 - _min_138;
                        uint32_t _mbar_peer_140;
                        asm(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mbar_peer_140) : "r"(red_full_addr), "r"(1));
                        uint32_t _cluster_mapa_141;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_cluster_mapa_141) : "r"(smem_red_addr + slot_b), "r"(1));
                        asm volatile(
                            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
                            " [%0], [%1], %2, [%3];"
                            :: "r"(_cluster_mapa_141), "r"(smem_stg_addr + (unsigned int)(o_idx_c_1 * 16896)), "r"((uint32_t)(16896)), "r"(_mbar_peer_140)
                            : "memory");
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    int _min_142 = ((1) < (rows_round) ? (1) : (rows_round));
                    mbarrier_arrive_expect_tx(red_full_addr, _min_142 * 16896);
                }
                mbarrier_wait_cluster_hint(red_full_addr, _phase_peers_free, 10000000);
                if (epi_group == 0) {
                    if (0 == my_rank) {
                        float total_1_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_0 = 0; src_1_0 < 2; src_1_0++) {
                            total_1_0 = total_1_0 + smem_red[src_1_0 * 4224 + lane_row * 33];
                        }
                        int tok_c_2_0 = tok0;
                        if (feature < n_valid && tok_c_2_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_0;
                        }
                        float total_3_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_1 = 0; src_1_1 < 2; src_1_1++) {
                            total_3_0 = total_3_0 + smem_red[src_1_1 * 4224 + lane_row * 33 + 1];
                        }
                        int tok_c_4_0 = tok0 + 1;
                        if (feature < n_valid && tok_c_4_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_0;
                        }
                        float total_5_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_2 = 0; src_1_2 < 2; src_1_2++) {
                            total_5_0 = total_5_0 + smem_red[src_1_2 * 4224 + lane_row * 33 + 2];
                        }
                        int tok_c_6_0 = tok0 + 2;
                        if (feature < n_valid && tok_c_6_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_0;
                        }
                        float total_7_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_3 = 0; src_1_3 < 2; src_1_3++) {
                            total_7_0 = total_7_0 + smem_red[src_1_3 * 4224 + lane_row * 33 + 3];
                        }
                        int tok_c_8_0 = tok0 + 3;
                        if (feature < n_valid && tok_c_8_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_0;
                        }
                        float total_9_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_4 = 0; src_1_4 < 2; src_1_4++) {
                            total_9_0 = total_9_0 + smem_red[src_1_4 * 4224 + lane_row * 33 + 4];
                        }
                        int tok_c_10_0 = tok0 + 4;
                        if (feature < n_valid && tok_c_10_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_0;
                        }
                        float total_11_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_5 = 0; src_1_5 < 2; src_1_5++) {
                            total_11_0 = total_11_0 + smem_red[src_1_5 * 4224 + lane_row * 33 + 5];
                        }
                        int tok_c_12_0 = tok0 + 5;
                        if (feature < n_valid && tok_c_12_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_0;
                        }
                        float total_13_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_6 = 0; src_1_6 < 2; src_1_6++) {
                            total_13_0 = total_13_0 + smem_red[src_1_6 * 4224 + lane_row * 33 + 6];
                        }
                        int tok_c_14_0 = tok0 + 6;
                        if (feature < n_valid && tok_c_14_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_0;
                        }
                        float total_15_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_7 = 0; src_1_7 < 2; src_1_7++) {
                            total_15_0 = total_15_0 + smem_red[src_1_7 * 4224 + lane_row * 33 + 7];
                        }
                        int tok_c_16_0 = tok0 + 7;
                        if (feature < n_valid && tok_c_16_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_16_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_15_0;
                        }
                        float total_17_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_8 = 0; src_1_8 < 2; src_1_8++) {
                            total_17_0 = total_17_0 + smem_red[src_1_8 * 4224 + lane_row * 33 + 8];
                        }
                        int tok_c_18_0 = tok0 + 8;
                        if (feature < n_valid && tok_c_18_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_18_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_17_0;
                        }
                        float total_19_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_9 = 0; src_1_9 < 2; src_1_9++) {
                            total_19_0 = total_19_0 + smem_red[src_1_9 * 4224 + lane_row * 33 + 9];
                        }
                        int tok_c_20_0 = tok0 + 9;
                        if (feature < n_valid && tok_c_20_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_20_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_19_0;
                        }
                        float total_21_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_10 = 0; src_1_10 < 2; src_1_10++) {
                            total_21_0 = total_21_0 + smem_red[src_1_10 * 4224 + lane_row * 33 + 10];
                        }
                        int tok_c_22_0 = tok0 + 10;
                        if (feature < n_valid && tok_c_22_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_22_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_21_0;
                        }
                        float total_23_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_11 = 0; src_1_11 < 2; src_1_11++) {
                            total_23_0 = total_23_0 + smem_red[src_1_11 * 4224 + lane_row * 33 + 11];
                        }
                        int tok_c_24_0 = tok0 + 11;
                        if (feature < n_valid && tok_c_24_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_24_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_23_0;
                        }
                        float total_25_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_12 = 0; src_1_12 < 2; src_1_12++) {
                            total_25_0 = total_25_0 + smem_red[src_1_12 * 4224 + lane_row * 33 + 12];
                        }
                        int tok_c_26_0 = tok0 + 12;
                        if (feature < n_valid && tok_c_26_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_26_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_25_0;
                        }
                        float total_27_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_13 = 0; src_1_13 < 2; src_1_13++) {
                            total_27_0 = total_27_0 + smem_red[src_1_13 * 4224 + lane_row * 33 + 13];
                        }
                        int tok_c_28_0 = tok0 + 13;
                        if (feature < n_valid && tok_c_28_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_28_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_27_0;
                        }
                        float total_29_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_14 = 0; src_1_14 < 2; src_1_14++) {
                            total_29_0 = total_29_0 + smem_red[src_1_14 * 4224 + lane_row * 33 + 14];
                        }
                        int tok_c_30_0 = tok0 + 14;
                        if (feature < n_valid && tok_c_30_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_30_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_29_0;
                        }
                        float total_r_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_15 = 0; src_1_15 < 2; src_1_15++) {
                            total_r_0 = total_r_0 + smem_red[src_1_15 * 4224 + lane_row * 33 + 15];
                        }
                        int tok_c_r_0 = tok0 + 15;
                        if (feature < n_valid && tok_c_r_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_0;
                        }
                        float total_r_r_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_16 = 0; src_1_16 < 2; src_1_16++) {
                            total_r_r_0 = total_r_r_0 + smem_red[src_1_16 * 4224 + lane_row * 33 + 16];
                        }
                        int tok_c_r_r_0 = tok0 + 16;
                        if (feature < n_valid && tok_c_r_r_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_0;
                        }
                        float total_r_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_17 = 0; src_1_17 < 2; src_1_17++) {
                            total_r_1 = total_r_1 + smem_red[src_1_17 * 4224 + lane_row * 33 + 17];
                        }
                        int tok_c_r_1 = tok0 + 17;
                        if (feature < n_valid && tok_c_r_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_1;
                        }
                        float total_r_r_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_18 = 0; src_1_18 < 2; src_1_18++) {
                            total_r_r_1 = total_r_r_1 + smem_red[src_1_18 * 4224 + lane_row * 33 + 18];
                        }
                        int tok_c_r_r_1 = tok0 + 18;
                        if (feature < n_valid && tok_c_r_r_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_1;
                        }
                        float total_r_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_19 = 0; src_1_19 < 2; src_1_19++) {
                            total_r_2 = total_r_2 + smem_red[src_1_19 * 4224 + lane_row * 33 + 19];
                        }
                        int tok_c_r_2 = tok0 + 19;
                        if (feature < n_valid && tok_c_r_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_2;
                        }
                        float total_r_r_2 = 0.0f;
                        #pragma unroll
                        for (int src_1_20 = 0; src_1_20 < 2; src_1_20++) {
                            total_r_r_2 = total_r_r_2 + smem_red[src_1_20 * 4224 + lane_row * 33 + 20];
                        }
                        int tok_c_r_r_2 = tok0 + 20;
                        if (feature < n_valid && tok_c_r_r_2 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_2 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_2;
                        }
                        float total_r_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_21 = 0; src_1_21 < 2; src_1_21++) {
                            total_r_3 = total_r_3 + smem_red[src_1_21 * 4224 + lane_row * 33 + 21];
                        }
                        int tok_c_r_3 = tok0 + 21;
                        if (feature < n_valid && tok_c_r_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_3;
                        }
                        float total_r_r_3 = 0.0f;
                        #pragma unroll
                        for (int src_1_22 = 0; src_1_22 < 2; src_1_22++) {
                            total_r_r_3 = total_r_r_3 + smem_red[src_1_22 * 4224 + lane_row * 33 + 22];
                        }
                        int tok_c_r_r_3 = tok0 + 22;
                        if (feature < n_valid && tok_c_r_r_3 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_3 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_3;
                        }
                        float total_r_4 = 0.0f;
                        #pragma unroll
                        for (int src_1_23 = 0; src_1_23 < 2; src_1_23++) {
                            total_r_4 = total_r_4 + smem_red[src_1_23 * 4224 + lane_row * 33 + 23];
                        }
                        int tok_c_r_4 = tok0 + 23;
                        if (feature < n_valid && tok_c_r_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_4 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_4;
                        }
                        float total_r_r_4 = 0.0f;
                        #pragma unroll
                        for (int src_1_24 = 0; src_1_24 < 2; src_1_24++) {
                            total_r_r_4 = total_r_r_4 + smem_red[src_1_24 * 4224 + lane_row * 33 + 24];
                        }
                        int tok_c_r_r_4 = tok0 + 24;
                        if (feature < n_valid && tok_c_r_r_4 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_4 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_4;
                        }
                        float total_r_5 = 0.0f;
                        #pragma unroll
                        for (int src_1_25 = 0; src_1_25 < 2; src_1_25++) {
                            total_r_5 = total_r_5 + smem_red[src_1_25 * 4224 + lane_row * 33 + 25];
                        }
                        int tok_c_r_5 = tok0 + 25;
                        if (feature < n_valid && tok_c_r_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_5 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_5;
                        }
                        float total_r_r_5 = 0.0f;
                        #pragma unroll
                        for (int src_1_26 = 0; src_1_26 < 2; src_1_26++) {
                            total_r_r_5 = total_r_r_5 + smem_red[src_1_26 * 4224 + lane_row * 33 + 26];
                        }
                        int tok_c_r_r_5 = tok0 + 26;
                        if (feature < n_valid && tok_c_r_r_5 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_5 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_5;
                        }
                        float total_r_6 = 0.0f;
                        #pragma unroll
                        for (int src_1_27 = 0; src_1_27 < 2; src_1_27++) {
                            total_r_6 = total_r_6 + smem_red[src_1_27 * 4224 + lane_row * 33 + 27];
                        }
                        int tok_c_r_6 = tok0 + 27;
                        if (feature < n_valid && tok_c_r_6 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_6 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_6;
                        }
                        float total_r_r_6 = 0.0f;
                        #pragma unroll
                        for (int src_1_28 = 0; src_1_28 < 2; src_1_28++) {
                            total_r_r_6 = total_r_r_6 + smem_red[src_1_28 * 4224 + lane_row * 33 + 28];
                        }
                        int tok_c_r_r_6 = tok0 + 28;
                        if (feature < n_valid && tok_c_r_r_6 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_6 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_6;
                        }
                        float total_r_7 = 0.0f;
                        #pragma unroll
                        for (int src_1_29 = 0; src_1_29 < 2; src_1_29++) {
                            total_r_7 = total_r_7 + smem_red[src_1_29 * 4224 + lane_row * 33 + 29];
                        }
                        int tok_c_r_7 = tok0 + 29;
                        if (feature < n_valid && tok_c_r_7 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_7 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_7;
                        }
                        float total_r_r_7 = 0.0f;
                        #pragma unroll
                        for (int src_1_30 = 0; src_1_30 < 2; src_1_30++) {
                            total_r_r_7 = total_r_r_7 + smem_red[src_1_30 * 4224 + lane_row * 33 + 30];
                        }
                        int tok_c_r_r_7 = tok0 + 30;
                        if (feature < n_valid && tok_c_r_r_7 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_7 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_7;
                        }
                        float total_62_0 = 0.0f;
                        #pragma unroll
                        for (int src_1_31 = 0; src_1_31 < 2; src_1_31++) {
                            total_62_0 = total_62_0 + smem_red[src_1_31 * 4224 + lane_row * 33 + 31];
                        }
                        int tok_c_63_0 = tok0 + 31;
                        if (feature < n_valid && tok_c_63_0 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_63_0 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_62_0;
                        }
                    }
                    if (1 == my_rank) {
                        float total_1_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_32 = 0; src_1_32 < 2; src_1_32++) {
                            total_1_1 = total_1_1 + smem_red[src_1_32 * 4224 + lane_row * 33];
                        }
                        int tok_c_2_1 = tok0 + 32;
                        if (feature < n_valid && tok_c_2_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_2_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_1_1;
                        }
                        float total_3_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_33 = 0; src_1_33 < 2; src_1_33++) {
                            total_3_1 = total_3_1 + smem_red[src_1_33 * 4224 + lane_row * 33 + 1];
                        }
                        int tok_c_4_1 = tok0 + 33;
                        if (feature < n_valid && tok_c_4_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_4_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_3_1;
                        }
                        float total_5_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_34 = 0; src_1_34 < 2; src_1_34++) {
                            total_5_1 = total_5_1 + smem_red[src_1_34 * 4224 + lane_row * 33 + 2];
                        }
                        int tok_c_6_1 = tok0 + 34;
                        if (feature < n_valid && tok_c_6_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_6_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_5_1;
                        }
                        float total_7_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_35 = 0; src_1_35 < 2; src_1_35++) {
                            total_7_1 = total_7_1 + smem_red[src_1_35 * 4224 + lane_row * 33 + 3];
                        }
                        int tok_c_8_1 = tok0 + 35;
                        if (feature < n_valid && tok_c_8_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_8_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_7_1;
                        }
                        float total_9_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_36 = 0; src_1_36 < 2; src_1_36++) {
                            total_9_1 = total_9_1 + smem_red[src_1_36 * 4224 + lane_row * 33 + 4];
                        }
                        int tok_c_10_1 = tok0 + 36;
                        if (feature < n_valid && tok_c_10_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_10_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_9_1;
                        }
                        float total_11_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_37 = 0; src_1_37 < 2; src_1_37++) {
                            total_11_1 = total_11_1 + smem_red[src_1_37 * 4224 + lane_row * 33 + 5];
                        }
                        int tok_c_12_1 = tok0 + 37;
                        if (feature < n_valid && tok_c_12_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_12_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_11_1;
                        }
                        float total_13_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_38 = 0; src_1_38 < 2; src_1_38++) {
                            total_13_1 = total_13_1 + smem_red[src_1_38 * 4224 + lane_row * 33 + 6];
                        }
                        int tok_c_14_1 = tok0 + 38;
                        if (feature < n_valid && tok_c_14_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_14_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_13_1;
                        }
                        float total_15_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_39 = 0; src_1_39 < 2; src_1_39++) {
                            total_15_1 = total_15_1 + smem_red[src_1_39 * 4224 + lane_row * 33 + 7];
                        }
                        int tok_c_16_1 = tok0 + 39;
                        if (feature < n_valid && tok_c_16_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_16_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_15_1;
                        }
                        float total_17_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_40 = 0; src_1_40 < 2; src_1_40++) {
                            total_17_1 = total_17_1 + smem_red[src_1_40 * 4224 + lane_row * 33 + 8];
                        }
                        int tok_c_18_1 = tok0 + 40;
                        if (feature < n_valid && tok_c_18_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_18_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_17_1;
                        }
                        float total_19_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_41 = 0; src_1_41 < 2; src_1_41++) {
                            total_19_1 = total_19_1 + smem_red[src_1_41 * 4224 + lane_row * 33 + 9];
                        }
                        int tok_c_20_1 = tok0 + 41;
                        if (feature < n_valid && tok_c_20_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_20_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_19_1;
                        }
                        float total_21_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_42 = 0; src_1_42 < 2; src_1_42++) {
                            total_21_1 = total_21_1 + smem_red[src_1_42 * 4224 + lane_row * 33 + 10];
                        }
                        int tok_c_22_1 = tok0 + 42;
                        if (feature < n_valid && tok_c_22_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_22_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_21_1;
                        }
                        float total_23_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_43 = 0; src_1_43 < 2; src_1_43++) {
                            total_23_1 = total_23_1 + smem_red[src_1_43 * 4224 + lane_row * 33 + 11];
                        }
                        int tok_c_24_1 = tok0 + 43;
                        if (feature < n_valid && tok_c_24_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_24_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_23_1;
                        }
                        float total_25_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_44 = 0; src_1_44 < 2; src_1_44++) {
                            total_25_1 = total_25_1 + smem_red[src_1_44 * 4224 + lane_row * 33 + 12];
                        }
                        int tok_c_26_1 = tok0 + 44;
                        if (feature < n_valid && tok_c_26_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_26_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_25_1;
                        }
                        float total_27_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_45 = 0; src_1_45 < 2; src_1_45++) {
                            total_27_1 = total_27_1 + smem_red[src_1_45 * 4224 + lane_row * 33 + 13];
                        }
                        int tok_c_28_1 = tok0 + 45;
                        if (feature < n_valid && tok_c_28_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_28_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_27_1;
                        }
                        float total_29_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_46 = 0; src_1_46 < 2; src_1_46++) {
                            total_29_1 = total_29_1 + smem_red[src_1_46 * 4224 + lane_row * 33 + 14];
                        }
                        int tok_c_30_1 = tok0 + 46;
                        if (feature < n_valid && tok_c_30_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_30_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_29_1;
                        }
                        float total_r_8 = 0.0f;
                        #pragma unroll
                        for (int src_1_47 = 0; src_1_47 < 2; src_1_47++) {
                            total_r_8 = total_r_8 + smem_red[src_1_47 * 4224 + lane_row * 33 + 15];
                        }
                        int tok_c_r_8 = tok0 + 47;
                        if (feature < n_valid && tok_c_r_8 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_8 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_8;
                        }
                        float total_r_r_8 = 0.0f;
                        #pragma unroll
                        for (int src_1_48 = 0; src_1_48 < 2; src_1_48++) {
                            total_r_r_8 = total_r_r_8 + smem_red[src_1_48 * 4224 + lane_row * 33 + 16];
                        }
                        int tok_c_r_r_8 = tok0 + 48;
                        if (feature < n_valid && tok_c_r_r_8 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_8 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_8;
                        }
                        float total_r_9 = 0.0f;
                        #pragma unroll
                        for (int src_1_49 = 0; src_1_49 < 2; src_1_49++) {
                            total_r_9 = total_r_9 + smem_red[src_1_49 * 4224 + lane_row * 33 + 17];
                        }
                        int tok_c_r_9 = tok0 + 49;
                        if (feature < n_valid && tok_c_r_9 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_9 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_9;
                        }
                        float total_r_r_9 = 0.0f;
                        #pragma unroll
                        for (int src_1_50 = 0; src_1_50 < 2; src_1_50++) {
                            total_r_r_9 = total_r_r_9 + smem_red[src_1_50 * 4224 + lane_row * 33 + 18];
                        }
                        int tok_c_r_r_9 = tok0 + 50;
                        if (feature < n_valid && tok_c_r_r_9 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_9 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_9;
                        }
                        float total_r_10 = 0.0f;
                        #pragma unroll
                        for (int src_1_51 = 0; src_1_51 < 2; src_1_51++) {
                            total_r_10 = total_r_10 + smem_red[src_1_51 * 4224 + lane_row * 33 + 19];
                        }
                        int tok_c_r_10 = tok0 + 51;
                        if (feature < n_valid && tok_c_r_10 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_10 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_10;
                        }
                        float total_r_r_10 = 0.0f;
                        #pragma unroll
                        for (int src_1_52 = 0; src_1_52 < 2; src_1_52++) {
                            total_r_r_10 = total_r_r_10 + smem_red[src_1_52 * 4224 + lane_row * 33 + 20];
                        }
                        int tok_c_r_r_10 = tok0 + 52;
                        if (feature < n_valid && tok_c_r_r_10 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_10 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_10;
                        }
                        float total_r_11 = 0.0f;
                        #pragma unroll
                        for (int src_1_53 = 0; src_1_53 < 2; src_1_53++) {
                            total_r_11 = total_r_11 + smem_red[src_1_53 * 4224 + lane_row * 33 + 21];
                        }
                        int tok_c_r_11 = tok0 + 53;
                        if (feature < n_valid && tok_c_r_11 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_11 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_11;
                        }
                        float total_r_r_11 = 0.0f;
                        #pragma unroll
                        for (int src_1_54 = 0; src_1_54 < 2; src_1_54++) {
                            total_r_r_11 = total_r_r_11 + smem_red[src_1_54 * 4224 + lane_row * 33 + 22];
                        }
                        int tok_c_r_r_11 = tok0 + 54;
                        if (feature < n_valid && tok_c_r_r_11 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_11 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_11;
                        }
                        float total_r_12 = 0.0f;
                        #pragma unroll
                        for (int src_1_55 = 0; src_1_55 < 2; src_1_55++) {
                            total_r_12 = total_r_12 + smem_red[src_1_55 * 4224 + lane_row * 33 + 23];
                        }
                        int tok_c_r_12 = tok0 + 55;
                        if (feature < n_valid && tok_c_r_12 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_12 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_12;
                        }
                        float total_r_r_12 = 0.0f;
                        #pragma unroll
                        for (int src_1_56 = 0; src_1_56 < 2; src_1_56++) {
                            total_r_r_12 = total_r_r_12 + smem_red[src_1_56 * 4224 + lane_row * 33 + 24];
                        }
                        int tok_c_r_r_12 = tok0 + 56;
                        if (feature < n_valid && tok_c_r_r_12 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_12 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_12;
                        }
                        float total_r_13 = 0.0f;
                        #pragma unroll
                        for (int src_1_57 = 0; src_1_57 < 2; src_1_57++) {
                            total_r_13 = total_r_13 + smem_red[src_1_57 * 4224 + lane_row * 33 + 25];
                        }
                        int tok_c_r_13 = tok0 + 57;
                        if (feature < n_valid && tok_c_r_13 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_13 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_13;
                        }
                        float total_r_r_13 = 0.0f;
                        #pragma unroll
                        for (int src_1_58 = 0; src_1_58 < 2; src_1_58++) {
                            total_r_r_13 = total_r_r_13 + smem_red[src_1_58 * 4224 + lane_row * 33 + 26];
                        }
                        int tok_c_r_r_13 = tok0 + 58;
                        if (feature < n_valid && tok_c_r_r_13 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_13 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_13;
                        }
                        float total_r_14 = 0.0f;
                        #pragma unroll
                        for (int src_1_59 = 0; src_1_59 < 2; src_1_59++) {
                            total_r_14 = total_r_14 + smem_red[src_1_59 * 4224 + lane_row * 33 + 27];
                        }
                        int tok_c_r_14 = tok0 + 59;
                        if (feature < n_valid && tok_c_r_14 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_14 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_14;
                        }
                        float total_r_r_14 = 0.0f;
                        #pragma unroll
                        for (int src_1_60 = 0; src_1_60 < 2; src_1_60++) {
                            total_r_r_14 = total_r_r_14 + smem_red[src_1_60 * 4224 + lane_row * 33 + 28];
                        }
                        int tok_c_r_r_14 = tok0 + 60;
                        if (feature < n_valid && tok_c_r_r_14 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_14 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_14;
                        }
                        float total_r_15 = 0.0f;
                        #pragma unroll
                        for (int src_1_61 = 0; src_1_61 < 2; src_1_61++) {
                            total_r_15 = total_r_15 + smem_red[src_1_61 * 4224 + lane_row * 33 + 29];
                        }
                        int tok_c_r_15 = tok0 + 61;
                        if (feature < n_valid && tok_c_r_15 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_15 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_15;
                        }
                        float total_r_r_15 = 0.0f;
                        #pragma unroll
                        for (int src_1_62 = 0; src_1_62 < 2; src_1_62++) {
                            total_r_r_15 = total_r_r_15 + smem_red[src_1_62 * 4224 + lane_row * 33 + 30];
                        }
                        int tok_c_r_r_15 = tok0 + 62;
                        if (feature < n_valid && tok_c_r_r_15 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_r_r_15 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_r_r_15;
                        }
                        float total_62_1 = 0.0f;
                        #pragma unroll
                        for (int src_1_63 = 0; src_1_63 < 2; src_1_63++) {
                            total_62_1 = total_62_1 + smem_red[src_1_63 * 4224 + lane_row * 33 + 31];
                        }
                        int tok_c_63_1 = tok0 + 63;
                        if (feature < n_valid && tok_c_63_1 < M) {
                            *(reinterpret_cast<__nv_bfloat16*>(out + ((unsigned long long)tok_c_63_1 * (unsigned long long)ldo + col)) + (0)) = (__nv_bfloat16)total_62_1;
                        }
                    }
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (n_items_e > it_e + 1) {
                    if (tid == 64) {
                        #pragma unroll
                        for (int p = 0; p < 2; p++) {
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
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Cleanup
}

} // extern "C"
