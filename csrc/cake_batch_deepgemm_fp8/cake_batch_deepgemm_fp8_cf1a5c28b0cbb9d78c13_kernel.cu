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
#define TMEM_NCOLS 80
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 64
#define TMEM_TMEM_SFB_OFFSET 72
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAINLOOP_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 43008
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 43008
#define SMEM_SMEM_SFA_OFF 41984
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 43008
#define SMEM_SMEM_SFB_OFF 43008
#define SMEM_SMEM_SFB_STAGE_BYTES 1024
#define SMEM_SMEM_SFB_STRIDE 43008
#define SMEM_EPI_STAGING_OFF 173056
#define SMEM_EPI_STAGING_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_STRIDE 16384
#define SMEM_TOTAL 189440
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


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
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


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}



__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


extern "C" {

__global__ __launch_bounds__(224, LAUNCH_MIN_BLOCKS) void
kernel_cake_batch_deepgemm_fp8_cf1a5c28b0cbb9d78c13(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, int* __restrict__ SFA_bits, int* __restrict__ SFB_bits, int* __restrict__ masked_m, const __grid_constant__ CUtensorMap C_tma, unsigned int num_groups, unsigned int shape_m, unsigned int shape_n, unsigned int grid_n, unsigned int k_tiles, unsigned int sf_cols)
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
    #define tma_free_addr (mbar_base + 32)
    #define sf_full_addr (mbar_base + 64)
    #define mainloop_done_addr (mbar_base + 96)
    #define epilogue_done_addr (mbar_base + 112)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int smem_sfa_addr = smem + 41984;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 43008);
    const int smem_sfb_addr = smem + 43008;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int epi_staging_addr = smem + 173056;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C_tma))) : "memory");

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // tma_free: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // sf_full: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // epilogue_done: 2 barriers, init_count=4
            mbarrier_init(smem + 112, 4);
            mbarrier_init(smem + 120, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 80 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 128);
    if (warp == 0) {
        int _tmem_hold = smem + 128;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 64;
    const int tmem_tmem_sfb = taddr + 72;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int epi_panel = 0;
            const int epi_warp = warp;
            unsigned int num_workers = (unsigned int)num_bids;
            unsigned int worker_idx = (unsigned int)bid;
            unsigned int exact_pair_blocks = 0;
            #pragma unroll 1
            for (int count_g = 0; count_g < num_groups; count_g++) {
                unsigned int count_m = (unsigned int)masked_m[count_g];
                unsigned int count_m_blocks = (count_m + 128 - 1) / 128;
                exact_pair_blocks = exact_pair_blocks + (count_m_blocks + 1 - 1);
            }
            unsigned int scheduled_total_tiles = exact_pair_blocks * grid_n;
            unsigned int current_group_idx = 0;
            unsigned int current_pair_cumsum = 0;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = worker_idx; tile_idx < scheduled_total_tiles; tile_idx += num_workers) {
                int has_tile = 0;
                unsigned int selected_group = 0;
                unsigned int selected_pair_start = 0;
                unsigned int selected_m_blocks = 0;
                unsigned int selected_pair_blocks = 1;
                #pragma unroll 1
                for (int scan_g = current_group_idx; scan_g < num_groups; scan_g++) {
                    unsigned int group_m = (unsigned int)masked_m[scan_g];
                    unsigned int m_blocks_scan = (group_m + 128 - 1) / 128;
                    unsigned int pair_blocks_scan = m_blocks_scan + 1 - 1;
                    unsigned int next_pair_cumsum = current_pair_cumsum + pair_blocks_scan;
                    if (has_tile == 0) {
                        if (tile_idx < next_pair_cumsum * grid_n) {
                            current_group_idx = scan_g;
                            selected_group = scan_g;
                            selected_pair_start = current_pair_cumsum;
                            selected_m_blocks = m_blocks_scan;
                            selected_pair_blocks = pair_blocks_scan;
                            has_tile = 1;
                            break;
                        } else {
                            current_pair_cumsum = next_pair_cumsum;
                        }
                    }
                }
                unsigned int zero_u32 = (unsigned int)0;
                unsigned int safe_tile_idx = ((has_tile != 0) ? tile_idx : zero_u32);
                unsigned int in_group = safe_tile_idx - selected_pair_start * grid_n;
                unsigned int pairs_per_l2 = selected_pair_blocks * 16;
                unsigned int l2_group = in_group / pairs_per_l2;
                unsigned int first_n = l2_group * 16;
                unsigned int in_l2 = in_group % pairs_per_l2;
                unsigned int remaining_n = grid_n - first_n;
                unsigned int n_l2_limit = (unsigned int)16;
                unsigned int n_in_l2 = ((remaining_n > n_l2_limit) ? n_l2_limit : remaining_n);
                unsigned int pair_block = in_l2 / n_in_l2;
                unsigned int n_block = first_n + in_l2 % n_in_l2;
                unsigned int raw_m_block = pair_block;
                int store_tile = ((raw_m_block < selected_m_blocks) ? 1 : 0);
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int off_m = raw_m_block * 128;
                unsigned int off_n = n_block * 32;
                int slab_row0 = epi_warp * 64;
                #pragma unroll
                for (int panel = 0; panel < 1; panel++) {
                    unsigned int panel_col0 = off_n + (unsigned int)(panel * 32);
                    int staging_row0 = slab_row0 + (int)(epi_panel % 2) * 32;
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncwarp();
                    #pragma unroll
                    for (int n_chunk = 0; n_chunk < 4; n_chunk++) {
                        int row = epi_warp * 32;
                        int col = (int)epi_stage * 32 + panel * 32 + n_chunk * 8;
                        int tmem_addr = taddr + (unsigned int)(row << 16) + (unsigned int)col;
                        float _tmem_load_0[8];
                        tmem_ld_x8(&_tmem_load_0[0], tmem_addr);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        uint32_t _tmem_load_0_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int staging_row = staging_row0 + lane;
                        int swz_col = n_chunk * 8 ^ (staging_row >> 1 & 3) << 3;
                        __nv_bfloat16* _sv_ptr_0 = reinterpret_cast<__nv_bfloat16*>(epi_staging + (staging_row * 32 + swz_col));
                        reinterpret_cast<int4*>(_sv_ptr_0 + 0)[0] = reinterpret_cast<int4*>(_tmem_load_0_bf16)[0];
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        if (store_tile != 0) {
                            if (panel_col0 < shape_n) {
                                tma_store_3d((&C_tma), panel_col0, off_m + (unsigned int)(epi_warp * 32), selected_group, epi_staging_addr + (unsigned int)(staging_row0 * 64));
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    __syncwarp();
                    epi_panel = epi_panel + 1;
                }
                if (elect_sync()) {
                    mbarrier_arrive(epilogue_done_addr + (epi_stage) * 8);
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
            }
            if (elect_sync()) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int tma_stage = 0;
            unsigned int epi_stage_1 = 0;
            unsigned int num_workers_1 = (unsigned int)num_bids;
            unsigned int worker_idx_1 = (unsigned int)bid;
            unsigned int exact_pair_blocks_1 = 0;
            #pragma unroll 1
            for (int count_g_1 = 0; count_g_1 < num_groups; count_g_1++) {
                unsigned int count_m_1 = (unsigned int)masked_m[count_g_1];
                unsigned int count_m_blocks_1 = (count_m_1 + 128 - 1) / 128;
                exact_pair_blocks_1 = exact_pair_blocks_1 + (count_m_blocks_1 + 1 - 1);
            }
            unsigned int scheduled_total_tiles_1 = exact_pair_blocks_1 * grid_n;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_sf_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_idx = worker_idx_1; _tile_idx < scheduled_total_tiles_1; _tile_idx += num_workers_1) {
                mbarrier_wait(epilogue_done_addr + (epi_stage_1) * 8, _phase_epilogue_done);
                #pragma unroll 1
                for (int iter_k = 0; iter_k < k_tiles; iter_k++) {
                    mbarrier_wait(tma_full_addr + (tma_stage) * 8, _phase_tma_full);
                    mbarrier_wait(sf_full_addr + (tma_stage) * 8, _phase_sf_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((iter_k == 0) ? 1 : 0);
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (tma_stage) * 2688)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (tma_stage) * 2688 + 32)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (tma_stage) * 2688)));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (tma_stage) * 2688 + 32)));
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (tma_stage) * 2688;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (tma_stage) * 2688;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 32)), a_desc + 0, b_desc + 0,
                                0x8880000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 32)), a_desc + 2, b_desc + 2,
                                0x28880010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 32)), a_desc + 4, b_desc + 4,
                                0x48880020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 32)), a_desc + 6, b_desc + 6,
                                0x68880030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                        }
                        int _mma_a_lo_1 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (tma_stage) * 2688;
                        int _mma_b_lo_1 = (((smem_b_addr + 4096) >> 4) & 0x3FFF) + (tma_stage) * 2688;
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 32)), a_desc + 0, b_desc + 0,
                                0x8880000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 32)), a_desc + 2, b_desc + 2,
                                0x28880010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 32)), a_desc + 4, b_desc + 4,
                                0x48880020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 32)), a_desc + 6, b_desc + 6,
                                0x68880030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                        }
                    }
                    elect_commit(tma_free_addr + (tma_stage) * 8);
                    tma_stage += 1;
                    if (tma_stage == 4) { tma_stage = 0; _phase_tma_full ^= 1; _phase_sf_full ^= 1; }
                }
                elect_commit(mainloop_done_addr + (epi_stage_1) * 8);
                epi_stage_1 += 1;
                if (epi_stage_1 == 2) { epi_stage_1 = 0; _phase_epilogue_done ^= 1; }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 5) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int num_workers_2 = (unsigned int)num_bids;
            unsigned int worker_idx_2 = (unsigned int)bid;
            unsigned int max_m_blocks = (shape_m + 128 - 1) / 128;
            unsigned int exact_pair_blocks_2 = 0;
            #pragma unroll 1
            for (int count_g_2 = 0; count_g_2 < num_groups; count_g_2++) {
                unsigned int count_m_2 = (unsigned int)masked_m[count_g_2];
                unsigned int count_m_blocks_2 = (count_m_2 + 128 - 1) / 128;
                exact_pair_blocks_2 = exact_pair_blocks_2 + (count_m_blocks_2 + 1 - 1);
            }
            unsigned int scheduled_total_tiles_2 = exact_pair_blocks_2 * grid_n;
            unsigned int current_group_idx_1 = 0;
            unsigned int current_pair_cumsum_1 = 0;
            unsigned int _phase_tma_free = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = worker_idx_2; tile_idx_1 < scheduled_total_tiles_2; tile_idx_1 += num_workers_2) {
                int has_tile_1 = 0;
                unsigned int selected_group_1 = 0;
                unsigned int selected_pair_start_1 = 0;
                unsigned int selected_m_blocks_1 = 0;
                unsigned int selected_pair_blocks_1 = 1;
                #pragma unroll 1
                for (int scan_g_1 = current_group_idx_1; scan_g_1 < num_groups; scan_g_1++) {
                    unsigned int group_m_1 = (unsigned int)masked_m[scan_g_1];
                    unsigned int m_blocks_scan_1 = (group_m_1 + 128 - 1) / 128;
                    unsigned int pair_blocks_scan_1 = m_blocks_scan_1 + 1 - 1;
                    unsigned int next_pair_cumsum_1 = current_pair_cumsum_1 + pair_blocks_scan_1;
                    if (has_tile_1 == 0) {
                        if (tile_idx_1 < next_pair_cumsum_1 * grid_n) {
                            current_group_idx_1 = scan_g_1;
                            selected_group_1 = scan_g_1;
                            selected_pair_start_1 = current_pair_cumsum_1;
                            selected_m_blocks_1 = m_blocks_scan_1;
                            selected_pair_blocks_1 = pair_blocks_scan_1;
                            has_tile_1 = 1;
                            break;
                        } else {
                            current_pair_cumsum_1 = next_pair_cumsum_1;
                        }
                    }
                }
                unsigned int zero_u32_1 = (unsigned int)0;
                unsigned int safe_tile_idx_1 = ((has_tile_1 != 0) ? tile_idx_1 : zero_u32_1);
                unsigned int in_group_1 = safe_tile_idx_1 - selected_pair_start_1 * grid_n;
                unsigned int pairs_per_l2_1 = selected_pair_blocks_1 * 16;
                unsigned int l2_group_1 = in_group_1 / pairs_per_l2_1;
                unsigned int first_n_1 = l2_group_1 * 16;
                unsigned int in_l2_1 = in_group_1 % pairs_per_l2_1;
                unsigned int remaining_n_1 = grid_n - first_n_1;
                unsigned int n_l2_limit_1 = (unsigned int)16;
                unsigned int n_in_l2_1 = ((remaining_n_1 > n_l2_limit_1) ? n_l2_limit_1 : remaining_n_1);
                unsigned int pair_block_1 = in_l2_1 / n_in_l2_1;
                unsigned int n_block_1 = first_n_1 + in_l2_1 % n_in_l2_1;
                unsigned int raw_m_block_1 = pair_block_1;
                unsigned int m_block = ((raw_m_block_1 < max_m_blocks) ? raw_m_block_1 : pair_block_1);
                unsigned int off_m_1 = m_block * 128;
                unsigned int off_n_1 = n_block_1 * 32;
                unsigned int flat_group_m = selected_group_1 * shape_m + off_m_1;
                unsigned int sfb_group_base = selected_group_1 * (shape_n / 128);
                unsigned int last_sfb_block = shape_n / 128 - 1;
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < k_tiles; iter_k_1++) {
                    mbarrier_wait(tma_free_addr + (load_stage) * 8, _phase_tma_free);
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_a_addr + load_stage * 43008, (&A), 0, off_m_1, (unsigned int)iter_k_1 * 2, selected_group_1, tma_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(smem_b_addr + load_stage * 43008, (&B), 0, off_n_1, (unsigned int)iter_k_1 * 2, selected_group_1, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 40960);
                    }
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; _phase_tma_free ^= 1; }
                }
            }
        }
    }
    // ---- Role: scales ----
    if (warp == 6) {
        { // scales_main
            unsigned int sf_stage = 0;
            unsigned int num_workers_3 = (unsigned int)num_bids;
            unsigned int worker_idx_3 = (unsigned int)bid;
            unsigned int max_m_blocks_1 = (shape_m + 128 - 1) / 128;
            unsigned int exact_pair_blocks_3 = 0;
            #pragma unroll 1
            for (int count_g_3 = 0; count_g_3 < num_groups; count_g_3++) {
                unsigned int count_m_3 = (unsigned int)masked_m[count_g_3];
                unsigned int count_m_blocks_3 = (count_m_3 + 128 - 1) / 128;
                exact_pair_blocks_3 = exact_pair_blocks_3 + (count_m_blocks_3 + 1 - 1);
            }
            unsigned int scheduled_total_tiles_3 = exact_pair_blocks_3 * grid_n;
            unsigned int current_group_idx_2 = 0;
            unsigned int current_pair_cumsum_2 = 0;
            unsigned int _phase_tma_free_1 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = worker_idx_3; tile_idx_2 < scheduled_total_tiles_3; tile_idx_2 += num_workers_3) {
                int has_tile_2 = 0;
                unsigned int selected_group_2 = 0;
                unsigned int selected_pair_start_2 = 0;
                unsigned int selected_m_blocks_2 = 0;
                unsigned int selected_pair_blocks_2 = 1;
                #pragma unroll 1
                for (int scan_g_2 = current_group_idx_2; scan_g_2 < num_groups; scan_g_2++) {
                    unsigned int group_m_2 = (unsigned int)masked_m[scan_g_2];
                    unsigned int m_blocks_scan_2 = (group_m_2 + 128 - 1) / 128;
                    unsigned int pair_blocks_scan_2 = m_blocks_scan_2 + 1 - 1;
                    unsigned int next_pair_cumsum_2 = current_pair_cumsum_2 + pair_blocks_scan_2;
                    if (has_tile_2 == 0) {
                        if (tile_idx_2 < next_pair_cumsum_2 * grid_n) {
                            current_group_idx_2 = scan_g_2;
                            selected_group_2 = scan_g_2;
                            selected_pair_start_2 = current_pair_cumsum_2;
                            selected_m_blocks_2 = m_blocks_scan_2;
                            selected_pair_blocks_2 = pair_blocks_scan_2;
                            has_tile_2 = 1;
                            break;
                        } else {
                            current_pair_cumsum_2 = next_pair_cumsum_2;
                        }
                    }
                }
                unsigned int zero_u32_2 = (unsigned int)0;
                unsigned int safe_tile_idx_2 = ((has_tile_2 != 0) ? tile_idx_2 : zero_u32_2);
                unsigned int in_group_2 = safe_tile_idx_2 - selected_pair_start_2 * grid_n;
                unsigned int pairs_per_l2_2 = selected_pair_blocks_2 * 16;
                unsigned int l2_group_2 = in_group_2 / pairs_per_l2_2;
                unsigned int first_n_2 = l2_group_2 * 16;
                unsigned int in_l2_2 = in_group_2 % pairs_per_l2_2;
                unsigned int remaining_n_2 = grid_n - first_n_2;
                unsigned int n_l2_limit_2 = (unsigned int)16;
                unsigned int n_in_l2_2 = ((remaining_n_2 > n_l2_limit_2) ? n_l2_limit_2 : remaining_n_2);
                unsigned int pair_block_2 = in_l2_2 / n_in_l2_2;
                unsigned int n_block_2 = first_n_2 + in_l2_2 % n_in_l2_2;
                unsigned int raw_m_block_2 = pair_block_2;
                unsigned int m_block_1 = ((raw_m_block_2 < max_m_blocks_1) ? raw_m_block_2 : pair_block_2);
                unsigned int off_m_2 = m_block_1 * 128;
                unsigned int off_n_2 = n_block_2 * 32;
                unsigned int flat_group_m_1 = selected_group_2 * shape_m + off_m_2;
                unsigned int sfb_group_base_1 = selected_group_2 * (shape_n / 128);
                unsigned int last_sfb_block_1 = shape_n / 128 - 1;
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < k_tiles; iter_k_2++) {
                    mbarrier_wait(tma_free_addr + (sf_stage) * 8, _phase_tma_free_1);
                    int sfa_base = smem_sfa_addr + sf_stage * 43008;
                    int sfb_base = smem_sfb_addr + sf_stage * 43008;
                    unsigned int sf_lane = (unsigned int)lane;
                    unsigned int sf_row = (unsigned int)0 + sf_lane;
                    unsigned int sfa_lane_dst = sf_lane * 16;
                    unsigned int sfa_idx0 = (flat_group_m_1 + sf_row) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfa_v = (unsigned int)(SFA_bits[sfa_idx0] >> 23);
                    unsigned int sfa_word = sfa_v | sfa_v << 8 | sfa_v << 16 | sfa_v << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sfa_lane_dst), "r"(sfa_word));
                    unsigned int sfa_v_0 = (unsigned int)(SFA_bits[sfa_idx0 + 1] >> 23);
                    unsigned int sfa_word_1 = sfa_v_0 | sfa_v_0 << 8 | sfa_v_0 << 16 | sfa_v_0 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfa_base + 512) + sfa_lane_dst), "r"(sfa_word_1));
                    unsigned int sf_row_2 = (unsigned int)32 + sf_lane;
                    unsigned int sfa_lane_dst_3 = sf_lane * 16 + 4;
                    unsigned int sfa_idx0_4 = (flat_group_m_1 + sf_row_2) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfa_v_5 = (unsigned int)(SFA_bits[sfa_idx0_4] >> 23);
                    unsigned int sfa_word_6 = sfa_v_5 | sfa_v_5 << 8 | sfa_v_5 << 16 | sfa_v_5 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sfa_lane_dst_3), "r"(sfa_word_6));
                    unsigned int sfa_v_7 = (unsigned int)(SFA_bits[sfa_idx0_4 + 1] >> 23);
                    unsigned int sfa_word_8 = sfa_v_7 | sfa_v_7 << 8 | sfa_v_7 << 16 | sfa_v_7 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfa_base + 512) + sfa_lane_dst_3), "r"(sfa_word_8));
                    unsigned int sf_row_9 = (unsigned int)64 + sf_lane;
                    unsigned int sfa_lane_dst_10 = sf_lane * 16 + 8;
                    unsigned int sfa_idx0_11 = (flat_group_m_1 + sf_row_9) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfa_v_12 = (unsigned int)(SFA_bits[sfa_idx0_11] >> 23);
                    unsigned int sfa_word_13 = sfa_v_12 | sfa_v_12 << 8 | sfa_v_12 << 16 | sfa_v_12 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sfa_lane_dst_10), "r"(sfa_word_13));
                    unsigned int sfa_v_14 = (unsigned int)(SFA_bits[sfa_idx0_11 + 1] >> 23);
                    unsigned int sfa_word_15 = sfa_v_14 | sfa_v_14 << 8 | sfa_v_14 << 16 | sfa_v_14 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfa_base + 512) + sfa_lane_dst_10), "r"(sfa_word_15));
                    unsigned int sf_row_16 = (unsigned int)96 + sf_lane;
                    unsigned int sfa_lane_dst_17 = sf_lane * 16 + 12;
                    unsigned int sfa_idx0_18 = (flat_group_m_1 + sf_row_16) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfa_v_19 = (unsigned int)(SFA_bits[sfa_idx0_18] >> 23);
                    unsigned int sfa_word_20 = sfa_v_19 | sfa_v_19 << 8 | sfa_v_19 << 16 | sfa_v_19 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sfa_lane_dst_17), "r"(sfa_word_20));
                    unsigned int sfa_v_21 = (unsigned int)(SFA_bits[sfa_idx0_18 + 1] >> 23);
                    unsigned int sfa_word_22 = sfa_v_21 | sfa_v_21 << 8 | sfa_v_21 << 16 | sfa_v_21 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfa_base + 512) + sfa_lane_dst_17), "r"(sfa_word_22));
                    unsigned int sfb_block_idx = 0;
                    unsigned int n_glob = off_n_2 + (unsigned int)0 + sf_lane;
                    unsigned int n_sf_block = n_glob / 128;
                    unsigned int n_sf_block_safe = ((n_glob < shape_n) ? n_sf_block : last_sfb_block_1);
                    unsigned int sfb_idx0 = (sfb_group_base_1 + n_sf_block_safe) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfb_lane_dst = sfb_block_idx * 512 + sf_lane * 16;
                    unsigned int sfb_v = (unsigned int)(SFB_bits[sfb_idx0] >> 23);
                    unsigned int sfb_word = sfb_v | sfb_v << 8 | sfb_v << 16 | sfb_v << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst), "r"(sfb_word));
                    unsigned int sfb_v_23 = (unsigned int)(SFB_bits[sfb_idx0 + 1] >> 23);
                    unsigned int sfb_word_24 = sfb_v_23 | sfb_v_23 << 8 | sfb_v_23 << 16 | sfb_v_23 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfb_base + 512) + sfb_lane_dst), "r"(sfb_word_24));
                    unsigned int sfb_block_idx_25 = 0;
                    unsigned int n_glob_26 = off_n_2 + (unsigned int)32 + sf_lane;
                    unsigned int n_sf_block_27 = n_glob_26 / 128;
                    unsigned int n_sf_block_safe_28 = ((n_glob_26 < shape_n) ? n_sf_block_27 : last_sfb_block_1);
                    unsigned int sfb_idx0_29 = (sfb_group_base_1 + n_sf_block_safe_28) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfb_lane_dst_30 = sfb_block_idx_25 * 512 + sf_lane * 16 + 4;
                    unsigned int sfb_v_31 = (unsigned int)(SFB_bits[sfb_idx0_29] >> 23);
                    unsigned int sfb_word_32 = sfb_v_31 | sfb_v_31 << 8 | sfb_v_31 << 16 | sfb_v_31 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_30), "r"(sfb_word_32));
                    unsigned int sfb_v_33 = (unsigned int)(SFB_bits[sfb_idx0_29 + 1] >> 23);
                    unsigned int sfb_word_34 = sfb_v_33 | sfb_v_33 << 8 | sfb_v_33 << 16 | sfb_v_33 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfb_base + 512) + sfb_lane_dst_30), "r"(sfb_word_34));
                    unsigned int sfb_block_idx_35 = 0;
                    unsigned int n_glob_36 = off_n_2 + (unsigned int)64 + sf_lane;
                    unsigned int n_sf_block_37 = n_glob_36 / 128;
                    unsigned int n_sf_block_safe_38 = ((n_glob_36 < shape_n) ? n_sf_block_37 : last_sfb_block_1);
                    unsigned int sfb_idx0_39 = (sfb_group_base_1 + n_sf_block_safe_38) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfb_lane_dst_40 = sfb_block_idx_35 * 512 + sf_lane * 16 + 8;
                    unsigned int sfb_v_41 = (unsigned int)(SFB_bits[sfb_idx0_39] >> 23);
                    unsigned int sfb_word_42 = sfb_v_41 | sfb_v_41 << 8 | sfb_v_41 << 16 | sfb_v_41 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_40), "r"(sfb_word_42));
                    unsigned int sfb_v_43 = (unsigned int)(SFB_bits[sfb_idx0_39 + 1] >> 23);
                    unsigned int sfb_word_44 = sfb_v_43 | sfb_v_43 << 8 | sfb_v_43 << 16 | sfb_v_43 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfb_base + 512) + sfb_lane_dst_40), "r"(sfb_word_44));
                    unsigned int sfb_block_idx_45 = 0;
                    unsigned int n_glob_46 = off_n_2 + (unsigned int)96 + sf_lane;
                    unsigned int n_sf_block_47 = n_glob_46 / 128;
                    unsigned int n_sf_block_safe_48 = ((n_glob_46 < shape_n) ? n_sf_block_47 : last_sfb_block_1);
                    unsigned int sfb_idx0_49 = (sfb_group_base_1 + n_sf_block_safe_48) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfb_lane_dst_50 = sfb_block_idx_45 * 512 + sf_lane * 16 + 12;
                    unsigned int sfb_v_51 = (unsigned int)(SFB_bits[sfb_idx0_49] >> 23);
                    unsigned int sfb_word_52 = sfb_v_51 | sfb_v_51 << 8 | sfb_v_51 << 16 | sfb_v_51 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_50), "r"(sfb_word_52));
                    unsigned int sfb_v_53 = (unsigned int)(SFB_bits[sfb_idx0_49 + 1] >> 23);
                    unsigned int sfb_word_54 = sfb_v_53 | sfb_v_53 << 8 | sfb_v_53 << 16 | sfb_v_53 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfb_base + 512) + sfb_lane_dst_50), "r"(sfb_word_54));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(sf_full_addr + (sf_stage) * 8);
                    }
                    sf_stage += 1;
                    if (sf_stage == 4) { sf_stage = 0; _phase_tma_free_1 ^= 1; }
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
