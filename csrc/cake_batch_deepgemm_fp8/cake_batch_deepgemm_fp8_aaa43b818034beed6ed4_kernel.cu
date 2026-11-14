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
    return (x & 0x7FFFFULL) >> 4ULL;
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
kernel_cake_batch_deepgemm_fp8_aaa43b818034beed6ed4(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, int* __restrict__ SFA_bits, int* __restrict__ SFB_bits, int* __restrict__ masked_m, const __grid_constant__ CUtensorMap C_tma, unsigned int num_groups, unsigned int shape_m, unsigned int shape_n, unsigned int grid_n, unsigned int k_tiles, unsigned int sf_cols)
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
    #define tma_free_addr (mbar_base + 24)
    #define sf_full_addr (mbar_base + 48)
    #define mainloop_done_addr (mbar_base + 72)
    #define epilogue_done_addr (mbar_base + 88)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 70656);
    const int smem_sfa_addr = smem + 70656;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 71680);
    const int smem_sfb_addr = smem + 71680;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 219136);
    const int epi_staging_addr = smem + 219136;

    // Kernel setup ops
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C_tma))) : "memory");
    unsigned int mms_line = (unsigned int)lane;
    if (warp == 0 && mms_line * 32 < num_groups) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(masked_m + mms_line * 32))); }

    // Mbarrier init (13 barriers at smem_raw[0..104))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // tma_free: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // sf_full: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // epilogue_done: 2 barriers, init_count=4
            mbarrier_init(smem + 88, 4);
            mbarrier_init(smem + 96, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 312 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 104);
    if (warp == 0) {
        int _tmem_hold = smem + 104;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_accum = taddr;
    const unsigned int tmem_tmem_sfa = taddr + 288;
    const unsigned int tmem_tmem_sfb = taddr + 296;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int epi_panel = 0;
            const int epi_warp = warp;
            unsigned int num_workers = (unsigned int)num_bids;
            unsigned int worker_idx = (unsigned int)bid;
            unsigned int mms_g0 = 0;
            unsigned int mms_base = 0;
            unsigned int mms_mb = 0;
            unsigned int mms_incl = 0;
            unsigned int mms_m = 0;
            if (mms_g0 + (unsigned int)lane < num_groups) {
                mms_m = (unsigned int)masked_m[(int)mms_g0 + lane];
            }
            mms_mb = (mms_m + 128 - 1) / 128;
            uint32_t _warp_scan_sum_0 = mms_mb;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0, 1, 32); if (lane >= 1) { _warp_scan_sum_0 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0, 2, 32); if (lane >= 2) { _warp_scan_sum_0 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0, 4, 32); if (lane >= 4) { _warp_scan_sum_0 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0, 8, 32); if (lane >= 8) { _warp_scan_sum_0 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0, 16, 32); if (lane >= 16) { _warp_scan_sum_0 += peer; } }
            mms_incl = _warp_scan_sum_0;
            unsigned int mms_blocks = 0;
            #pragma unroll 32
            for (int mms_g = 0; mms_g < num_groups; mms_g++) {
                mms_blocks = mms_blocks + ((unsigned int)masked_m[mms_g] + 128 - 1) / 128;
            }
            unsigned int mms_tiles = mms_blocks * grid_n;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = worker_idx; tile_idx < mms_tiles; tile_idx += num_workers) {
                unsigned int mms_hit = __ballot_sync(0xFFFFFFFF, tile_idx < (mms_base + mms_incl) * grid_n);
                #pragma unroll 1
                for (unsigned int mms_c = mms_g0 + 32; mms_c < num_groups; mms_c += 32) {
                    if (mms_hit != 0) {
                        break;
                    }
                    unsigned int _shfl_sync_1 = __shfl_sync(0xFFFFFFFF, mms_incl, 31);
                    mms_base = mms_base + _shfl_sync_1;
                    mms_g0 = mms_c;
                    unsigned int mms_m_2 = 0;
                    if (mms_g0 + (unsigned int)lane < num_groups) {
                        mms_m_2 = (unsigned int)masked_m[(int)mms_g0 + lane];
                    }
                    mms_mb = (mms_m_2 + 128 - 1) / 128;
                    uint32_t _warp_scan_sum_2 = mms_mb;
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2, 1, 32); if (lane >= 1) { _warp_scan_sum_2 += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2, 2, 32); if (lane >= 2) { _warp_scan_sum_2 += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2, 4, 32); if (lane >= 4) { _warp_scan_sum_2 += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2, 8, 32); if (lane >= 8) { _warp_scan_sum_2 += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2, 16, 32); if (lane >= 16) { _warp_scan_sum_2 += peer; } }
                    mms_incl = _warp_scan_sum_2;
                    unsigned int _warp_vote_3 = __ballot_sync(0xFFFFFFFF, tile_idx < (mms_base + mms_incl) * grid_n);
                    mms_hit = _warp_vote_3;
                }
                int _find_first_set_4 = __ffs(mms_hit);
                int mms_lane = _find_first_set_4 - 1;
                unsigned int mms_group = mms_g0 + (unsigned int)mms_lane;
                unsigned int mms_blocks_2 = __shfl_sync(0xFFFFFFFF, mms_mb, mms_lane);
                unsigned int _shfl_sync_5 = __shfl_sync(0xFFFFFFFF, mms_incl, mms_lane);
                unsigned int mms_start = mms_base + _shfl_sync_5 - mms_blocks_2;
                unsigned int mms_in_group = tile_idx - mms_start * grid_n;
                unsigned int mms_per_l2 = mms_blocks_2 * 16;
                unsigned int mms_first_n = mms_in_group / mms_per_l2 * 16;
                unsigned int mms_in_l2 = mms_in_group % mms_per_l2;
                unsigned int mms_left_n = grid_n - mms_first_n;
                unsigned int mms_n_in_l2 = ((mms_left_n > (unsigned int)16) ? (unsigned int)16 : mms_left_n);
                unsigned int mms_pair_block = mms_in_l2 / mms_n_in_l2;
                unsigned int mms_n_block = mms_first_n + mms_in_l2 % mms_n_in_l2;
                unsigned int raw_m_block = mms_pair_block;
                int store_tile = ((raw_m_block < mms_blocks_2) ? 1 : 0);
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                unsigned int off_m = raw_m_block * 128;
                unsigned int off_n = mms_n_block * 144;
                int slab_row0 = epi_warp * 64;
                #pragma unroll
                for (int panel = 0; panel < 9; panel++) {
                    unsigned int panel_col0 = off_n + (unsigned int)(panel * 16);
                    int staging_row0 = slab_row0 + (int)(epi_panel % 2) * 32;
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    __syncwarp();
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int n_chunk = 0; n_chunk < 2; n_chunk++) {
                        int row = epi_warp * 32;
                        int col = (int)epi_stage * 144 + panel * 16 + n_chunk * 8;
                        int tmem_addr = taddr + (unsigned int)(row << 16) + (unsigned int)col;
                        float _tmem_load_0[8];
                        tmem_ld_x8(&_tmem_load_0[0], tmem_addr);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        unsigned int _tmem_load_0_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int staging_row = staging_row0 + lane;
                        int swz_col = n_chunk * 8 ^ (staging_row >> 2 & 1) << 3;
                        __nv_bfloat16* _sv_ptr = reinterpret_cast<__nv_bfloat16*>(epi_staging + (staging_row * 16 + swz_col));
                        reinterpret_cast<int4*>(_sv_ptr + 0)[0] = reinterpret_cast<int4*>(_tmem_load_0_bf16)[0];
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        if (store_tile != 0) {
                            if (panel_col0 < shape_n) {
                                tma_store_3d((&C_tma), panel_col0, off_m + (unsigned int)(epi_warp * 32), mms_group, epi_staging_addr + (unsigned int)(staging_row0 * 32));
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
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            unsigned int tma_stage = 0;
            unsigned int epi_stage_1 = 0;
            unsigned int num_workers_1 = (unsigned int)num_bids;
            unsigned int worker_idx_1 = (unsigned int)bid;
            unsigned int mms_g0_1 = 0;
            unsigned int mms_base_1 = 0;
            unsigned int mms_mb_1 = 0;
            unsigned int mms_incl_1 = 0;
            unsigned int mms_m_1 = 0;
            if (mms_g0_1 + (unsigned int)lane < num_groups) {
                mms_m_1 = (unsigned int)masked_m[(int)mms_g0_1 + lane];
            }
            mms_mb_1 = (mms_m_1 + 128 - 1) / 128;
            uint32_t _warp_scan_sum_0_1 = mms_mb_1;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_1, 1, 32); if (lane >= 1) { _warp_scan_sum_0_1 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_1, 2, 32); if (lane >= 2) { _warp_scan_sum_0_1 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_1, 4, 32); if (lane >= 4) { _warp_scan_sum_0_1 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_1, 8, 32); if (lane >= 8) { _warp_scan_sum_0_1 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_1, 16, 32); if (lane >= 16) { _warp_scan_sum_0_1 += peer; } }
            mms_incl_1 = _warp_scan_sum_0_1;
            unsigned int mms_blocks_1 = 0;
            #pragma unroll 32
            for (int mms_g_1 = 0; mms_g_1 < num_groups; mms_g_1++) {
                mms_blocks_1 = mms_blocks_1 + ((unsigned int)masked_m[mms_g_1] + 128 - 1) / 128;
            }
            unsigned int mms_tiles_1 = mms_blocks_1 * grid_n;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_idx = worker_idx_1; _tile_idx < mms_tiles_1; _tile_idx += num_workers_1) {
                mbarrier_wait(epilogue_done_addr + (epi_stage_1) * 8, _phase_epilogue_done);
                #pragma unroll 1
                for (int iter_k = 0; iter_k < k_tiles; iter_k++) {
                    mbarrier_wait(tma_full_addr + (tma_stage) * 8, _phase_tma_full);
                    mbarrier_wait(sf_full_addr + (tma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((iter_k == 0) ? 1 : 0);
                    uint32_t _elected_1 = elect_sync();
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (tma_stage) * 4544)), _elected_1);
                    tcgen05_cp_32x128b_warpx4_leader((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (tma_stage) * 4544 + 32)), _elected_1);
                    tcgen05_cp_32x128b_warpx4_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (tma_stage) * 4544)), _elected_1);
                    tcgen05_cp_32x128b_warpx4_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (tma_stage) * 4544 + 32)), _elected_1);
                    tcgen05_cp_32x128b_warpx4_leader((tmem_tmem_sfb + 8), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (tma_stage) * 4544 + 64)), _elected_1);
                    tcgen05_cp_32x128b_warpx4_leader((tmem_tmem_sfb + 12), make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (tma_stage) * 4544 + 96)), _elected_1);
                    int _smem_desc_lo_2 = (((smem_a_addr) >> 4) & 0x7FFF) + (tma_stage) * 4544;
                    int _smem_desc_lo_3 = (((smem_b_addr) >> 4) & 0x7FFF) + (tma_stage) * 4544;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (epi_stage_1 * 144)), a_desc + 0, b_desc + 0,
                            0x8a40000U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1), _elected_1);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (epi_stage_1 * 144)), a_desc + 2, b_desc + 2,
                            0x28a40010U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 1, _elected_1);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (epi_stage_1 * 144)), a_desc + 4, b_desc + 4,
                            0x48a40020U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 1, _elected_1);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (epi_stage_1 * 144)), a_desc + 6, b_desc + 6,
                            0x68a40030U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 1, _elected_1);
                    }
                    int _smem_desc_lo_4 = (((smem_a_addr + 16384) >> 4) & 0x7FFF) + (tma_stage) * 4544;
                    int _smem_desc_lo_5 = (((smem_b_addr + 18432) >> 4) & 0x7FFF) + (tma_stage) * 4544;
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (epi_stage_1 * 144)), a_desc + 0, b_desc + 0,
                            0x8a40000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_1);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (epi_stage_1 * 144)), a_desc + 2, b_desc + 2,
                            0x28a40010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_1);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (epi_stage_1 * 144)), a_desc + 4, b_desc + 4,
                            0x48a40020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_1);
                        tcgen05_mma_mxf8_bs_leader((tmem_accum + (epi_stage_1 * 144)), a_desc + 6, b_desc + 6,
                            0x68a40030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 8, 1, _elected_1);
                    }
                    elect_commit(tma_free_addr + (tma_stage) * 8);
                    tma_stage += 1;
                    if (tma_stage == 3) { tma_stage = 0; _phase_tma_full ^= 1; }
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
            unsigned int mms_g0_2 = 0;
            unsigned int mms_base_2 = 0;
            unsigned int mms_mb_2 = 0;
            unsigned int mms_incl_2 = 0;
            unsigned int mms_m_3 = 0;
            if (mms_g0_2 + (unsigned int)lane < num_groups) {
                mms_m_3 = (unsigned int)masked_m[(int)mms_g0_2 + lane];
            }
            mms_mb_2 = (mms_m_3 + 128 - 1) / 128;
            uint32_t _warp_scan_sum_0_2 = mms_mb_2;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_2, 1, 32); if (lane >= 1) { _warp_scan_sum_0_2 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_2, 2, 32); if (lane >= 2) { _warp_scan_sum_0_2 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_2, 4, 32); if (lane >= 4) { _warp_scan_sum_0_2 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_2, 8, 32); if (lane >= 8) { _warp_scan_sum_0_2 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_2, 16, 32); if (lane >= 16) { _warp_scan_sum_0_2 += peer; } }
            mms_incl_2 = _warp_scan_sum_0_2;
            unsigned int mms_blocks_3 = 0;
            #pragma unroll 32
            for (int mms_g_2 = 0; mms_g_2 < num_groups; mms_g_2++) {
                mms_blocks_3 = mms_blocks_3 + ((unsigned int)masked_m[mms_g_2] + 128 - 1) / 128;
            }
            unsigned int mms_tiles_2 = mms_blocks_3 * grid_n;
            unsigned int _phase_tma_free = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = worker_idx_2; tile_idx_1 < mms_tiles_2; tile_idx_1 += num_workers_2) {
                unsigned int mms_hit_1 = __ballot_sync(0xFFFFFFFF, tile_idx_1 < (mms_base_2 + mms_incl_2) * grid_n);
                #pragma unroll 1
                for (unsigned int mms_c_1 = mms_g0_2 + 32; mms_c_1 < num_groups; mms_c_1 += 32) {
                    if (mms_hit_1 != 0) {
                        break;
                    }
                    unsigned int _shfl_sync_1_1 = __shfl_sync(0xFFFFFFFF, mms_incl_2, 31);
                    mms_base_2 = mms_base_2 + _shfl_sync_1_1;
                    mms_g0_2 = mms_c_1;
                    unsigned int mms_m_0 = 0;
                    if (mms_g0_2 + (unsigned int)lane < num_groups) {
                        mms_m_0 = (unsigned int)masked_m[(int)mms_g0_2 + lane];
                    }
                    mms_mb_2 = (mms_m_0 + 128 - 1) / 128;
                    uint32_t _warp_scan_sum_2_1 = mms_mb_2;
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_1, 1, 32); if (lane >= 1) { _warp_scan_sum_2_1 += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_1, 2, 32); if (lane >= 2) { _warp_scan_sum_2_1 += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_1, 4, 32); if (lane >= 4) { _warp_scan_sum_2_1 += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_1, 8, 32); if (lane >= 8) { _warp_scan_sum_2_1 += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_1, 16, 32); if (lane >= 16) { _warp_scan_sum_2_1 += peer; } }
                    mms_incl_2 = _warp_scan_sum_2_1;
                    unsigned int _warp_vote_3_1 = __ballot_sync(0xFFFFFFFF, tile_idx_1 < (mms_base_2 + mms_incl_2) * grid_n);
                    mms_hit_1 = _warp_vote_3_1;
                }
                int _find_first_set_4_1 = __ffs(mms_hit_1);
                int mms_lane_1 = _find_first_set_4_1 - 1;
                unsigned int mms_group_1 = mms_g0_2 + (unsigned int)mms_lane_1;
                unsigned int mms_blocks_0 = __shfl_sync(0xFFFFFFFF, mms_mb_2, mms_lane_1);
                unsigned int _shfl_sync_5_1 = __shfl_sync(0xFFFFFFFF, mms_incl_2, mms_lane_1);
                unsigned int mms_start_1 = mms_base_2 + _shfl_sync_5_1 - mms_blocks_0;
                unsigned int mms_in_group_1 = tile_idx_1 - mms_start_1 * grid_n;
                unsigned int mms_per_l2_1 = mms_blocks_0 * 16;
                unsigned int mms_first_n_1 = mms_in_group_1 / mms_per_l2_1 * 16;
                unsigned int mms_in_l2_1 = mms_in_group_1 % mms_per_l2_1;
                unsigned int mms_left_n_1 = grid_n - mms_first_n_1;
                unsigned int mms_n_in_l2_1 = ((mms_left_n_1 > (unsigned int)16) ? (unsigned int)16 : mms_left_n_1);
                unsigned int mms_pair_block_1 = mms_in_l2_1 / mms_n_in_l2_1;
                unsigned int mms_n_block_1 = mms_first_n_1 + mms_in_l2_1 % mms_n_in_l2_1;
                unsigned int raw_m_block_1 = mms_pair_block_1;
                unsigned int m_block = ((raw_m_block_1 < max_m_blocks) ? raw_m_block_1 : mms_pair_block_1);
                unsigned int off_m_1 = m_block * 128;
                unsigned int off_n_1 = mms_n_block_1 * 144;
                unsigned int flat_group_m = mms_group_1 * shape_m + off_m_1;
                unsigned int sfb_group_base = mms_group_1 * (shape_n / 128);
                unsigned int last_sfb_block = shape_n / 128 - 1;
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < k_tiles; iter_k_1++) {
                    mbarrier_wait(tma_free_addr + (load_stage) * 8, _phase_tma_free);
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_a_addr + load_stage * 72704, (&A), 0, off_m_1, (unsigned int)iter_k_1 * 2, mms_group_1, tma_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(smem_b_addr + load_stage * 72704, (&B), 0, off_n_1, (unsigned int)iter_k_1 * 2, mms_group_1, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 69632);
                    }
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_tma_free ^= 1; }
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
            unsigned int mms_g0_3 = 0;
            unsigned int mms_base_3 = 0;
            unsigned int mms_mb_3 = 0;
            unsigned int mms_incl_3 = 0;
            unsigned int mms_m_4 = 0;
            if (mms_g0_3 + (unsigned int)lane < num_groups) {
                mms_m_4 = (unsigned int)masked_m[(int)mms_g0_3 + lane];
            }
            mms_mb_3 = (mms_m_4 + 128 - 1) / 128;
            uint32_t _warp_scan_sum_0_3 = mms_mb_3;
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_3, 1, 32); if (lane >= 1) { _warp_scan_sum_0_3 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_3, 2, 32); if (lane >= 2) { _warp_scan_sum_0_3 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_3, 4, 32); if (lane >= 4) { _warp_scan_sum_0_3 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_3, 8, 32); if (lane >= 8) { _warp_scan_sum_0_3 += peer; } }
            { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_0_3, 16, 32); if (lane >= 16) { _warp_scan_sum_0_3 += peer; } }
            mms_incl_3 = _warp_scan_sum_0_3;
            unsigned int mms_blocks_4 = 0;
            #pragma unroll 32
            for (int mms_g_3 = 0; mms_g_3 < num_groups; mms_g_3++) {
                mms_blocks_4 = mms_blocks_4 + ((unsigned int)masked_m[mms_g_3] + 128 - 1) / 128;
            }
            unsigned int mms_tiles_3 = mms_blocks_4 * grid_n;
            unsigned int _phase_tma_free_1 = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = worker_idx_3; tile_idx_2 < mms_tiles_3; tile_idx_2 += num_workers_3) {
                unsigned int mms_hit_2 = __ballot_sync(0xFFFFFFFF, tile_idx_2 < (mms_base_3 + mms_incl_3) * grid_n);
                #pragma unroll 1
                for (unsigned int mms_c_2 = mms_g0_3 + 32; mms_c_2 < num_groups; mms_c_2 += 32) {
                    if (mms_hit_2 != 0) {
                        break;
                    }
                    unsigned int _shfl_sync_1_2 = __shfl_sync(0xFFFFFFFF, mms_incl_3, 31);
                    mms_base_3 = mms_base_3 + _shfl_sync_1_2;
                    mms_g0_3 = mms_c_2;
                    unsigned int mms_m_1_1 = 0;
                    if (mms_g0_3 + (unsigned int)lane < num_groups) {
                        mms_m_1_1 = (unsigned int)masked_m[(int)mms_g0_3 + lane];
                    }
                    mms_mb_3 = (mms_m_1_1 + 128 - 1) / 128;
                    uint32_t _warp_scan_sum_2_2 = mms_mb_3;
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_2, 1, 32); if (lane >= 1) { _warp_scan_sum_2_2 += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_2, 2, 32); if (lane >= 2) { _warp_scan_sum_2_2 += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_2, 4, 32); if (lane >= 4) { _warp_scan_sum_2_2 += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_2, 8, 32); if (lane >= 8) { _warp_scan_sum_2_2 += peer; } }
                    { uint32_t peer = __shfl_up_sync(0xFFFFFFFF, _warp_scan_sum_2_2, 16, 32); if (lane >= 16) { _warp_scan_sum_2_2 += peer; } }
                    mms_incl_3 = _warp_scan_sum_2_2;
                    unsigned int _warp_vote_3_2 = __ballot_sync(0xFFFFFFFF, tile_idx_2 < (mms_base_3 + mms_incl_3) * grid_n);
                    mms_hit_2 = _warp_vote_3_2;
                }
                int _find_first_set_4_2 = __ffs(mms_hit_2);
                int mms_lane_2 = _find_first_set_4_2 - 1;
                unsigned int mms_group_2 = mms_g0_3 + (unsigned int)mms_lane_2;
                unsigned int mms_blocks_1_1 = __shfl_sync(0xFFFFFFFF, mms_mb_3, mms_lane_2);
                unsigned int _shfl_sync_5_2 = __shfl_sync(0xFFFFFFFF, mms_incl_3, mms_lane_2);
                unsigned int mms_start_2 = mms_base_3 + _shfl_sync_5_2 - mms_blocks_1_1;
                unsigned int mms_in_group_2 = tile_idx_2 - mms_start_2 * grid_n;
                unsigned int mms_per_l2_2 = mms_blocks_1_1 * 16;
                unsigned int mms_first_n_2 = mms_in_group_2 / mms_per_l2_2 * 16;
                unsigned int mms_in_l2_2 = mms_in_group_2 % mms_per_l2_2;
                unsigned int mms_left_n_2 = grid_n - mms_first_n_2;
                unsigned int mms_n_in_l2_2 = ((mms_left_n_2 > (unsigned int)16) ? (unsigned int)16 : mms_left_n_2);
                unsigned int mms_pair_block_2 = mms_in_l2_2 / mms_n_in_l2_2;
                unsigned int mms_n_block_2 = mms_first_n_2 + mms_in_l2_2 % mms_n_in_l2_2;
                unsigned int raw_m_block_2 = mms_pair_block_2;
                unsigned int m_block_1 = ((raw_m_block_2 < max_m_blocks_1) ? raw_m_block_2 : mms_pair_block_2);
                unsigned int off_m_2 = m_block_1 * 128;
                unsigned int off_n_2 = mms_n_block_2 * 144;
                unsigned int flat_group_m_1 = mms_group_2 * shape_m + off_m_2;
                unsigned int sfb_group_base_1 = mms_group_2 * (shape_n / 128);
                unsigned int last_sfb_block_1 = shape_n / 128 - 1;
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < k_tiles; iter_k_2++) {
                    mbarrier_wait(tma_free_addr + (sf_stage) * 8, _phase_tma_free_1);
                    int sfa_base = smem_sfa_addr + sf_stage * 72704;
                    int sfb_base = smem_sfb_addr + sf_stage * 72704;
                    unsigned int sf_lane = (unsigned int)lane;
                    unsigned int sf_row_1_0 = (unsigned int)0 + sf_lane;
                    unsigned int sfa_lane_dst_1_0 = sf_lane * 16;
                    unsigned int sfa_idx0_1_0 = (flat_group_m_1 + sf_row_1_0) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfa_v_1_0 = (unsigned int)(SFA_bits[sfa_idx0_1_0] >> 23);
                    unsigned int sfa_word_2_0 = sfa_v_1_0 | sfa_v_1_0 << 8 | sfa_v_1_0 << 16 | sfa_v_1_0 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sfa_lane_dst_1_0), "r"(sfa_word_2_0));
                    unsigned int sfa_v_1_1 = (unsigned int)(SFA_bits[sfa_idx0_1_0 + 1] >> 23);
                    unsigned int sfa_word_2_1 = sfa_v_1_1 | sfa_v_1_1 << 8 | sfa_v_1_1 << 16 | sfa_v_1_1 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfa_base + 512) + sfa_lane_dst_1_0), "r"(sfa_word_2_1));
                    unsigned int sf_row_1_1 = (unsigned int)32 + sf_lane;
                    unsigned int sfa_lane_dst_1_1 = sf_lane * 16 + 4;
                    unsigned int sfa_idx0_1_1 = (flat_group_m_1 + sf_row_1_1) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfa_v_1_2 = (unsigned int)(SFA_bits[sfa_idx0_1_1] >> 23);
                    unsigned int sfa_word_2_2 = sfa_v_1_2 | sfa_v_1_2 << 8 | sfa_v_1_2 << 16 | sfa_v_1_2 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sfa_lane_dst_1_1), "r"(sfa_word_2_2));
                    unsigned int sfa_v_1_3 = (unsigned int)(SFA_bits[sfa_idx0_1_1 + 1] >> 23);
                    unsigned int sfa_word_2_3 = sfa_v_1_3 | sfa_v_1_3 << 8 | sfa_v_1_3 << 16 | sfa_v_1_3 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfa_base + 512) + sfa_lane_dst_1_1), "r"(sfa_word_2_3));
                    unsigned int sf_row_1_2 = (unsigned int)64 + sf_lane;
                    unsigned int sfa_lane_dst_1_2 = sf_lane * 16 + 8;
                    unsigned int sfa_idx0_1_2 = (flat_group_m_1 + sf_row_1_2) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfa_v_1_4 = (unsigned int)(SFA_bits[sfa_idx0_1_2] >> 23);
                    unsigned int sfa_word_2_4 = sfa_v_1_4 | sfa_v_1_4 << 8 | sfa_v_1_4 << 16 | sfa_v_1_4 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sfa_lane_dst_1_2), "r"(sfa_word_2_4));
                    unsigned int sfa_v_1_5 = (unsigned int)(SFA_bits[sfa_idx0_1_2 + 1] >> 23);
                    unsigned int sfa_word_2_5 = sfa_v_1_5 | sfa_v_1_5 << 8 | sfa_v_1_5 << 16 | sfa_v_1_5 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfa_base + 512) + sfa_lane_dst_1_2), "r"(sfa_word_2_5));
                    unsigned int sf_row_1_3 = (unsigned int)96 + sf_lane;
                    unsigned int sfa_lane_dst_1_3 = sf_lane * 16 + 12;
                    unsigned int sfa_idx0_1_3 = (flat_group_m_1 + sf_row_1_3) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfa_v_1_6 = (unsigned int)(SFA_bits[sfa_idx0_1_3] >> 23);
                    unsigned int sfa_word_2_6 = sfa_v_1_6 | sfa_v_1_6 << 8 | sfa_v_1_6 << 16 | sfa_v_1_6 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sfa_lane_dst_1_3), "r"(sfa_word_2_6));
                    unsigned int sfa_v_1_7 = (unsigned int)(SFA_bits[sfa_idx0_1_3 + 1] >> 23);
                    unsigned int sfa_word_2_7 = sfa_v_1_7 | sfa_v_1_7 << 8 | sfa_v_1_7 << 16 | sfa_v_1_7 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfa_base + 512) + sfa_lane_dst_1_3), "r"(sfa_word_2_7));
                    unsigned int sfb_block_idx = 0;
                    unsigned int n_glob = off_n_2 + (unsigned int)0 + sf_lane;
                    unsigned int n_sf_block = n_glob / 128;
                    unsigned int n_sf_block_safe = ((n_glob < shape_n) ? n_sf_block : last_sfb_block_1);
                    unsigned int sfb_idx0 = (sfb_group_base_1 + n_sf_block_safe) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfb_lane_dst = sfb_block_idx * 512 + sf_lane * 16;
                    unsigned int sfb_v_1 = (unsigned int)(SFB_bits[sfb_idx0] >> 23);
                    unsigned int sfb_word_1 = sfb_v_1 | sfb_v_1 << 8 | sfb_v_1 << 16 | sfb_v_1 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst), "r"(sfb_word_1));
                    unsigned int sfb_v_1_0 = (unsigned int)(SFB_bits[sfb_idx0 + 1] >> 23);
                    unsigned int sfb_word_1_0 = sfb_v_1_0 | sfb_v_1_0 << 8 | sfb_v_1_0 << 16 | sfb_v_1_0 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfb_base + 1024) + sfb_lane_dst), "r"(sfb_word_1_0));
                    unsigned int sfb_block_idx_r_0 = 0;
                    unsigned int n_glob_r_0 = off_n_2 + (unsigned int)32 + sf_lane;
                    unsigned int n_sf_block_r_0 = n_glob_r_0 / 128;
                    unsigned int n_sf_block_safe_r_0 = ((n_glob_r_0 < shape_n) ? n_sf_block_r_0 : last_sfb_block_1);
                    unsigned int sfb_idx0_r_0 = (sfb_group_base_1 + n_sf_block_safe_r_0) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfb_lane_dst_r_0 = sfb_block_idx_r_0 * 512 + sf_lane * 16 + 4;
                    unsigned int sfb_v_1_1 = (unsigned int)(SFB_bits[sfb_idx0_r_0] >> 23);
                    unsigned int sfb_word_1_1 = sfb_v_1_1 | sfb_v_1_1 << 8 | sfb_v_1_1 << 16 | sfb_v_1_1 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_r_0), "r"(sfb_word_1_1));
                    unsigned int sfb_v_1_2 = (unsigned int)(SFB_bits[sfb_idx0_r_0 + 1] >> 23);
                    unsigned int sfb_word_1_2 = sfb_v_1_2 | sfb_v_1_2 << 8 | sfb_v_1_2 << 16 | sfb_v_1_2 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfb_base + 1024) + sfb_lane_dst_r_0), "r"(sfb_word_1_2));
                    unsigned int sfb_block_idx_r_1 = 0;
                    unsigned int n_glob_r_1 = off_n_2 + (unsigned int)64 + sf_lane;
                    unsigned int n_sf_block_r_1 = n_glob_r_1 / 128;
                    unsigned int n_sf_block_safe_r_1 = ((n_glob_r_1 < shape_n) ? n_sf_block_r_1 : last_sfb_block_1);
                    unsigned int sfb_idx0_r_1 = (sfb_group_base_1 + n_sf_block_safe_r_1) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfb_lane_dst_r_1 = sfb_block_idx_r_1 * 512 + sf_lane * 16 + 8;
                    unsigned int sfb_v_1_3 = (unsigned int)(SFB_bits[sfb_idx0_r_1] >> 23);
                    unsigned int sfb_word_1_3 = sfb_v_1_3 | sfb_v_1_3 << 8 | sfb_v_1_3 << 16 | sfb_v_1_3 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_r_1), "r"(sfb_word_1_3));
                    unsigned int sfb_v_1_4 = (unsigned int)(SFB_bits[sfb_idx0_r_1 + 1] >> 23);
                    unsigned int sfb_word_1_4 = sfb_v_1_4 | sfb_v_1_4 << 8 | sfb_v_1_4 << 16 | sfb_v_1_4 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfb_base + 1024) + sfb_lane_dst_r_1), "r"(sfb_word_1_4));
                    unsigned int sfb_block_idx_r_2 = 0;
                    unsigned int n_glob_r_2 = off_n_2 + (unsigned int)96 + sf_lane;
                    unsigned int n_sf_block_r_2 = n_glob_r_2 / 128;
                    unsigned int n_sf_block_safe_r_2 = ((n_glob_r_2 < shape_n) ? n_sf_block_r_2 : last_sfb_block_1);
                    unsigned int sfb_idx0_r_2 = (sfb_group_base_1 + n_sf_block_safe_r_2) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfb_lane_dst_r_2 = sfb_block_idx_r_2 * 512 + sf_lane * 16 + 12;
                    unsigned int sfb_v_1_5 = (unsigned int)(SFB_bits[sfb_idx0_r_2] >> 23);
                    unsigned int sfb_word_1_5 = sfb_v_1_5 | sfb_v_1_5 << 8 | sfb_v_1_5 << 16 | sfb_v_1_5 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_r_2), "r"(sfb_word_1_5));
                    unsigned int sfb_v_1_6 = (unsigned int)(SFB_bits[sfb_idx0_r_2 + 1] >> 23);
                    unsigned int sfb_word_1_6 = sfb_v_1_6 | sfb_v_1_6 << 8 | sfb_v_1_6 << 16 | sfb_v_1_6 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfb_base + 1024) + sfb_lane_dst_r_2), "r"(sfb_word_1_6));
                    unsigned int sfb_block_idx_55 = 1;
                    unsigned int n_glob_56 = off_n_2 + (unsigned int)128 + sf_lane;
                    unsigned int n_sf_block_57 = n_glob_56 / 128;
                    unsigned int n_sf_block_safe_58 = ((n_glob_56 < shape_n) ? n_sf_block_57 : last_sfb_block_1);
                    unsigned int sfb_idx0_59 = (sfb_group_base_1 + n_sf_block_safe_58) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfb_lane_dst_60 = sfb_block_idx_55 * 512 + sf_lane * 16;
                    unsigned int sfb_v_1_7 = (unsigned int)(SFB_bits[sfb_idx0_59] >> 23);
                    unsigned int sfb_word_1_7 = sfb_v_1_7 | sfb_v_1_7 << 8 | sfb_v_1_7 << 16 | sfb_v_1_7 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_60), "r"(sfb_word_1_7));
                    unsigned int sfb_v_1_8 = (unsigned int)(SFB_bits[sfb_idx0_59 + 1] >> 23);
                    unsigned int sfb_word_1_8 = sfb_v_1_8 | sfb_v_1_8 << 8 | sfb_v_1_8 << 16 | sfb_v_1_8 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfb_base + 1024) + sfb_lane_dst_60), "r"(sfb_word_1_8));
                    unsigned int sfb_block_idx_r_3 = 1;
                    unsigned int n_glob_r_3 = off_n_2 + (unsigned int)160 + sf_lane;
                    unsigned int n_sf_block_r_3 = n_glob_r_3 / 128;
                    unsigned int n_sf_block_safe_r_3 = ((n_glob_r_3 < shape_n) ? n_sf_block_r_3 : last_sfb_block_1);
                    unsigned int sfb_idx0_r_3 = (sfb_group_base_1 + n_sf_block_safe_r_3) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfb_lane_dst_r_3 = sfb_block_idx_r_3 * 512 + sf_lane * 16 + 4;
                    unsigned int sfb_v_1_9 = (unsigned int)(SFB_bits[sfb_idx0_r_3] >> 23);
                    unsigned int sfb_word_1_9 = sfb_v_1_9 | sfb_v_1_9 << 8 | sfb_v_1_9 << 16 | sfb_v_1_9 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_r_3), "r"(sfb_word_1_9));
                    unsigned int sfb_v_1_10 = (unsigned int)(SFB_bits[sfb_idx0_r_3 + 1] >> 23);
                    unsigned int sfb_word_1_10 = sfb_v_1_10 | sfb_v_1_10 << 8 | sfb_v_1_10 << 16 | sfb_v_1_10 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfb_base + 1024) + sfb_lane_dst_r_3), "r"(sfb_word_1_10));
                    unsigned int sfb_block_idx_r_4 = 1;
                    unsigned int n_glob_r_4 = off_n_2 + (unsigned int)192 + sf_lane;
                    unsigned int n_sf_block_r_4 = n_glob_r_4 / 128;
                    unsigned int n_sf_block_safe_r_4 = ((n_glob_r_4 < shape_n) ? n_sf_block_r_4 : last_sfb_block_1);
                    unsigned int sfb_idx0_r_4 = (sfb_group_base_1 + n_sf_block_safe_r_4) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfb_lane_dst_r_4 = sfb_block_idx_r_4 * 512 + sf_lane * 16 + 8;
                    unsigned int sfb_v_1_11 = (unsigned int)(SFB_bits[sfb_idx0_r_4] >> 23);
                    unsigned int sfb_word_1_11 = sfb_v_1_11 | sfb_v_1_11 << 8 | sfb_v_1_11 << 16 | sfb_v_1_11 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_r_4), "r"(sfb_word_1_11));
                    unsigned int sfb_v_1_12 = (unsigned int)(SFB_bits[sfb_idx0_r_4 + 1] >> 23);
                    unsigned int sfb_word_1_12 = sfb_v_1_12 | sfb_v_1_12 << 8 | sfb_v_1_12 << 16 | sfb_v_1_12 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfb_base + 1024) + sfb_lane_dst_r_4), "r"(sfb_word_1_12));
                    unsigned int sfb_block_idx_r_5 = 1;
                    unsigned int n_glob_r_5 = off_n_2 + (unsigned int)224 + sf_lane;
                    unsigned int n_sf_block_r_5 = n_glob_r_5 / 128;
                    unsigned int n_sf_block_safe_r_5 = ((n_glob_r_5 < shape_n) ? n_sf_block_r_5 : last_sfb_block_1);
                    unsigned int sfb_idx0_r_5 = (sfb_group_base_1 + n_sf_block_safe_r_5) * sf_cols + (unsigned int)iter_k_2 * 2;
                    unsigned int sfb_lane_dst_r_5 = sfb_block_idx_r_5 * 512 + sf_lane * 16 + 12;
                    unsigned int sfb_v_1_13 = (unsigned int)(SFB_bits[sfb_idx0_r_5] >> 23);
                    unsigned int sfb_word_1_13 = sfb_v_1_13 | sfb_v_1_13 << 8 | sfb_v_1_13 << 16 | sfb_v_1_13 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sfb_lane_dst_r_5), "r"(sfb_word_1_13));
                    unsigned int sfb_v_1_14 = (unsigned int)(SFB_bits[sfb_idx0_r_5 + 1] >> 23);
                    unsigned int sfb_word_1_14 = sfb_v_1_14 | sfb_v_1_14 << 8 | sfb_v_1_14 << 16 | sfb_v_1_14 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)(sfb_base + 1024) + sfb_lane_dst_r_5), "r"(sfb_word_1_14));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(sf_full_addr + (sf_stage) * 8);
                    }
                    sf_stage += 1;
                    if (sf_stage == 3) { sf_stage = 0; _phase_tma_free_1 ^= 1; }
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
