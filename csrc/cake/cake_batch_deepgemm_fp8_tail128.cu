/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// clang-format off
// BEGIN FROZEN CAKE EXPORT
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 272
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 264
#define NUM_TMA_PIPE_STAGES 3
#define NUM_MAINLOOP_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 67584
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 67584
#define SMEM_SMEM_SFA_OFF 66560
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 67584
#define SMEM_SMEM_SFB_OFF 67584
#define SMEM_SMEM_SFB_STAGE_BYTES 1024
#define SMEM_SMEM_SFB_STRIDE 67584
#define SMEM_TOTAL 203776

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
        :: "r"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
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


__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::mxf8f6f4 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
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


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo256(int addr) {
    const int SBO = 256;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

extern "C" {

__global__ __launch_bounds__(192, 1) void
kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_large_nk_cta1_tail128(CakeTensorMap const* A, CakeTensorMap const* B, int* __restrict__ SFA_packed, int* __restrict__ SFB_packed, int* __restrict__ masked_m, __nv_bfloat16* __restrict__ C, unsigned int num_groups, unsigned int shape_m, unsigned int grid_n, unsigned int k_tiles, unsigned int sf_cols, unsigned int scheduled_m_blocks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_sfa_addr = smem + 66560;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_sfb_addr = smem + 67584;
    if (warp == 5) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(A)) : "memory"); }
    if (warp == 5) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(B)) : "memory"); }

    // Mbarrier init (4 groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

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
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // epilogue_done: 2 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 272 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 0) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define tma_free_addr (mbar_base + 24)
    #define mainloop_done_addr (mbar_base + 48)
    #define epilogue_done_addr (mbar_base + 64)
    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 264;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp;
            const int epi_tid = epi_warp * 32 + lane;
            unsigned int num_workers = (unsigned int)num_bids;
            unsigned int worker_idx = (unsigned int)bid;
            unsigned int exact_pair_blocks = 0;
            #pragma unroll 1
            for (int count_g = 0; count_g < num_groups; count_g++) {
                unsigned int count_m = (unsigned int)masked_m[count_g];
                if (shape_m == 256 && count_m <= 224) {
                    count_m = 0;
                }
                if (shape_m == 256 && count_m > 224) {
                    count_m = 128;
                }
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
                    if (shape_m == 256 && group_m <= 224) {
                        group_m = 0;
                    }
                    if (shape_m == 256 && group_m > 224) {
                        group_m = 128;
                    }
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
                if (shape_m == 256) {
                    off_m = off_m + 128;
                }
                unsigned int off_n = n_block * 128;
                unsigned long long flat_row = (unsigned long long)selected_group * (unsigned long long)shape_m + (unsigned long long)off_m + (unsigned long long)epi_tid;
                #pragma unroll
                for (int n_chunk = 0; n_chunk < 16; n_chunk++) {
                    int row = epi_warp * 32;
                    int col = (int)epi_stage * 128 + n_chunk * 8;
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
                    if (store_tile != 0) {
                        unsigned long long out_col = (unsigned long long)off_n + (unsigned long long)(n_chunk * 8);
                        reinterpret_cast<int4*>(C + (flat_row * ((unsigned long long)grid_n * (unsigned long long)128) + out_col))[0] = reinterpret_cast<int4*>(_tmem_load_0_bf16)[0];
                    }
                }
                if (elect_sync()) {
                    mbarrier_arrive(epilogue_done_addr + (epi_stage) * 8);
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
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
                if (shape_m == 256 && count_m_1 <= 224) {
                    count_m_1 = 0;
                }
                if (shape_m == 256 && count_m_1 > 224) {
                    count_m_1 = 128;
                }
                unsigned int count_m_blocks_1 = (count_m_1 + 128 - 1) / 128;
                exact_pair_blocks_1 = exact_pair_blocks_1 + (count_m_blocks_1 + 1 - 1);
            }
            unsigned int scheduled_total_tiles_1 = exact_pair_blocks_1 * grid_n;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_idx = worker_idx_1; _tile_idx < scheduled_total_tiles_1; _tile_idx += num_workers_1) {
                mbarrier_wait(epilogue_done_addr + (epi_stage_1) * 8, _phase_epilogue_done);
                #pragma unroll 1
                for (int iter_k = 0; iter_k < k_tiles; iter_k++) {
                    mbarrier_wait(tma_full_addr + (tma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((iter_k == 0) ? 1 : 0);
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfa, make_sf_cp_desc_sbo256(smem_sfa_addr + tma_stage * 67584));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfa + 4), make_sf_cp_desc_sbo256((smem_sfa_addr + tma_stage * 67584 + 128)));
                        tcgen05_cp_32x128b_warpx4(tmem_tmem_sfb, make_sf_cp_desc_sbo256(smem_sfb_addr + tma_stage * 67584));
                        tcgen05_cp_32x128b_warpx4((tmem_tmem_sfb + 4), make_sf_cp_desc_sbo256((smem_sfb_addr + tma_stage * 67584 + 128)));
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (tma_stage) * 4224;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (tma_stage) * 4224;
                        {
                            uint64_t a_desc = ((uint64_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 128)), a_desc + 0, b_desc + 0,
                                0x8a00000U, tmem_tmem_sfa, tmem_tmem_sfb, ((init_flag) ? 0 : 1));
                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 128)), a_desc + 2, b_desc + 2,
                                0x28a00010U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 128)), a_desc + 4, b_desc + 4,
                                0x48a00020U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 128)), a_desc + 6, b_desc + 6,
                                0x68a00030U, tmem_tmem_sfa, tmem_tmem_sfb, 1);
                        }
                        int _mma_a_lo_1 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (tma_stage) * 4224;
                        int _mma_b_lo_1 = (((smem_b_addr + 16384) >> 4) & 0x3FFF) + (tma_stage) * 4224;
                        {
                            uint64_t a_desc = ((uint64_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 128)), a_desc + 0, b_desc + 0,
                                0x8a00000U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 128)), a_desc + 2, b_desc + 2,
                                0x28a00010U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 128)), a_desc + 4, b_desc + 4,
                                0x48a00020U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                            tcgen05_mma_mxf8_bs((tmem_accum + (epi_stage_1 * 128)), a_desc + 6, b_desc + 6,
                                0x68a00030U, tmem_tmem_sfa + 4, tmem_tmem_sfb + 4, 1);
                        }
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
            unsigned int sf_packed_cols = sf_cols / 4;
            unsigned int num_workers_2 = (unsigned int)num_bids;
            unsigned int worker_idx_2 = (unsigned int)bid;
            unsigned int max_m_blocks = (shape_m + 128 - 1) / 128;
            unsigned int exact_pair_blocks_2 = 0;
            #pragma unroll 1
            for (int count_g_2 = 0; count_g_2 < num_groups; count_g_2++) {
                unsigned int count_m_2 = (unsigned int)masked_m[count_g_2];
                if (shape_m == 256 && count_m_2 <= 224) {
                    count_m_2 = 0;
                }
                if (shape_m == 256 && count_m_2 > 224) {
                    count_m_2 = 128;
                }
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
                    if (shape_m == 256 && group_m_1 <= 224) {
                        group_m_1 = 0;
                    }
                    if (shape_m == 256 && group_m_1 > 224) {
                        group_m_1 = 128;
                    }
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
                if (shape_m == 256) {
                    m_block = m_block + 1;
                }
                unsigned int off_m_1 = m_block * 128;
                unsigned int off_n_1 = n_block_1 * 128;
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < k_tiles; iter_k_1++) {
                    mbarrier_wait(tma_free_addr + (load_stage) * 8, _phase_tma_free);
                    int sfa_base = smem_sfa_addr + load_stage * 67584;
                    int sfb_base = smem_sfb_addr + load_stage * 67584;
                    unsigned int sf_pack_col = (unsigned int)iter_k_1 / 2;
                    unsigned int sf_shift = (unsigned int)iter_k_1 % 2 * 16;
                    unsigned int sfb_idx = (selected_group_1 * sf_packed_cols + sf_pack_col) * grid_n + n_block_1;
                    unsigned int sfb_packed_word = (unsigned int)SFB_packed[sfb_idx];
                    unsigned int sfb0 = sfb_packed_word >> sf_shift & 255;
                    unsigned int sfb1 = sfb_packed_word >> sf_shift + 8 & 255;
                    unsigned int sfb0_word = sfb0 | sfb0 << 8 | sfb0 << 16 | sfb0 << 24;
                    unsigned int sfb1_word = sfb1 | sfb1 << 8 | sfb1 << 16 | sfb1 << 24;
                    unsigned int sf_row = (unsigned int)0 + (unsigned int)lane;
                    unsigned int sf_c = (unsigned int)lane / 8;
                    unsigned int sf_d = (unsigned int)lane % 8;
                    unsigned int sf_dst0 = (sf_c * 2 * 8 + sf_d) * 16;
                    unsigned int sf_dst1 = ((sf_c * 2 + 1) * 8 + sf_d) * 16;
                    unsigned int sfa_idx = (selected_group_1 * sf_packed_cols + sf_pack_col) * shape_m + off_m_1 + sf_row;
                    unsigned int sfa_packed_word = (unsigned int)SFA_packed[sfa_idx];
                    unsigned int sfa0 = sfa_packed_word >> sf_shift & 255;
                    unsigned int sfa1 = sfa_packed_word >> sf_shift + 8 & 255;
                    unsigned int sfa0_word = sfa0 | sfa0 << 8 | sfa0 << 16 | sfa0 << 24;
                    unsigned int sfa1_word = sfa1 | sfa1 << 8 | sfa1 << 16 | sfa1 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sf_dst0), "r"(sfa0_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sf_dst1), "r"(sfa1_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sf_dst0), "r"(sfb0_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sf_dst1), "r"(sfb1_word));
                    unsigned int sf_row_0 = (unsigned int)32 + (unsigned int)lane;
                    unsigned int sf_c_1 = (unsigned int)lane / 8;
                    unsigned int sf_d_2 = (unsigned int)lane % 8;
                    unsigned int sf_dst0_3 = (sf_c_1 * 2 * 8 + sf_d_2) * 16 + 4;
                    unsigned int sf_dst1_4 = ((sf_c_1 * 2 + 1) * 8 + sf_d_2) * 16 + 4;
                    unsigned int sfa_idx_5 = (selected_group_1 * sf_packed_cols + sf_pack_col) * shape_m + off_m_1 + sf_row_0;
                    unsigned int sfa_packed_word_6 = (unsigned int)SFA_packed[sfa_idx_5];
                    unsigned int sfa0_7 = sfa_packed_word_6 >> sf_shift & 255;
                    unsigned int sfa1_8 = sfa_packed_word_6 >> sf_shift + 8 & 255;
                    unsigned int sfa0_word_9 = sfa0_7 | sfa0_7 << 8 | sfa0_7 << 16 | sfa0_7 << 24;
                    unsigned int sfa1_word_10 = sfa1_8 | sfa1_8 << 8 | sfa1_8 << 16 | sfa1_8 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sf_dst0_3), "r"(sfa0_word_9));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sf_dst1_4), "r"(sfa1_word_10));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sf_dst0_3), "r"(sfb0_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sf_dst1_4), "r"(sfb1_word));
                    unsigned int sf_row_11 = (unsigned int)64 + (unsigned int)lane;
                    unsigned int sf_c_12 = (unsigned int)lane / 8;
                    unsigned int sf_d_13 = (unsigned int)lane % 8;
                    unsigned int sf_dst0_14 = (sf_c_12 * 2 * 8 + sf_d_13) * 16 + 8;
                    unsigned int sf_dst1_15 = ((sf_c_12 * 2 + 1) * 8 + sf_d_13) * 16 + 8;
                    unsigned int sfa_idx_16 = (selected_group_1 * sf_packed_cols + sf_pack_col) * shape_m + off_m_1 + sf_row_11;
                    unsigned int sfa_packed_word_17 = (unsigned int)SFA_packed[sfa_idx_16];
                    unsigned int sfa0_18 = sfa_packed_word_17 >> sf_shift & 255;
                    unsigned int sfa1_19 = sfa_packed_word_17 >> sf_shift + 8 & 255;
                    unsigned int sfa0_word_20 = sfa0_18 | sfa0_18 << 8 | sfa0_18 << 16 | sfa0_18 << 24;
                    unsigned int sfa1_word_21 = sfa1_19 | sfa1_19 << 8 | sfa1_19 << 16 | sfa1_19 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sf_dst0_14), "r"(sfa0_word_20));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sf_dst1_15), "r"(sfa1_word_21));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sf_dst0_14), "r"(sfb0_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sf_dst1_15), "r"(sfb1_word));
                    unsigned int sf_row_22 = (unsigned int)96 + (unsigned int)lane;
                    unsigned int sf_c_23 = (unsigned int)lane / 8;
                    unsigned int sf_d_24 = (unsigned int)lane % 8;
                    unsigned int sf_dst0_25 = (sf_c_23 * 2 * 8 + sf_d_24) * 16 + 12;
                    unsigned int sf_dst1_26 = ((sf_c_23 * 2 + 1) * 8 + sf_d_24) * 16 + 12;
                    unsigned int sfa_idx_27 = (selected_group_1 * sf_packed_cols + sf_pack_col) * shape_m + off_m_1 + sf_row_22;
                    unsigned int sfa_packed_word_28 = (unsigned int)SFA_packed[sfa_idx_27];
                    unsigned int sfa0_29 = sfa_packed_word_28 >> sf_shift & 255;
                    unsigned int sfa1_30 = sfa_packed_word_28 >> sf_shift + 8 & 255;
                    unsigned int sfa0_word_31 = sfa0_29 | sfa0_29 << 8 | sfa0_29 << 16 | sfa0_29 << 24;
                    unsigned int sfa1_word_32 = sfa1_30 | sfa1_30 << 8 | sfa1_30 << 16 | sfa1_30 << 24;
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sf_dst0_25), "r"(sfa0_word_31));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfa_base + sf_dst1_26), "r"(sfa1_word_32));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sf_dst0_25), "r"(sfb0_word));
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"((unsigned int)sfb_base + sf_dst1_26), "r"(sfb1_word));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_a_addr + load_stage * 67584, A, 0, off_m_1, (unsigned int)iter_k_1 * 2, selected_group_1, tma_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(smem_b_addr + load_stage * 67584, B, 0, off_n_1, (unsigned int)iter_k_1 * 2, selected_group_1, tma_full_addr + (load_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 65536);
                    }
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_tma_free ^= 1; }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"

// END FROZEN CAKE EXPORT
// clang-format on
