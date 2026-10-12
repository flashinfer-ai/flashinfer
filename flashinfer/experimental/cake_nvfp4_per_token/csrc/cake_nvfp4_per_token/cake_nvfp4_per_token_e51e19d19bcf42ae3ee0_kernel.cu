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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_nvfp4_per_token_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 464
#define TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 192
#define TMEM_TMEM_SFB_OFFSET 256
#define TMEM_SFB_PAD_OFFSET 448
#define NUM_TMA_PIPE_STAGES 4
#define NUM_ACC_PIPE_STAGES 1
#define SMEM_SMEM_V0_OFF 1024
#define SMEM_SMEM_V0_STAGE_BYTES 16384
#define SMEM_SMEM_V0_STRIDE 49152
#define SMEM_SMEM_V1_OFF 41984
#define SMEM_SMEM_V1_STAGE_BYTES 2048
#define SMEM_SMEM_V1_STRIDE 49152
#define SMEM_SMEM_V2_OFF 17408
#define SMEM_SMEM_V2_STAGE_BYTES 8192
#define SMEM_SMEM_V2_STRIDE 49152
#define SMEM_SMEM_V3_OFF 44032
#define SMEM_SMEM_V3_STAGE_BYTES 2048
#define SMEM_SMEM_V3_STRIDE 49152
#define SMEM_SMEM_V4_OFF 25600
#define SMEM_SMEM_V4_STAGE_BYTES 8192
#define SMEM_SMEM_V4_STRIDE 49152
#define SMEM_SMEM_V5_OFF 46080
#define SMEM_SMEM_V5_STAGE_BYTES 2048
#define SMEM_SMEM_V5_STRIDE 49152
#define SMEM_SMEM_V6_OFF 33792
#define SMEM_SMEM_V6_STAGE_BYTES 8192
#define SMEM_SMEM_V6_STRIDE 49152
#define SMEM_SMEM_V7_OFF 48128
#define SMEM_SMEM_V7_STAGE_BYTES 2048
#define SMEM_SMEM_V7_STRIDE 49152
#define SMEM_SMEM_OUT_OFF 197632
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_TOTAL 230400
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_cake_nvfp4_per_token_e51e19d19bcf42ae3ee0(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ alpha, const __grid_constant__ CUtensorMap out, int M, int N, int K_tiles, int tok_tiles, int num_tiles)
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
    #define tma_empty_addr (mbar_base + 32)
    #define acc_full_addr (mbar_base + 64)
    #define acc_empty_addr (mbar_base + 72)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_v0 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v0_addr = smem + 1024;
    uint8_t* smem_v1 = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int smem_v1_addr = smem + 41984;
    uint8_t* smem_v2 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v2_addr = smem + 17408;
    uint8_t* smem_v3 = reinterpret_cast<uint8_t*>(smem_raw + 44032);
    const int smem_v3_addr = smem + 44032;
    uint8_t* smem_v4 = reinterpret_cast<uint8_t*>(smem_raw + 25600);
    const int smem_v4_addr = smem + 25600;
    uint8_t* smem_v5 = reinterpret_cast<uint8_t*>(smem_raw + 46080);
    const int smem_v5_addr = smem + 46080;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_v6_addr = smem + 33792;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + 48128);
    const int smem_v7_addr = smem + 48128;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int smem_out_addr = smem + 197632;
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // tma_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // acc_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 72, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 464 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 80);
    if (warp == 2) {
        int _tmem_hold = smem + 80;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_tmem_sfa = taddr + 192;
    const int tmem_tmem_sfb = taddr + 256;
    const int tmem_sfb_pad = taddr + 448;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            int tiles_end = num_tiles;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_tma_empty = 1;
            #pragma unroll 1
            for (unsigned int tile = bid; tile < tiles_end; tile += num_bids) {
                int tile_c = (int)tile;
                int w_idx = tile_c / tok_tiles;
                int tok_idx = tile_c - w_idx * tok_tiles;
                int a_rows = tok_idx * 128;
                int b_rows = w_idx * 192;
                int a_atom = a_rows / 128;
                int b_atom = b_rows / 128;
                int b_in_atom = b_rows - b_atom * 128;
                int sfb_g4 = b_in_atom % 32 / 8;
                #pragma unroll 1
                for (unsigned int k_tile = 0; k_tile < K_tiles; k_tile++) {
                    mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 49152);
                        tma_3d_gmem2smem(smem_v0_addr + load_stage * 49152, (&A), 0, a_rows, (int)k_tile, tma_full_addr + (load_stage) * 8);
                        tma_3d_gmem2smem(smem_v1_addr + load_stage * 49152, (&SFA), 0, 4 * (int)k_tile, a_atom, tma_full_addr + (load_stage) * 8);
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_v2_addr + load_stage * 49152), "l"((&B)), "r"(0), "r"(b_rows), "r"((int)k_tile),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_v3_addr + load_stage * 49152), "l"((&SFB)), "r"(0), "r"(4 * (int)k_tile), "r"(b_atom + b_in_atom / 128),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_v4_addr + load_stage * 49152), "l"((&B)), "r"(0), "r"(b_rows + 64), "r"((int)k_tile),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_v5_addr + load_stage * 49152), "l"((&SFB)), "r"(0), "r"(4 * (int)k_tile), "r"(b_atom + (b_in_atom + 64) / 128),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_v6_addr + load_stage * 49152), "l"((&B)), "r"(0), "r"(b_rows + 128), "r"((int)k_tile),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_v7_addr + load_stage * 49152), "l"((&SFB)), "r"(0), "r"(4 * (int)k_tile), "r"(b_atom + (b_in_atom + 128) / 128),
                               "r"(tma_full_addr + (load_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; _phase_tma_empty ^= 1; }
                }
            }
            #pragma unroll
            for (int _tail = 0; _tail < 4; _tail++) {
                mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                load_stage += 1;
                if (load_stage == 4) { load_stage = 0; _phase_tma_empty ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int completed = 0;
            int tiles_end_1 = num_tiles;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (unsigned int tile_1 = bid; tile_1 < tiles_end_1; tile_1 += num_bids) {
                int tile_c_1 = (int)tile_1;
                int w_idx_1 = tile_c_1 / tok_tiles;
                int tok_idx_1 = tile_c_1 - w_idx_1 * tok_tiles;
                int b_rows_1 = w_idx_1 * 192;
                int sfb_p = b_rows_1 % 128 / 32;
                int acc_base = (int)acc_stage * 192;
                mbarrier_wait(acc_empty_addr + (acc_stage) * 8, _phase_acc_empty);
                #pragma unroll 1
                for (unsigned int k_tile_1 = 0; k_tile_1 < K_tiles; k_tile_1++) {
                    mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int sfa_col = (int)mma_stage * 16;
                    int sfb_col = (int)mma_stage * 48;
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4((unsigned int)tmem_tmem_sfa + mma_stage * 16, make_sf_cp_desc_lo_sbo128((((smem_v1_addr) >> 4) + (mma_stage) * 3072)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v1_addr) >> 4) + (mma_stage) * 3072 + 32)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v1_addr) >> 4) + (mma_stage) * 3072 + 64)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v1_addr) >> 4) + (mma_stage) * 3072 + 96)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4((unsigned int)tmem_tmem_sfb + mma_stage * 3 * 16, make_sf_cp_desc_lo_sbo128((((smem_v3_addr) >> 4) + (mma_stage) * 3072)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfb + mma_stage * 3 * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v3_addr) >> 4) + (mma_stage) * 3072 + 32)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfb + mma_stage * 3 * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v3_addr) >> 4) + (mma_stage) * 3072 + 64)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfb + mma_stage * 3 * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v3_addr) >> 4) + (mma_stage) * 3072 + 96)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4((unsigned int)tmem_tmem_sfb + (mma_stage * 3 + 1) * 16, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_stage) * 3072)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfb + (mma_stage * 3 + 1) * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_stage) * 3072 + 32)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfb + (mma_stage * 3 + 1) * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_stage) * 3072 + 64)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfb + (mma_stage * 3 + 1) * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_stage) * 3072 + 96)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4((unsigned int)tmem_tmem_sfb + (mma_stage * 3 + 2) * 16, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_stage) * 3072)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfb + (mma_stage * 3 + 2) * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_stage) * 3072 + 32)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfb + (mma_stage * 3 + 2) * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_stage) * 3072 + 64)));
                        tcgen05_cp_32x128b_warpx4(((unsigned int)tmem_tmem_sfb + (mma_stage * 3 + 2) * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_stage) * 3072 + 96)));
                    }
                    int init_flag = ((k_tile_1 == 0) ? 1 : 0);
                    int _mma_a_lo_0 = make_warp_uniform((((smem_v0_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_v2_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfa + sfa_col + 0, tmem_tmem_sfb + (sfb_col + b_rows_1 % 128 / 32) + 0, ((((1) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((smem_v0_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_v4_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base + 64)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfa + sfa_col + 0, tmem_tmem_sfb + (sfb_col + 16 + (b_rows_1 + 64) % 128 / 32) + 0, ((((1) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_2 = make_warp_uniform((((smem_v0_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_v6_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base + 128)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfa + sfa_col + 0, tmem_tmem_sfb + (sfb_col + 32 + (b_rows_1 + 128) % 128 / 32) + 0, ((((1) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_3 = make_warp_uniform((((smem_v0_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_v2_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfa + (sfa_col + 4) + 0, tmem_tmem_sfb + (sfb_col + 4 + b_rows_1 % 128 / 32) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_4 = make_warp_uniform((((smem_v0_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_v4_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base + 64)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfa + (sfa_col + 4) + 0, tmem_tmem_sfb + (sfb_col + 16 + 4 + (b_rows_1 + 64) % 128 / 32) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_5 = make_warp_uniform((((smem_v0_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    int _mma_b_lo_5 = make_warp_uniform((((smem_v6_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base + 128)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfa + (sfa_col + 4) + 0, tmem_tmem_sfb + (sfb_col + 32 + 4 + (b_rows_1 + 128) % 128 / 32) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_6 = make_warp_uniform((((smem_v0_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    int _mma_b_lo_6 = make_warp_uniform((((smem_v2_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfa + (sfa_col + 8) + 0, tmem_tmem_sfb + (sfb_col + 8 + b_rows_1 % 128 / 32) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_7 = make_warp_uniform((((smem_v0_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_v4_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base + 64)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfa + (sfa_col + 8) + 0, tmem_tmem_sfb + (sfb_col + 16 + 8 + (b_rows_1 + 64) % 128 / 32) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_8 = make_warp_uniform((((smem_v0_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    int _mma_b_lo_8 = make_warp_uniform((((smem_v6_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_8) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_8) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base + 128)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfa + (sfa_col + 8) + 0, tmem_tmem_sfb + (sfb_col + 32 + 8 + (b_rows_1 + 128) % 128 / 32) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_9 = make_warp_uniform((((smem_v0_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    int _mma_b_lo_9 = make_warp_uniform((((smem_v2_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_9) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_9) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfa + (sfa_col + 12) + 0, tmem_tmem_sfb + (sfb_col + 12 + b_rows_1 % 128 / 32) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_10 = make_warp_uniform((((smem_v0_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    int _mma_b_lo_10 = make_warp_uniform((((smem_v4_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_10) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_10) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base + 64)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfa + (sfa_col + 12) + 0, tmem_tmem_sfb + (sfb_col + 16 + 12 + (b_rows_1 + 64) % 128 / 32) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_11 = make_warp_uniform((((smem_v0_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    int _mma_b_lo_11 = make_warp_uniform((((smem_v6_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 3072);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_11) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_11) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_acc + (acc_base + 128)), a_desc + 0, b_desc + 0,
                                0x8100480U, tmem_tmem_sfa + (sfa_col + 12) + 0, tmem_tmem_sfb + (sfb_col + 32 + 12 + (b_rows_1 + 128) % 128 / 32) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    elect_commit(tma_empty_addr + (mma_stage) * 8);
                    mma_stage += 1;
                    if (mma_stage == 4) { mma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(acc_full_addr + (acc_stage) * 8);
                _phase_acc_empty ^= 1;
                completed += 1;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            if (completed > 0) {
                mbarrier_wait(acc_empty_addr, completed - 1 & 1);
            }
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: prefetch ----
    if (warp == 2) {
        // idle — no tasks assigned
    }
    // ---- Role: idle ----
    if (warp == 3) {
        // idle — no tasks assigned
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp - 4;
            int epi_row = epi_warp * 32 + lane;
            unsigned int acc_stage_1 = 0;
            unsigned int store_stage = 0;
            int tiles_end_2 = num_tiles;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int tile_2 = bid; tile_2 < tiles_end_2; tile_2 += num_bids) {
                int tile_c_2 = (int)tile_2;
                int w_idx_2 = tile_c_2 / tok_tiles;
                int tok_idx_2 = tile_c_2 - w_idx_2 * tok_tiles;
                int off_tok = tok_idx_2 * 128;
                int off_w = w_idx_2 * 192;
                int acc_base_1 = (int)acc_stage_1 * 192;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_base_1;
                float alpha_row = 0.0f;
                float alphas[192];
                float frag_alpha[2];
                int n_col = off_w + epi_row;
                long long row_ptr = (long long)off_tok * (long long)N + (long long)n_col;
                long long n_stride = N;
                int _min_0 = ((192) < (M - off_tok) ? (192) : (M - off_tok));
                int n_rows = _min_0;
                int _min_1 = ((off_tok + epi_row) < (M - 1) ? (off_tok + epi_row) : (M - 1));
                int tok_row = _min_1;
                alpha_row = alpha[tok_row];
                mbarrier_wait(acc_full_addr + (acc_stage_1) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int subtile = 0; subtile < 6; subtile++) {
                    int tmem_addr = lane_addr + subtile * 32;
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(tmem_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (subtile == 5) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(acc_empty_addr + (acc_stage_1) * 8);
                    }
                    {
                        float2 _pair_scale_even2_0 = make_float2(alpha_row, alpha_row);
                        float2 _pair_scale_odd2_0 = make_float2(alpha_row, alpha_row);
                        float2* _pair_scale_src2_0 = reinterpret_cast<float2*>(&_tmem_load_0[0]);
                        #if __CUDA_ARCH__ >= 1000
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                        asm volatile("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        #else
                        _tmem_load_0[0] *= alpha_row;
                        _tmem_load_0[1] *= alpha_row;
                        _tmem_load_0[2] *= alpha_row;
                        _tmem_load_0[3] *= alpha_row;
                        _tmem_load_0[4] *= alpha_row;
                        _tmem_load_0[5] *= alpha_row;
                        _tmem_load_0[6] *= alpha_row;
                        _tmem_load_0[7] *= alpha_row;
                        _tmem_load_0[8] *= alpha_row;
                        _tmem_load_0[9] *= alpha_row;
                        _tmem_load_0[10] *= alpha_row;
                        _tmem_load_0[11] *= alpha_row;
                        _tmem_load_0[12] *= alpha_row;
                        _tmem_load_0[13] *= alpha_row;
                        _tmem_load_0[14] *= alpha_row;
                        _tmem_load_0[15] *= alpha_row;
                        _tmem_load_0[16] *= alpha_row;
                        _tmem_load_0[17] *= alpha_row;
                        _tmem_load_0[18] *= alpha_row;
                        _tmem_load_0[19] *= alpha_row;
                        _tmem_load_0[20] *= alpha_row;
                        _tmem_load_0[21] *= alpha_row;
                        _tmem_load_0[22] *= alpha_row;
                        _tmem_load_0[23] *= alpha_row;
                        _tmem_load_0[24] *= alpha_row;
                        _tmem_load_0[25] *= alpha_row;
                        _tmem_load_0[26] *= alpha_row;
                        _tmem_load_0[27] *= alpha_row;
                        _tmem_load_0[28] *= alpha_row;
                        _tmem_load_0[29] *= alpha_row;
                        _tmem_load_0[30] *= alpha_row;
                        _tmem_load_0[31] *= alpha_row;
                        #endif
                    }
                    uint32_t _tmem_load_0_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (subtile > 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                if ((unsigned int)tile_c_2 == tile_2) {
                                    tma_store_2d((&out), off_w + (subtile - 1) * 32, off_tok, smem_out_addr + store_stage * 8192);
                                }
                            }
                        }
                        if (warp == 4) {
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 3;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        store_stage = store_stage + 1;
                        if (store_stage == 4) {
                            store_stage = 0;
                        }
                    }
                    int out_stage_row = store_stage * 128 + (unsigned int)epi_row;
                    unsigned int out_abs = smem_out_addr + (unsigned int)(out_stage_row * 64);
                    unsigned int out_swz = out_abs / 8 & 48;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (0 ^ out_swz)))), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (16 ^ out_swz)))), "r"(_tmem_load_0_bf16[4]), "r"(_tmem_load_0_bf16[5]), "r"(_tmem_load_0_bf16[6]), "r"(_tmem_load_0_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (32 ^ out_swz)))), "r"(_tmem_load_0_bf16[8]), "r"(_tmem_load_0_bf16[9]), "r"(_tmem_load_0_bf16[10]), "r"(_tmem_load_0_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (48 ^ out_swz)))), "r"(_tmem_load_0_bf16[12]), "r"(_tmem_load_0_bf16[13]), "r"(_tmem_load_0_bf16[14]), "r"(_tmem_load_0_bf16[15]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        if ((unsigned int)tile_c_2 == tile_2) {
                            tma_store_2d((&out), off_w + 160, off_tok, smem_out_addr + store_stage * 8192);
                        }
                    }
                }
                if (warp == 4) {
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 3;");
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                store_stage = store_stage + 1;
                if (store_stage == 4) {
                    store_stage = 0;
                }
                _phase_acc_full ^= 1;
            }
            if (warp == 4) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    // Cleanup
}

} // extern "C"
