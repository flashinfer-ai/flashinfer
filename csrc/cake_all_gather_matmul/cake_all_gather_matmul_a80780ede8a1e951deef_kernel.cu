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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_all_gather_matmul_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_TMA_PIPE_STAGES 4
#define NUM_MAINLOOP_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 49152
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 49152
#define SMEM_TOTAL 197632
#define THREADS 192

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_all_gather_matmul_a80780ede8a1e951deef(const __grid_constant__ CUtensorMap A_local, const __grid_constant__ CUtensorMap A_scratch, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ scratch_payload, unsigned int* __restrict__ ready, unsigned int ready_target, int rank, int M, int scratch_pitch, int signal_rows, int n_tiles, int remote_order)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 12 barriers)
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
            // mainloop_done: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // epilogue_done: 2 barriers, init_count=4
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 96);
    if (warp == 0) {
        int _tmem_hold = smem + 96;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_local))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_scratch))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
            unsigned int load_stage = 0;
            unsigned int load_phase = 1;
            #pragma unroll 1
            for (int tile = bid; tile < 4 * ((M + 127) / 128 * 128 / 128 * n_tiles); tile += num_bids) {
                int peer_pass_pm = tile / ((M + 127) / 128 * 128 / 128 * n_tiles);
                int rem_pm = tile - peer_pass_pm * ((M + 127) / 128 * 128 / 128 * n_tiles);
                int c_pm_raw = rem_pm / ((((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128 * n_tiles);
                int c_pm = ((c_pm_raw < ((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) ? c_pm_raw : ((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1);
                int inner_pm = rem_pm - c_pm * ((((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128 * n_tiles);
                int u_cm = tile - (M + 127) / 128 * 128 / 128 * n_tiles;
                int c_cm_raw = u_cm / (3 * ((((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128 * n_tiles));
                int c_cm = ((c_cm_raw < ((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) ? c_cm_raw : ((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1);
                int v_cm = u_cm - c_cm * (3 * ((((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128 * n_tiles));
                int tiles_m_cm = ((c_cm < ((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) ? (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128 : (M + 127) / 128 * 128 / 128 - (((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) * ((((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128));
                int tiles_c_cm = tiles_m_cm * n_tiles;
                int pp_cm = 1 + v_cm / tiles_c_cm;
                int inner_cm = v_cm - (pp_cm - 1) * tiles_c_cm;
                int local_tile = ((tile < (M + 127) / 128 * 128 / 128 * n_tiles) ? 1 : 0);
                int use_pm = ((remote_order == 0) ? 1 : local_tile);
                int peer_pass = ((use_pm == 1) ? peer_pass_pm : pp_cm);
                int chunk_idx = ((use_pm == 1) ? c_pm : c_cm);
                int inner = ((use_pm == 1) ? inner_pm : inner_cm);
                int tiles_m_c = ((chunk_idx < ((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) ? (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128 : (M + 127) / 128 * 128 / 128 - (((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) * ((((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128));
                int bid_m = inner % tiles_m_c;
                int bid_n = inner / tiles_m_c;
                int peer = (rank - peer_pass + 4) % 4;
                int off_m = chunk_idx * (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) + bid_m * 128;
                int off_n = bid_n * 256;
                if (peer_pass != 0) {
                    if (elect_sync()) {
                        {
                            unsigned int* _gca_p = reinterpret_cast<unsigned int*>(ready) + (peer * (((M + 127) / 128 * 128 + signal_rows - 1) / signal_rows) + off_m / signal_rows);
                            while (true) {
                                unsigned int _gca_v;
                                asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(_gca_v) : "l"(_gca_p));
                                if (_gca_v >= (unsigned int)(ready_target)) break;
                            }
                        }
                    }
                }
                #pragma unroll 1
                for (int iter_k = 0; iter_k < 128; iter_k++) {
                    uint32_t _mbar_token_0 = mbarrier_try_wait(mma_done_addr + (load_stage) * 8, load_phase);
                    mbarrier_wait_token(mma_done_addr + (load_stage) * 8, load_phase, _mbar_token_0);
                    int off_k = iter_k * 64;
                    if (elect_sync()) {
                        if (peer_pass == 0) {
                            tma_3d_gmem2smem(smem_a_addr + load_stage * 49152, (&A_local), 0, off_m, iter_k, tma_full_addr + (load_stage) * 8);
                        } else {
                            tma_3d_gmem2smem(smem_a_addr + load_stage * 49152, (&A_scratch), 0, peer * scratch_pitch + off_m, iter_k, tma_full_addr + (load_stage) * 8);
                        }
                        #pragma unroll
                        for (int b_panel = 0; b_panel < 4; b_panel++) {
                            tma_3d_gmem2smem(smem_b_addr + load_stage * 49152 + (unsigned int)(b_panel * 8192), (&B), off_n + b_panel * 64, off_k, 0, tma_full_addr + (load_stage) * 8);
                        }
                        mbarrier_arrive_expect_tx(tma_full_addr + (load_stage) * 8, 49152);
                    }
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; }
                    if (load_stage == 0) {
                        load_phase = load_phase ^ 1;
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
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            #pragma unroll 1
            for (int tile_1 = bid; tile_1 < 4 * ((M + 127) / 128 * 128 / 128 * n_tiles); tile_1 += num_bids) {
                mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < 128; iter_k_1++) {
                    mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_flag = ((iter_k_1 == 0) ? 1 : 0);
                    int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3072);
                    int _mma_b_lo_0 = make_warp_uniform(((((smem_b_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 256)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 138478736, ((init_flag) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 256)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 138478736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 256)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 138478736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_accum + (mma_epi_stage * 256)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 138478736, 1);
                        }
                    }
                    elect_commit(mma_done_addr + (mma_tma_stage) * 8);
                    mma_tma_stage += 1;
                    if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                }
                elect_commit(mainloop_done_addr + (mma_epi_stage) * 8);
                mma_epi_stage += 1;
                if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (int tile_2 = bid; tile_2 < 4 * ((M + 127) / 128 * 128 / 128 * n_tiles); tile_2 += num_bids) {
                int peer_pass_pm_1 = tile_2 / ((M + 127) / 128 * 128 / 128 * n_tiles);
                int rem_pm_1 = tile_2 - peer_pass_pm_1 * ((M + 127) / 128 * 128 / 128 * n_tiles);
                int c_pm_raw_1 = rem_pm_1 / ((((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128 * n_tiles);
                int c_pm_1 = ((c_pm_raw_1 < ((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) ? c_pm_raw_1 : ((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1);
                int inner_pm_1 = rem_pm_1 - c_pm_1 * ((((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128 * n_tiles);
                int u_cm_1 = tile_2 - (M + 127) / 128 * 128 / 128 * n_tiles;
                int c_cm_raw_1 = u_cm_1 / (3 * ((((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128 * n_tiles));
                int c_cm_1 = ((c_cm_raw_1 < ((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) ? c_cm_raw_1 : ((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1);
                int v_cm_1 = u_cm_1 - c_cm_1 * (3 * ((((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128 * n_tiles));
                int tiles_m_cm_1 = ((c_cm_1 < ((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) ? (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128 : (M + 127) / 128 * 128 / 128 - (((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) * ((((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128));
                int tiles_c_cm_1 = tiles_m_cm_1 * n_tiles;
                int pp_cm_1 = 1 + v_cm_1 / tiles_c_cm_1;
                int inner_cm_1 = v_cm_1 - (pp_cm_1 - 1) * tiles_c_cm_1;
                int local_tile_1 = ((tile_2 < (M + 127) / 128 * 128 / 128 * n_tiles) ? 1 : 0);
                int use_pm_1 = ((remote_order == 0) ? 1 : local_tile_1);
                int peer_pass_1 = ((use_pm_1 == 1) ? peer_pass_pm_1 : pp_cm_1);
                int chunk_idx_1 = ((use_pm_1 == 1) ? c_pm_1 : c_cm_1);
                int inner_1 = ((use_pm_1 == 1) ? inner_pm_1 : inner_cm_1);
                int tiles_m_c_1 = ((chunk_idx_1 < ((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) ? (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128 : (M + 127) / 128 * 128 / 128 - (((M + 127) / 128 * 128 + (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) / (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) - 1) * ((((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) / 128));
                int bid_m_1 = inner_1 % tiles_m_c_1;
                int bid_n_1 = inner_1 / tiles_m_c_1;
                int peer_1 = (rank - peer_pass_1 + 4) % 4;
                int off_m_1 = chunk_idx_1 * (((M + 127) / 128 * 128 < 2432) ? (M + 127) / 128 * 128 : 2432) + bid_m_1 * 128;
                int off_n_1 = bid_n_1 * 256;
                int local_row = off_m_1 + epi_tid;
                long long out_row = (long long)(peer_1 * M + local_row);
                long long out_base = out_row * (long long)(n_tiles * 256) + (long long)off_n_1;
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (off_m_1 + 128 <= M) {
                    #pragma unroll
                    for (int n_chunk = 0; n_chunk < 32; n_chunk++) {
                        int row = epi_warp * 32;
                        int col = epi_stage * 256 + (unsigned int)(n_chunk * 8);
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
                        reinterpret_cast<int4*>(C + (out_base + (long long)(n_chunk * 8)))[0] = reinterpret_cast<int4*>(_tmem_load_0_bf16)[0];
                    }
                }
                if (off_m_1 + 128 > M) {
                    #pragma unroll
                    for (int n_chunk_1 = 0; n_chunk_1 < 32; n_chunk_1++) {
                        int row_1 = epi_warp * 32;
                        int col_1 = epi_stage * 256 + (unsigned int)(n_chunk_1 * 8);
                        int tmem_addr_1 = taddr + (unsigned int)(row_1 << 16) + (unsigned int)col_1;
                        float _tmem_load_1[8];
                        tmem_ld_x8(&_tmem_load_1[0], tmem_addr_1);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        uint32_t _tmem_load_1_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                            _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        if (local_row < M) {
                            reinterpret_cast<int4*>(C + (out_base + (long long)(n_chunk_1 * 8)))[0] = reinterpret_cast<int4*>(_tmem_load_1_bf16)[0];
                        }
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

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
