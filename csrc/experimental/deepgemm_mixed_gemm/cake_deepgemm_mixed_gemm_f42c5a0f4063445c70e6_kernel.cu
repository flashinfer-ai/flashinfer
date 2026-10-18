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
 *
 * Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek, licensed under
 * the MIT License; see DEEPGEMM_NOTICE.txt in this directory.
 */

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_deepgemm_mixed_gemm_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 40
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SCALE_A_OFFSET 32
#define TMEM_SCALE_B_OFFSET 36
#define NUM_PIPE_STAGES 12
#define NUM_EPI_PIPE_STAGES 2
#define SMEM_EPI_OFF 0
#define SMEM_EPI_STAGE_BYTES 4096
#define SMEM_EPI_STRIDE 4096
#define SMEM_SMEM_A_OFF 8192
#define SMEM_SMEM_A_STAGE_BYTES 1024
#define SMEM_SMEM_A_STRIDE 1024
#define SMEM_SMEM_B_OFF 20480
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_SFA_OFF 217088
#define SMEM_SMEM_SFA_STAGE_BYTES 512
#define SMEM_SMEM_SFA_STRIDE 512
#define SMEM_SMEM_SFB_OFF 223232
#define SMEM_SMEM_SFB_STAGE_BYTES 512
#define SMEM_SMEM_SFB_STRIDE 512
#define SMEM_TICK_SMEM_OFF 229376
#define SMEM_TICK_SMEM_STAGE_BYTES 16
#define SMEM_TICK_SMEM_STRIDE 16
#define SMEM_TOTAL 229760
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_deepgemm_mixed_gemm_f42c5a0f4063445c70e6(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C_tma, int M, int N, int K, int grid_m, int grid_n, int K_tiles, CakeFastDivmod SCHED_DIV, CakeFastDivmod TAIL_DIV, float* __restrict__ WS, unsigned int* __restrict__ CNT)
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

    const int mbar_base = smem + 229392;
    #define full_addr (mbar_base + 0)
    #define sf_full_addr (mbar_base + 96)
    #define empty_addr (mbar_base + 192)
    #define tmem_full_addr (mbar_base + 288)
    #define tmem_empty_addr (mbar_base + 304)
    #define tmem_overlap_addr (mbar_base + 320)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* epi = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_EPI_OFF);
    const int epi_addr = smem + SMEM_EPI_OFF;
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    unsigned int* smem_sfa = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_SFA_OFF);
    const int smem_sfa_addr = smem + SMEM_SMEM_SFA_OFF;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_SFB_OFF);
    const int smem_sfb_addr = smem + SMEM_SMEM_SFB_OFF;
    unsigned int* tick_smem = reinterpret_cast<unsigned int*>(smem_raw + SMEM_TICK_SMEM_OFF);
    const int tick_smem_addr = smem + SMEM_TICK_SMEM_OFF;
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&C_tma))) : "memory"); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 42 barriers)
    // Mbarriers at smem_raw[229392..229728)

    if (warp == 1) {
        // --- pipeline 'pipe' ---
        // full: 12 barriers, init_count=66
        // sf_full: 12 barriers, init_count=1
        // empty: 12 barriers, init_count=1
        // --- pipeline 'epi_pipe' ---
        // tmem_full: 2 barriers, init_count=1
        // tmem_empty: 2 barriers, init_count=256
        // tmem_overlap: 2 barriers, init_count=256
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_1_0 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_1_0) : "r"(lane), "n"(12), "r"((uint32_t)(66)));
        mbarrier_init(smem + 229392 + lane * 8, _mbarrier_init_count_1_0);
        uint32_t _mbarrier_init_count_1_32 = 256;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_1_32) : "r"(lane), "n"(6), "r"((uint32_t)(1)));
        if (lane < 10) {
            mbarrier_init(smem + 229648 + lane * 8, _mbarrier_init_count_1_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (64 columns, 40 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 229728);
    if (warp == 2) {
        int _tmem_hold = smem + 229728;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        __syncwarp();
    }

    // Partial post-allocation TMEM rendezvous (224 threads on named barrier 9)
    if (warp >= 1 && warp <= 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(9), "r"(224) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }

    const int taddr = (warp >= 1 && warp <= 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_scale_a = taddr + 32;
    const int tmem_scale_b = taddr + 36;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int stage = 0;
            unsigned int _phase_empty = 1;
            #pragma unroll 1
            for (unsigned int bid_1 = bid; bid_1 < grid_m * grid_n * 3; bid_1 += num_bids) {
                unsigned int T = grid_m * grid_n;
                unsigned int k_slice = bid_1 / T;
                unsigned int tile = bid_1 - k_slice * T;
                uint32_t _fast_div_q_0 = (SCHED_DIV.divisor != 1) ? (__umulhi((uint32_t)(tile), SCHED_DIV.multiplier) >> SCHED_DIV.shift_right) : (uint32_t)(tile);
                uint32_t _fast_div_r_0 = (uint32_t)(tile) - _fast_div_q_0 * (uint32_t)(SCHED_DIV.divisor);
                unsigned int first_n = _fast_div_q_0 * 8;
                uint32_t _fast_div_q_1 = (TAIL_DIV.divisor != 1) ? (__umulhi((uint32_t)(_fast_div_r_0), TAIL_DIV.multiplier) >> TAIL_DIV.shift_right) : (uint32_t)(_fast_div_r_0);
                uint32_t _fast_div_r_1 = (uint32_t)(_fast_div_r_0) - _fast_div_q_1 * (uint32_t)(TAIL_DIV.divisor);
                unsigned int m_tile = (((unsigned int)grid_n - first_n < 8) ? _fast_div_q_1 : _fast_div_r_0 / 8);
                unsigned int n_tile = (((unsigned int)grid_n - first_n < 8) ? _fast_div_r_1 : _fast_div_r_0 % 8);
                unsigned int off_m = m_tile * 16;
                unsigned int off_n = (first_n + n_tile) * 128;
                int k_lo = k_slice * (unsigned int)K_tiles / 3;
                int k_hi = (k_slice + 1) * (unsigned int)K_tiles / 3;
                #pragma unroll 1
                for (int k = k_lo; k < k_hi; k++) {
                    mbarrier_wait(empty_addr + (stage) * 8, _phase_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem(smem_sfa_addr + stage * 512, (&SFA), off_m, k, sf_full_addr + (stage) * 8);
                        tma_2d_gmem2smem(smem_sfb_addr + stage * 512, (&SFB), off_n, k, sf_full_addr + (stage) * 8);
                        mbarrier_arrive_expect_tx(sf_full_addr + (stage) * 8, 1024);
                        tma_2d_gmem2smem_cta2(smem_a_addr + stage * 1024, (&A), k * 128, off_m + (unsigned int)(cta_rank * 8), ((full_addr + (stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_b_addr + stage * 16384, (&B), k * 128, off_n, ((full_addr + (stage) * 8) & 0xFEFFFFFF));
                        if (cta_rank == 0) {
                            mbarrier_arrive_expect_tx(full_addr + (stage) * 8, 18432);
                        } else {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 12) { stage = 0; _phase_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int stage_1 = 0;
            unsigned int accum_stage = 0;
            unsigned int completed = 0;
            unsigned int _phase_tmem_empty = 1;
            unsigned int _phase_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int bid_2 = bid; bid_2 < grid_m * grid_n * 3; bid_2 += num_bids) {
                    unsigned int T_1 = grid_m * grid_n;
                    unsigned int k_slice_1 = bid_2 / T_1;
                    unsigned int tile_1 = bid_2 - k_slice_1 * T_1;
                    uint32_t _fast_div_q_4 = (SCHED_DIV.divisor != 1) ? (__umulhi((uint32_t)(tile_1), SCHED_DIV.multiplier) >> SCHED_DIV.shift_right) : (uint32_t)(tile_1);
                    uint32_t _fast_div_r_4 = (uint32_t)(tile_1) - _fast_div_q_4 * (uint32_t)(SCHED_DIV.divisor);
                    unsigned int first_n_1 = _fast_div_q_4 * 8;
                    uint32_t _fast_div_q_5 = (TAIL_DIV.divisor != 1) ? (__umulhi((uint32_t)(_fast_div_r_4), TAIL_DIV.multiplier) >> TAIL_DIV.shift_right) : (uint32_t)(_fast_div_r_4);
                    uint32_t _fast_div_r_5 = (uint32_t)(_fast_div_r_4) - _fast_div_q_5 * (uint32_t)(TAIL_DIV.divisor);
                    unsigned int m_tile_1 = (((unsigned int)grid_n - first_n_1 < 8) ? _fast_div_q_5 : _fast_div_r_4 / 8);
                    unsigned int n_tile_1 = (((unsigned int)grid_n - first_n_1 < 8) ? _fast_div_r_5 : _fast_div_r_4 % 8);
                    unsigned int off_m_1 = m_tile_1 * 16;
                    unsigned int off_n_1 = (first_n_1 + n_tile_1) * 128;
                    int k_lo_1 = k_slice_1 * (unsigned int)K_tiles / 3;
                    int k_hi_1 = (k_slice_1 + 1) * (unsigned int)K_tiles / 3;
                    mbarrier_wait(tmem_empty_addr + (accum_stage) * 8, _phase_tmem_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 4
                    for (int k_1 = k_lo_1; k_1 < k_hi_1; k_1++) {
                        mbarrier_wait(full_addr + (stage_1) * 8, _phase_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_scale_a, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (stage_1) * 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_scale_b, make_sf_cp_desc_lo_sbo128((((smem_sfb_addr) >> 4) + (stage_1) * 32)));
                        }
                        __syncwarp();
                        int init_flag = ((k_1 == k_lo_1) ? 1 : 0);
                        if (elect_sync()) {
                            int _mma_a_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                            int _mma_b_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_1) * 64;
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (accum_stage * 16)), a_desc + 0, b_desc + 0,
                                    (0x10840280U | ((0) << 29) | ((0) << 4)), tmem_scale_b, tmem_scale_a, ((init_flag) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (accum_stage * 16)), a_desc + 2, b_desc + 2,
                                    (0x10840280U | ((1) << 29) | ((1) << 4)), tmem_scale_b, tmem_scale_a, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (accum_stage * 16)), a_desc + 4, b_desc + 4,
                                    (0x10840280U | ((2) << 29) | ((2) << 4)), tmem_scale_b, tmem_scale_a, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (accum_stage * 16)), a_desc + 6, b_desc + 6,
                                    (0x10840280U | ((3) << 29) | ((3) << 4)), tmem_scale_b, tmem_scale_a, 1);
                            }
                        }
                        __syncwarp();
                        elect_commit_cg2_multicast(empty_addr + (stage_1) * 8, (uint16_t)(3));
                        if (k_1 == k_hi_1 - 1) {
                            elect_commit_cg2_multicast(tmem_full_addr + (accum_stage) * 8, (uint16_t)(3));
                        }
                        __syncwarp();
                        stage_1 += 1;
                        if (stage_1 == 12) { stage_1 = 0; _phase_full ^= 1; }
                    }
                    accum_stage += 1;
                    if (accum_stage == 2) { accum_stage = 0; _phase_tmem_empty ^= 1; }
                    completed = completed + 1;
                }
                if (completed > 0) {
                    mbarrier_wait(tmem_empty_addr + ((completed - 1) % 2) * 8, (completed - 1) / 2 % 2);
                }
            }
        }
    }
    // ---- Role: transpose ----
    if (warp == 2) {
        { // transpose_main
            unsigned int stage_2 = 0;
            unsigned int _phase_sf_full = 0;
            #pragma unroll 1
            for (unsigned int bid_3 = bid; bid_3 < grid_m * grid_n * 3; bid_3 += num_bids) {
                unsigned int T_2 = grid_m * grid_n;
                unsigned int k_slice_2 = bid_3 / T_2;
                unsigned int tile_2 = bid_3 - k_slice_2 * T_2;
                uint32_t _fast_div_q_2 = (SCHED_DIV.divisor != 1) ? (__umulhi((uint32_t)(tile_2), SCHED_DIV.multiplier) >> SCHED_DIV.shift_right) : (uint32_t)(tile_2);
                uint32_t _fast_div_r_2 = (uint32_t)(tile_2) - _fast_div_q_2 * (uint32_t)(SCHED_DIV.divisor);
                unsigned int first_n_2 = _fast_div_q_2 * 8;
                uint32_t _fast_div_q_3 = (TAIL_DIV.divisor != 1) ? (__umulhi((uint32_t)(_fast_div_r_2), TAIL_DIV.multiplier) >> TAIL_DIV.shift_right) : (uint32_t)(_fast_div_r_2);
                uint32_t _fast_div_r_3 = (uint32_t)(_fast_div_r_2) - _fast_div_q_3 * (uint32_t)(TAIL_DIV.divisor);
                unsigned int m_tile_2 = (((unsigned int)grid_n - first_n_2 < 8) ? _fast_div_q_3 : _fast_div_r_2 / 8);
                unsigned int n_tile_2 = (((unsigned int)grid_n - first_n_2 < 8) ? _fast_div_r_3 : _fast_div_r_2 % 8);
                unsigned int off_m_2 = m_tile_2 * 16;
                unsigned int off_n_2 = (first_n_2 + n_tile_2) * 128;
                int k_lo_2 = k_slice_2 * (unsigned int)K_tiles / 3;
                int k_hi_2 = (k_slice_2 + 1) * (unsigned int)K_tiles / 3;
                #pragma unroll 1
                for (int k_2 = k_lo_2; k_2 < k_hi_2; k_2++) {
                    mbarrier_wait(sf_full_addr + (stage_2) * 8, _phase_sf_full);
                    #pragma unroll
                    for (int sf_block = 0; sf_block < 1; sf_block++) {
                        unsigned int words[4];
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[i])) : "r"(smem_sfa_addr + stage_2 * 512 + (unsigned int)(sf_block * 512) + (unsigned int)((i * 32 + lane) * 4)));
                        }
                        __syncwarp();
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                            "r"(smem_sfa_addr + stage_2 * 512 + (unsigned int)(sf_block * 512) + (unsigned int)(lane * 16)), "r"(*reinterpret_cast<uint32_t*>(&words[0])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3])));
                    }
                    unsigned int words_1[4];
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 4; i_1++) {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words_1[i_1])) : "r"(smem_sfb_addr + stage_2 * 512 + (unsigned int)((i_1 * 32 + lane) * 4)));
                    }
                    __syncwarp();
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(smem_sfb_addr + stage_2 * 512 + (unsigned int)(lane * 16)), "r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3])));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((full_addr + (stage_2) * 8) & 0xFEFFFFFF) : "memory");
                    stage_2 += 1;
                    if (stage_2 == 12) { stage_2 = 0; _phase_sf_full ^= 1; }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            unsigned int accum_stage_1 = 0;
            unsigned int store_stage = 0;
            unsigned int warp_0 = warp - 4;
            unsigned int epi_tid = warp_0 * 32 + (unsigned int)lane;
            unsigned int item = 0;
            unsigned int _phase_tmem_full = 0;
            #pragma unroll 1
            for (unsigned int bid_4 = bid; bid_4 < grid_m * grid_n * 3; bid_4 += num_bids) {
                unsigned int T_3 = grid_m * grid_n;
                unsigned int k_slice_3 = bid_4 / T_3;
                unsigned int tile_3 = bid_4 - k_slice_3 * T_3;
                uint32_t _fast_div_q_6 = (SCHED_DIV.divisor != 1) ? (__umulhi((uint32_t)(tile_3), SCHED_DIV.multiplier) >> SCHED_DIV.shift_right) : (uint32_t)(tile_3);
                uint32_t _fast_div_r_6 = (uint32_t)(tile_3) - _fast_div_q_6 * (uint32_t)(SCHED_DIV.divisor);
                unsigned int first_n_3 = _fast_div_q_6 * 8;
                uint32_t _fast_div_q_7 = (TAIL_DIV.divisor != 1) ? (__umulhi((uint32_t)(_fast_div_r_6), TAIL_DIV.multiplier) >> TAIL_DIV.shift_right) : (uint32_t)(_fast_div_r_6);
                uint32_t _fast_div_r_7 = (uint32_t)(_fast_div_r_6) - _fast_div_q_7 * (uint32_t)(TAIL_DIV.divisor);
                unsigned int m_tile_3 = (((unsigned int)grid_n - first_n_3 < 8) ? _fast_div_q_7 : _fast_div_r_6 / 8);
                unsigned int n_tile_3 = (((unsigned int)grid_n - first_n_3 < 8) ? _fast_div_r_7 : _fast_div_r_6 % 8);
                unsigned int off_m_3 = m_tile_3 * 16;
                unsigned int off_n_3 = (first_n_3 + n_tile_3) * 128;
                int k_lo_3 = k_slice_3 * (unsigned int)K_tiles / 3;
                int k_hi_3 = (k_slice_3 + 1) * (unsigned int)K_tiles / 3;
                unsigned int own = bid_4 * 2048;
                unsigned int tile_base = tile_3 * 2048;
                unsigned int slice_stride = grid_m * grid_n * 2048;
                mbarrier_wait(tmem_full_addr + (accum_stage_1) * 8, _phase_tmem_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int s = 0; s < 1; s++) {
                    #pragma unroll
                    for (int i_2 = 0; i_2 < 2; i_2++) {
                        unsigned int address = taddr + accum_stage_1 * 16 + (unsigned int)(s * 16) + (unsigned int)(i_2 * 8);
                        float _tmem_load_0[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                            : "r"(address));
                        float _tmem_load_1[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                            : "r"(address | 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (s == 0 && i_2 == 1) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((tmem_empty_addr + (accum_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        unsigned int quad = (s * 2 + i_2) * 2;
                        {
                            float4 _v4 = make_float4(_tmem_load_0[0 + 0], _tmem_load_0[0 + 1], _tmem_load_0[0 + 2], _tmem_load_0[0 + 3]);
                            *reinterpret_cast<float4*>((WS + (own + (quad * 128 + epi_tid) * 4)) + 0) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(_tmem_load_1[0 + 0], _tmem_load_1[0 + 1], _tmem_load_1[0 + 2], _tmem_load_1[0 + 3]);
                            *reinterpret_cast<float4*>((WS + (own + ((quad + 1) * 128 + epi_tid) * 4)) + 0) = _v4;
                        }
                    }
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        unsigned int _atomic_old_0;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(_atomic_old_0) : "l"(&CNT[tile_3]), "r"(static_cast<uint32_t>(1)) : "memory");
                        tick_smem[0] = _atomic_old_0;
                    }
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                unsigned int my_ticket = tick_smem[0];
                if (my_ticket == 2) {
                    if (warp == 4) {
                        if (elect_sync()) {
                            asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(CNT) + (tile_3))), "r"(static_cast<unsigned int>(0)) : "memory");
                        }
                    }
                    #pragma unroll
                    for (int so = 0; so < 1; so++) {
                        if (warp_0 == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                        asm volatile("barrier.sync 0, 128;" ::: "memory");
                        unsigned int stage_base = epi_addr + store_stage * 4096;
                        #pragma unroll
                        for (int io = 0; io < 2; io++) {
                            unsigned int quad_o = (so * 2 + io) * 2;
                            unsigned int lane_a = tile_base + (quad_o * 128 + epi_tid) * 4;
                            unsigned int lane_b = tile_base + ((quad_o + 1) * 128 + epi_tid) * 4;
                            float acc_a[4];
                            float acc_b[4];
                            float part_a[4];
                            float part_b[4];
                            {
                                unsigned _v4_0_0;
                                unsigned _v4_0_1;
                                unsigned _v4_0_2;
                                unsigned _v4_0_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_0_0), "=r"(_v4_0_1), "=r"(_v4_0_2), "=r"(_v4_0_3) : "l"((const void*)(WS + lane_a + (0))) : "memory");
                                acc_a[0 + 0] = __uint_as_float(_v4_0_0);
                                acc_a[0 + 1] = __uint_as_float(_v4_0_1);
                                acc_a[0 + 2] = __uint_as_float(_v4_0_2);
                                acc_a[0 + 3] = __uint_as_float(_v4_0_3);
                            }
                            {
                                unsigned _v4_1_0;
                                unsigned _v4_1_1;
                                unsigned _v4_1_2;
                                unsigned _v4_1_3;
                                asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                    : "=r"(_v4_1_0), "=r"(_v4_1_1), "=r"(_v4_1_2), "=r"(_v4_1_3) : "l"((const void*)(WS + lane_b + (0))) : "memory");
                                acc_b[0 + 0] = __uint_as_float(_v4_1_0);
                                acc_b[0 + 1] = __uint_as_float(_v4_1_1);
                                acc_b[0 + 2] = __uint_as_float(_v4_1_2);
                                acc_b[0 + 3] = __uint_as_float(_v4_1_3);
                            }
                            #pragma unroll
                            for (int j = 1; j < 3; j++) {
                                {
                                    unsigned _v4_2_0;
                                    unsigned _v4_2_1;
                                    unsigned _v4_2_2;
                                    unsigned _v4_2_3;
                                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                        : "=r"(_v4_2_0), "=r"(_v4_2_1), "=r"(_v4_2_2), "=r"(_v4_2_3) : "l"((const void*)(WS + (lane_a + (unsigned int)j * slice_stride) + (0))) : "memory");
                                    part_a[0 + 0] = __uint_as_float(_v4_2_0);
                                    part_a[0 + 1] = __uint_as_float(_v4_2_1);
                                    part_a[0 + 2] = __uint_as_float(_v4_2_2);
                                    part_a[0 + 3] = __uint_as_float(_v4_2_3);
                                }
                                {
                                    unsigned _v4_3_0;
                                    unsigned _v4_3_1;
                                    unsigned _v4_3_2;
                                    unsigned _v4_3_3;
                                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                        : "=r"(_v4_3_0), "=r"(_v4_3_1), "=r"(_v4_3_2), "=r"(_v4_3_3) : "l"((const void*)(WS + (lane_b + (unsigned int)j * slice_stride) + (0))) : "memory");
                                    part_b[0 + 0] = __uint_as_float(_v4_3_0);
                                    part_b[0 + 1] = __uint_as_float(_v4_3_1);
                                    part_b[0 + 2] = __uint_as_float(_v4_3_2);
                                    part_b[0 + 3] = __uint_as_float(_v4_3_3);
                                }
                                #pragma unroll
                                for (int v = 0; v < 4; v++) {
                                    acc_a[v] = acc_a[v] + part_a[v];
                                    acc_b[v] = acc_b[v] + part_b[v];
                                }
                            }
                            uint32_t acc_a_bf16[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_a[_lp*2 + 0], acc_a[_lp*2+1 + 0]));
                                acc_a_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t acc_b_bf16[2];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_b[_lp*2 + 0], acc_b[_lp*2+1 + 0]));
                                acc_b_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            unsigned int row = lane % 8;
                            unsigned int col = warp_0 % 2 * 4 + (unsigned int)(lane / 8);
                            unsigned int write_addr = stage_base + warp_0 / 2 * 16 * 128 + (unsigned int)(io * 8 * 128) + row * 128 + (col ^ row) * 16;
                            uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(write_addr);
                            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&acc_a_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&acc_a_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&acc_b_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&acc_b_bf16[1]))
                                : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 0, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                tma_store_2d((&C_tma), off_n_3, off_m_3 + (unsigned int)(so * 16), stage_base);
                                tma_store_2d((&C_tma), off_n_3 + 64, off_m_3 + (unsigned int)(so * 16), stage_base + 2048);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        __syncwarp();
                        store_stage = (store_stage + 1) % 2;
                    }
                }
                accum_stage_1 += 1;
                if (accum_stage_1 == 2) { accum_stage_1 = 0; _phase_tmem_full ^= 1; }
                item = item + 1;
            }
        }
    }

    // Kernel teardown ops
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(64));
    }

    // Cleanup
}

} // extern "C"
