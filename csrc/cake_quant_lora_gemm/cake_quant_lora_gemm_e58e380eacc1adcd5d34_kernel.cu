/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include "cake_quant_lora_gemm_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_ACC_OFFSET 0
#define NUM_TMA_PIPE_STAGES 4
#define NUM_ACC_PIPE_STAGES 1
#define NUM_CLC_PIPE_STAGES 2
#define NUM_THR_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 49152
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 49152
#define SMEM_SMEM_D_OFF 1024
#define SMEM_SMEM_D_STAGE_BYTES 16384
#define SMEM_SMEM_D_STRIDE 49152
#define SMEM_SMEM_LORA_B_OFF 33792
#define SMEM_SMEM_LORA_B_STAGE_BYTES 16384
#define SMEM_SMEM_LORA_B_STRIDE 49152
#define SMEM_SMEM_A_HI_OFF 17408
#define SMEM_SMEM_A_HI_STAGE_BYTES 16384
#define SMEM_SMEM_A_HI_STRIDE 49152
#define SMEM_SMEM_D_HI_OFF 17408
#define SMEM_SMEM_D_HI_STAGE_BYTES 16384
#define SMEM_SMEM_D_HI_STRIDE 49152
#define SMEM_SMEM_OUT_OFF 197632
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_BIAS_PAIRS_OFF 230400
#define SMEM_SMEM_BIAS_PAIRS_STAGE_BYTES 512
#define SMEM_SMEM_BIAS_PAIRS_STRIDE 512
#define SMEM_CLC_RESP_OFF 230912
#define SMEM_CLC_RESP_STAGE_BYTES 16
#define SMEM_CLC_RESP_STRIDE 16
#define SMEM_TOTAL 231040
#define THREADS 256
#define RANK_CHUNKS (((RANK + 64) - 1) / 64)
#define LAUNCH_MIN_BLOCKS 1

template <int RANK>
__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_quant_lora_gemm_e58e380eacc1adcd5d34(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap D, float* __restrict__ a_scale, float* __restrict__ b_scale, float* __restrict__ lora_scale, __nv_bfloat16* __restrict__ bias, const __grid_constant__ CUtensorMap out, int M, int N, int K_tiles, int tok_tiles, int w_tiles, int num_tiles, int raster_group, int has_bias, int has_lora, int rank, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap LORA_B)
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
    #define base_full_addr (mbar_base + 80)
    #define rescale_done_addr (mbar_base + 88)
    #define tmem_dealloc_bar_addr (mbar_base + 96)
    #define clc_full_addr (mbar_base + 104)
    #define clc_empty_addr (mbar_base + 120)
    #define thr_full_addr (mbar_base + 136)
    #define thr_empty_addr (mbar_base + 152)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_D_OFF);
    const int smem_d_addr = smem + SMEM_SMEM_D_OFF;
    __nv_bfloat16* smem_lora_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_LORA_B_OFF);
    const int smem_lora_b_addr = smem + SMEM_SMEM_LORA_B_OFF;
    uint8_t* smem_a_hi = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_A_HI_OFF);
    const int smem_a_hi_addr = smem + SMEM_SMEM_A_HI_OFF;
    __nv_bfloat16* smem_d_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_D_HI_OFF);
    const int smem_d_hi_addr = smem + SMEM_SMEM_D_HI_OFF;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_OUT_OFF);
    const int smem_out_addr = smem + SMEM_SMEM_OUT_OFF;
    unsigned int* smem_bias_pairs = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_BIAS_PAIRS_OFF);
    const int smem_bias_pairs_addr = smem + SMEM_SMEM_BIAS_PAIRS_OFF;
    unsigned int* clc_resp = reinterpret_cast<unsigned int*>(smem_raw + SMEM_CLC_RESP_OFF);
    const int clc_resp_addr = smem + SMEM_CLC_RESP_OFF;
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&D))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&LORA_B))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }

    // Mbarrier init (11 pipeline groups, 0 ordered-sequence groups, 21 barriers)
    // Mbarriers at smem_raw[0..168)

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
            // acc_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 72, 256);
            // base_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // rescale_done: 1 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            // tmem_dealloc_bar: 1 barriers, init_count=32
            mbarrier_init(smem + 96, 32);
            // --- pipeline 'clc_pipe' ---
            // clc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // clc_empty: 2 barriers, init_count=416
            mbarrier_init(smem + 120, 416);
            mbarrier_init(smem + 128, 416);
            // --- pipeline 'thr_pipe' ---
            // thr_full: 2 barriers, init_count=32
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            // thr_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 168);
    if (warp == 2) {
        int _tmem_hold = smem + 168;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    // Partial post-allocation TMEM rendezvous (192 threads on named barrier 2)
    if (warp == 1 || warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(2), "r"(192) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }

    const int taddr = (warp == 1 || warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int lora_on = has_lora;
            float s_value = 0.0f;
            if (lora_on != 0) {
                s_value = lora_scale[0];
                if (s_value == 0.0f) {
                    lora_on = 0;
                }
            }
            int lora_chunks = RANK_CHUNKS;
            if (lora_on == 0) {
                lora_chunks = 0;
            }
            int rank_offset = 0;
            unsigned int load_clc_stage = 0;
            unsigned int load_thr_stage = 0;
            unsigned int tile_l = cluster_id;
            unsigned int _phase_thr_empty = 1;
            unsigned int _phase_tma_empty = 1;
            unsigned int _phase_clc_full = 0;
            #pragma unroll 1
            for (unsigned int tile = 0; tile < num_tiles; tile++) {
                if (cta_rank == 0) {
                    mbarrier_wait(thr_empty_addr + (load_thr_stage) * 8, _phase_thr_empty);
                    mbarrier_arrive(thr_full_addr + (load_thr_stage) * 8);
                    load_thr_stage += 1;
                    if (load_thr_stage == 2) { load_thr_stage = 0; _phase_thr_empty ^= 1; }
                }
                int w_idx = tile_l / (unsigned int)tok_tiles;
                int tok_idx = tile_l - (unsigned int)(w_idx * tok_tiles);
                if (raster_group != 0) {
                    int _min_0 = ((raster_group - raster_group / 256 * 256) < ((unsigned int)tok_tiles - tile_l / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)(raster_group - raster_group / 256 * 256)) ? (raster_group - raster_group / 256 * 256) : ((unsigned int)tok_tiles - tile_l / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)(raster_group - raster_group / 256 * 256)));
                    w_idx = (tile_l - tile_l / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles)) / (unsigned int)(_min_0 * (raster_group / 256)) * (unsigned int)(raster_group / 256) + (tile_l - tile_l / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) - (tile_l - tile_l / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles)) / (unsigned int)(_min_0 * (raster_group / 256)) * (unsigned int)(_min_0 * (raster_group / 256))) / (unsigned int)_min_0;
                    int _min_1 = ((raster_group - raster_group / 256 * 256) < ((unsigned int)tok_tiles - tile_l / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)(raster_group - raster_group / 256 * 256)) ? (raster_group - raster_group / 256 * 256) : ((unsigned int)tok_tiles - tile_l / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)(raster_group - raster_group / 256 * 256)));
                    tok_idx = tile_l / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)(raster_group - raster_group / 256 * 256) + (tile_l - tile_l / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) - (tile_l - tile_l / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles)) / (unsigned int)(_min_1 * (raster_group / 256)) * (unsigned int)(_min_1 * (raster_group / 256))) - (tile_l - tile_l / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) - (tile_l - tile_l / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles)) / (unsigned int)(_min_1 * (raster_group / 256)) * (unsigned int)(_min_1 * (raster_group / 256))) / (unsigned int)_min_1 * (unsigned int)_min_1;
                }
                int a_rows = tok_idx * 512 + cta_rank * 256;
                int b_rows = w_idx * 256 + cta_rank * 128;
                #pragma unroll 1
                for (unsigned int k_tile = 0; k_tile < K_tiles; k_tile++) {
                    mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(98304)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 49152, (&A), 0, a_rows, k_tile, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_a_hi_addr + load_stage * 49152, (&A), 0, a_rows + 128, k_tile, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 49152, (&B), 0, b_rows, k_tile, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; _phase_tma_empty ^= 1; }
                }
                #pragma unroll 1
                for (unsigned int rank_chunk = 0; rank_chunk < lora_chunks; rank_chunk++) {
                    rank_offset = (int)rank_chunk * 64;
                    mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(98304)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        tma_2d_gmem2smem_cta2(smem_d_addr + load_stage * 49152, (&D), rank_offset, a_rows, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_d_hi_addr + load_stage * 49152, (&D), rank_offset, a_rows + 128, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_lora_b_addr + load_stage * 49152, (&LORA_B), rank_offset, b_rows, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; _phase_tma_empty ^= 1; }
                }
                mbarrier_wait(clc_full_addr + (load_clc_stage) * 8, _phase_clc_full);
                uint32_t _clc_valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_1)
                    : "r"(clc_resp_addr + load_clc_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_0)
                    : "r"(clc_resp_addr + load_clc_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(clc_empty_addr + load_clc_stage * 8), "r"(0) : "memory");
                load_clc_stage += 1;
                if (load_clc_stage == 2) { load_clc_stage = 0; _phase_clc_full ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                tile_l = _clc_ctaid_0 / 2;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int completed = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int lora_on_1 = has_lora;
            float s_value_1 = 0.0f;
            if (lora_on_1 != 0) {
                s_value_1 = lora_scale[0];
                if (s_value_1 == 0.0f) {
                    lora_on_1 = 0;
                }
            }
            int lora_chunks_1 = RANK_CHUNKS;
            if (lora_on_1 == 0) {
                lora_chunks_1 = 0;
            }
            int rank_cols_left = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_rescale_done = 0;
            unsigned int _phase_clc_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int mma_clc_stage = 0;
                #pragma unroll 1
                for (unsigned int tile_1 = 0; tile_1 < num_tiles; tile_1++) {
                    int acc_base = (int)acc_stage * 512;
                    mbarrier_wait(acc_empty_addr + (acc_stage) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (unsigned int k_tile_1 = 0; k_tile_1 < K_tiles; k_tile_1++) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((k_tile_1 == 0) ? 1 : 0);
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272629776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_acc + (acc_base))), "r"(((init_flag) ? 0 : 1)));
                        int _mma_a_lo_1 = (((smem_a_hi_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072;
                        int _mma_b_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272629776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_acc + (acc_base + 256))), "r"(((init_flag) ? 0 : 1)));
                        elect_commit_cg2_multicast(tma_empty_addr + (mma_stage) * 8, (uint16_t)(3));
                        mma_stage += 1;
                        if (mma_stage == 4) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(base_full_addr + (acc_stage) * 8, (uint16_t)(3));
                    if (lora_on_1 != 0) {
                        mbarrier_wait(rescale_done_addr + (acc_stage) * 8, _phase_rescale_done);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    #pragma unroll 1
                    for (unsigned int rank_chunk_mma = 0; rank_chunk_mma < lora_chunks_1; rank_chunk_mma++) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        rank_cols_left = RANK - (int)rank_chunk_mma * 64;
                        if (rank_cols_left == 32) {
                            int _mma_a_lo_2 = (((smem_d_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072;
                            int _mma_b_lo_2 = (((smem_lora_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_acc + (acc_base))), "r"(1));
                            int _mma_a_lo_3 = (((smem_d_hi_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072;
                            int _mma_b_lo_3 = (((smem_lora_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_acc + (acc_base + 256))), "r"(1));
                        } else {
                            int _mma_a_lo_4 = (((smem_d_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072;
                            int _mma_b_lo_4 = (((smem_lora_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_acc + (acc_base))), "r"(1));
                            int _mma_a_lo_5 = (((smem_d_hi_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072;
                            int _mma_b_lo_5 = (((smem_lora_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 3072;
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 272630928;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_acc + (acc_base + 256))), "r"(1));
                        }
                        elect_commit_cg2_multicast(tma_empty_addr + (mma_stage) * 8, (uint16_t)(3));
                        mma_stage += 1;
                        if (mma_stage == 4) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(acc_full_addr + (acc_stage) * 8, (uint16_t)(3));
                    _phase_acc_empty ^= 1;
                    _phase_rescale_done ^= 1;
                    completed += 1;
                    mbarrier_wait(clc_full_addr + (mma_clc_stage) * 8, _phase_clc_full_1);
                    uint32_t _clc_valid_2 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_2)
                        : "r"(clc_resp_addr + mma_clc_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(clc_empty_addr + mma_clc_stage * 8), "r"(0) : "memory");
                    mma_clc_stage += 1;
                    if (mma_clc_stage == 2) { mma_clc_stage = 0; _phase_clc_full_1 ^= 1; }
                    if (_clc_valid_2 == 0) {
                        break;
                    }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                if (completed > 0) {
                    mbarrier_wait(acc_empty_addr, completed - 1 & 1);
                }
            }
            unsigned int _phase_clc_full_2 = 0;
            if (cta_rank != 0) {
                unsigned int peer_clc_stage = 0;
                #pragma unroll 1
                for (unsigned int _p = 0; _p < num_tiles; _p++) {
                    mbarrier_wait(clc_full_addr + (peer_clc_stage) * 8, _phase_clc_full_2);
                    uint32_t _clc_valid_3 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_3)
                        : "r"(clc_resp_addr + peer_clc_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(clc_empty_addr + peer_clc_stage * 8), "r"(0) : "memory");
                    peer_clc_stage += 1;
                    if (peer_clc_stage == 2) { peer_clc_stage = 0; _phase_clc_full_2 ^= 1; }
                    if (_clc_valid_3 == 0) {
                        break;
                    }
                }
            }
            int dealloc_peer_rank = cta_rank ^ 1;
            if (cta_rank != 0) {
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
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_bar_addr), "r"(dealloc_peer_rank) : "memory");
            }
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: prefetch ----
    if (warp == 2) {
        // idle — no tasks assigned
    }
    // ---- Role: idle ----
    if (warp == 3) {
        { // idle_main
            unsigned int _phase_thr_full = 0;
            unsigned int _phase_clc_empty = 1;
            unsigned int _phase_clc_full_3 = 0;
            if (cta_rank == 0) {
                unsigned int sched_clc_stage = 0;
                unsigned int sched_thr_stage = 0;
                #pragma unroll 1
                for (unsigned int _p_1 = 0; _p_1 < num_tiles; _p_1++) {
                    mbarrier_wait(thr_full_addr + (sched_thr_stage) * 8, _phase_thr_full);
                    mbarrier_arrive(thr_empty_addr + (sched_thr_stage) * 8);
                    sched_thr_stage += 1;
                    if (sched_thr_stage == 2) { sched_thr_stage = 0; _phase_thr_full ^= 1; }
                    mbarrier_wait(clc_empty_addr + (sched_clc_stage) * 8, _phase_clc_empty);
                    if (lane < 2) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(clc_full_addr + sched_clc_stage * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(clc_resp_addr + sched_clc_stage * 16 + 0 * 16), "r"(clc_full_addr + sched_clc_stage * 8)
                            : "memory");
                    }
                    mbarrier_wait(clc_full_addr + (sched_clc_stage) * 8, _phase_clc_full_3);
                    uint32_t _clc_valid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_0)
                        : "r"(clc_resp_addr + sched_clc_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(clc_empty_addr + sched_clc_stage * 8), "r"(0) : "memory");
                    sched_clc_stage += 1;
                    if (sched_clc_stage == 2) { sched_clc_stage = 0; _phase_clc_empty ^= 1; _phase_clc_full_3 ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                }
                #pragma unroll
                for (int _ = 0; _ < 2; _++) {
                    mbarrier_wait(clc_empty_addr + (sched_clc_stage) * 8, _phase_clc_empty);
                    sched_clc_stage += 1;
                    if (sched_clc_stage == 2) { sched_clc_stage = 0; _phase_clc_empty ^= 1; _phase_clc_full_3 ^= 1; }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp - 4;
            int epi_tid = epi_warp * 32 + lane;
            unsigned int acc_stage_1 = 0;
            unsigned int store_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int lora_on_2 = has_lora;
            float s_value_2 = 0.0f;
            if (lora_on_2 != 0) {
                s_value_2 = lora_scale[0];
                if (s_value_2 == 0.0f) {
                    lora_on_2 = 0;
                }
            }
            float alpha_value = a_scale[0] * b_scale[0];
            float2 _f2_0 = make_float2(alpha_value, alpha_value);
            float2 alpha_pair = _f2_0;
            float2 _f2_1 = make_float2(s_value_2, s_value_2);
            float2 s_pair = _f2_1;
            float2 final_pair = alpha_pair;
            float ratio = 1.0f;
            if (lora_on_2 != 0) {
                final_pair = s_pair;
                float _fdiv_rn_0 = __fdiv_rn(alpha_value, s_value_2);
                ratio = _fdiv_rn_0;
            }
            int lane_row_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            unsigned int epi_clc_stage = 0;
            unsigned int tile_e = cluster_id;
            unsigned int _phase_clc_full_4 = 0;
            unsigned int _phase_base_full = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int tile_2 = 0; tile_2 < num_tiles; tile_2++) {
                mbarrier_wait(clc_full_addr + (epi_clc_stage) * 8, _phase_clc_full_4);
                uint32_t _clc_valid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_4)
                    : "r"(clc_resp_addr + epi_clc_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_1)
                    : "r"(clc_resp_addr + epi_clc_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(clc_empty_addr + epi_clc_stage * 8), "r"(0) : "memory");
                epi_clc_stage += 1;
                if (epi_clc_stage == 2) { epi_clc_stage = 0; _phase_clc_full_4 ^= 1; }
                int w_idx_1 = tile_e / (unsigned int)tok_tiles;
                int tok_idx_1 = tile_e - (unsigned int)(w_idx_1 * tok_tiles);
                if (raster_group != 0) {
                    int _min_2 = ((raster_group - raster_group / 256 * 256) < ((unsigned int)tok_tiles - tile_e / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)(raster_group - raster_group / 256 * 256)) ? (raster_group - raster_group / 256 * 256) : ((unsigned int)tok_tiles - tile_e / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)(raster_group - raster_group / 256 * 256)));
                    w_idx_1 = (tile_e - tile_e / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles)) / (unsigned int)(_min_2 * (raster_group / 256)) * (unsigned int)(raster_group / 256) + (tile_e - tile_e / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) - (tile_e - tile_e / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles)) / (unsigned int)(_min_2 * (raster_group / 256)) * (unsigned int)(_min_2 * (raster_group / 256))) / (unsigned int)_min_2;
                    int _min_3 = ((raster_group - raster_group / 256 * 256) < ((unsigned int)tok_tiles - tile_e / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)(raster_group - raster_group / 256 * 256)) ? (raster_group - raster_group / 256 * 256) : ((unsigned int)tok_tiles - tile_e / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)(raster_group - raster_group / 256 * 256)));
                    tok_idx_1 = tile_e / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)(raster_group - raster_group / 256 * 256) + (tile_e - tile_e / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) - (tile_e - tile_e / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles)) / (unsigned int)(_min_3 * (raster_group / 256)) * (unsigned int)(_min_3 * (raster_group / 256))) - (tile_e - tile_e / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) - (tile_e - tile_e / (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles) * (unsigned int)((raster_group - raster_group / 256 * 256) * w_tiles)) / (unsigned int)(_min_3 * (raster_group / 256)) * (unsigned int)(_min_3 * (raster_group / 256))) / (unsigned int)_min_3 * (unsigned int)_min_3;
                }
                int off_tok = tok_idx_1 * 512 + cta_rank * 256;
                int off_w = w_idx_1 * 256;
                int acc_base_1 = (int)acc_stage_1 * 512;
                if (epi_tid < 128) {
                    unsigned int staged_bias_pair = 0;
                    int bias_col = off_w + epi_tid * 2;
                    if (has_bias != 0) {
                        if (bias_col < N) {
                            unsigned int _vec_load_0[2];
                            {
                                uint32_t _bf16x2_bits_0;
                                _bf16x2_bits_0 = *reinterpret_cast<const uint32_t*>(bias + bias_col);
                                _vec_load_0[0] = _bf16x2_bits_0;
                            }
                            staged_bias_pair = _vec_load_0[0];
                        }
                    }
                    smem_bias_pairs[epi_tid] = staged_bias_pair;
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                mbarrier_wait(base_full_addr + (acc_stage_1) * 8, _phase_base_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (lora_on_2 != 0) {
                    #pragma unroll
                    for (int rescale_batch = 0; rescale_batch < 4; rescale_batch++) {
                        int rescale_addr = lane_row_addr + acc_base_1 + rescale_batch * 128;
                        float _tmem_load_0[128];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(rescale_addr));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                            : "r"(rescale_addr + 32));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95])
                            : "r"(rescale_addr + 64));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                            : "r"(rescale_addr + 96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int r = 0; r < 128; r++) {
                            _tmem_load_0[r] = _tmem_load_0[r] * ratio;
                        }
                        #pragma unroll
                        for (int rescale_chunk = 0; rescale_chunk < 4; rescale_chunk++) {
                            int chunk_addr = rescale_addr + rescale_chunk * 32;
                            tmem_st_x32_f32(chunk_addr, (_tmem_load_0 + rescale_chunk * 32));
                        }
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((rescale_done_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                }
                mbarrier_wait(acc_full_addr + (acc_stage_1) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int subtile = 0; subtile < 16; subtile++) {
                    int tmem_addr = lane_row_addr + acc_base_1 + subtile * 32;
                    float _tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(tmem_addr));
                    int row_half = subtile / 8;
                    int col_sub = subtile - row_half * 8;
                    unsigned int bias_pairs[16];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[0])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(0) + 3]))
                        : "r"(smem_bias_pairs_addr + (unsigned int)(col_sub * 16 * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[4])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(4) + 3]))
                        : "r"(smem_bias_pairs_addr + (unsigned int)((col_sub * 16 + 4) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[8])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(8) + 3]))
                        : "r"(smem_bias_pairs_addr + (unsigned int)((col_sub * 16 + 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[12])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(12) + 3]))
                        : "r"(smem_bias_pairs_addr + (unsigned int)((col_sub * 16 + 12) * 4)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (subtile == 15) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((acc_empty_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    float result[32];
                    unsigned int bias_pair = 0;
                    unsigned int bias_lo_bits = 0;
                    float bias_lo = 0.0f;
                    unsigned int bias_hi_bits = 0;
                    float bias_hi = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        bias_pair = bias_pairs[j];
                        bias_lo_bits = bias_pair << 16;
                        bias_lo = reinterpret_cast<float*>(&bias_lo_bits)[0];
                        bias_hi_bits = bias_pair & 4294901760u;
                        bias_hi = reinterpret_cast<float*>(&bias_hi_bits)[0];
                        float2 _f2_2 = make_float2(_tmem_load_1[2 * j], _tmem_load_1[2 * j + 1]);
                        float2 value_pair = _f2_2;
                        float2 _f2_3 = make_float2(bias_lo, bias_hi);
                        float2 bias_value_pair = _f2_3;
                        float2 result_pair = fma_f32x2_rn_ftz(value_pair, final_pair, bias_value_pair);
                        result[2 * j] = result_pair.x;
                        result[2 * j + 1] = result_pair.y;
                    }
                    uint32_t result_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(result[_lp*2 + 0], result[_lp*2+1 + 0]));
                        result_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    if (subtile > 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int prev_half = (subtile - 1) / 8;
                        if (warp == 4) {
                            if (elect_sync()) {
                                tma_store_2d((&out), off_w + (subtile - 1 - prev_half * 8) * 32, off_tok + prev_half * 128, smem_out_addr + store_stage * 8192);
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
                    int out_stage_row = store_stage * 128 + (unsigned int)epi_tid;
                    unsigned int out_abs = smem_out_addr + (unsigned int)(out_stage_row * 64);
                    unsigned int out_swz = out_abs / 8 & 48;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (0 ^ out_swz)))), "r"(result_bf16[0]), "r"(result_bf16[1]), "r"(result_bf16[2]), "r"(result_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (16 ^ out_swz)))), "r"(result_bf16[4]), "r"(result_bf16[5]), "r"(result_bf16[6]), "r"(result_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (32 ^ out_swz)))), "r"(result_bf16[8]), "r"(result_bf16[9]), "r"(result_bf16[10]), "r"(result_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (48 ^ out_swz)))), "r"(result_bf16[12]), "r"(result_bf16[13]), "r"(result_bf16[14]), "r"(result_bf16[15]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        tma_store_2d((&out), off_w + 224, off_tok + 128, smem_out_addr + store_stage * 8192);
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
                _phase_base_full ^= 1;
                _phase_acc_full ^= 1;
                if (_clc_valid_4 == 0) {
                    break;
                }
                tile_e = _clc_ctaid_1 / 2;
            }
            if (warp == 4) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    // Cleanup
}

template __global__ void kernel_cake_quant_lora_gemm_e58e380eacc1adcd5d34<32>(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap D, float* __restrict__ a_scale, float* __restrict__ b_scale, float* __restrict__ lora_scale, __nv_bfloat16* __restrict__ bias, const __grid_constant__ CUtensorMap out, int M, int N, int K_tiles, int tok_tiles, int w_tiles, int num_tiles, int raster_group, int has_bias, int has_lora, int rank, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap LORA_B);
template __global__ void kernel_cake_quant_lora_gemm_e58e380eacc1adcd5d34<64>(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap D, float* __restrict__ a_scale, float* __restrict__ b_scale, float* __restrict__ lora_scale, __nv_bfloat16* __restrict__ bias, const __grid_constant__ CUtensorMap out, int M, int N, int K_tiles, int tok_tiles, int w_tiles, int num_tiles, int raster_group, int has_bias, int has_lora, int rank, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap LORA_B);
template __global__ void kernel_cake_quant_lora_gemm_e58e380eacc1adcd5d34<128>(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap D, float* __restrict__ a_scale, float* __restrict__ b_scale, float* __restrict__ lora_scale, __nv_bfloat16* __restrict__ bias, const __grid_constant__ CUtensorMap out, int M, int N, int K_tiles, int tok_tiles, int w_tiles, int num_tiles, int raster_group, int has_bias, int has_lora, int rank, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap LORA_B);
