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
#define TMEM_NCOLS 512
#define TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 128
#define TMEM_TMEM_SFB_OFFSET 304
#define NUM_TMA_PIPE_STAGES 11
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 8192
#define SMEM_SMEM_A_STRIDE 18432
#define SMEM_SMEM_B_OFF 9216
#define SMEM_SMEM_B_STAGE_BYTES 4096
#define SMEM_SMEM_B_STRIDE 18432
#define SMEM_SMEM_SFA_OFF 13312
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 18432
#define SMEM_SMEM_V3_OFF 15360
#define SMEM_SMEM_V3_STAGE_BYTES 2048
#define SMEM_SMEM_V3_STRIDE 18432
#define SMEM_SMEM_V4_OFF 15360
#define SMEM_SMEM_V4_STAGE_BYTES 2048
#define SMEM_SMEM_V4_STRIDE 18432
#define SMEM_SMEM_OUT_OFF 203776
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_SFB_CP_OFF 15360
#define SMEM_SMEM_SFB_CP_STAGE_BYTES 4096
#define SMEM_SMEM_SFB_CP_STRIDE 18432
#define SMEM_SMEM_W_OFF 1024
#define SMEM_SMEM_W_STAGE_BYTES 227328
#define SMEM_SMEM_W_STRIDE 227328
#define SMEM_TOTAL 228352
#define THREADS 256

extern "C" {

__global__ __launch_bounds__(256, 1) __cluster_dims__(2,1,1) void
kernel_cake_nvfp4_per_token_becd24cbb8333dc48dcb(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, uint8_t* __restrict__ SFA_RAW, uint8_t* __restrict__ SFB_RAW, float* __restrict__ alpha, const __grid_constant__ CUtensorMap out, int M, int N, int K_tiles, int tok_tiles, int num_tiles)
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
    #define tma_empty_addr (mbar_base + 88)
    #define sf_done_addr (mbar_base + 176)
    #define acc_full_addr (mbar_base + 264)
    #define acc_empty_addr (mbar_base + 280)
    #define tmem_dealloc_bar_addr (mbar_base + 296)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_b_addr = smem + 9216;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 13312);
    const int smem_sfa_addr = smem + 13312;
    uint8_t* smem_v3 = reinterpret_cast<uint8_t*>(smem_raw + 15360);
    const int smem_v3_addr = smem + 15360;
    uint8_t* smem_v4 = reinterpret_cast<uint8_t*>(smem_raw + 15360);
    const int smem_v4_addr = smem + 15360;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 203776);
    const int smem_out_addr = smem + 203776;
    uint8_t* smem_sfb_cp = reinterpret_cast<uint8_t*>(smem_raw + 15360);
    const int smem_sfb_cp_addr = smem + 15360;
    unsigned int* smem_w = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_w_addr = smem + 1024;
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 38 barriers)
    // Mbarriers at smem_raw[0..304)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 11 barriers, init_count=34
            mbarrier_init(smem + 0, 34);
            mbarrier_init(smem + 8, 34);
            mbarrier_init(smem + 16, 34);
            mbarrier_init(smem + 24, 34);
            mbarrier_init(smem + 32, 34);
            mbarrier_init(smem + 40, 34);
            mbarrier_init(smem + 48, 34);
            mbarrier_init(smem + 56, 34);
            mbarrier_init(smem + 64, 34);
            mbarrier_init(smem + 72, 34);
            mbarrier_init(smem + 80, 34);
            // tma_empty: 11 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // sf_done: 11 barriers, init_count=32
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            mbarrier_init(smem + 192, 32);
            mbarrier_init(smem + 200, 32);
            mbarrier_init(smem + 208, 32);
            mbarrier_init(smem + 216, 32);
            mbarrier_init(smem + 224, 32);
            mbarrier_init(smem + 232, 32);
            mbarrier_init(smem + 240, 32);
            mbarrier_init(smem + 248, 32);
            mbarrier_init(smem + 256, 32);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // acc_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 280, 256);
            mbarrier_init(smem + 288, 256);
            // tmem_dealloc_bar: 1 barriers, init_count=32
            mbarrier_init(smem + 296, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 480 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 304);
    if (warp == 2) {
        int _tmem_hold = smem + 304;
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
    const int tmem_tmem_sfa = taddr + 128;
    const int tmem_tmem_sfb = taddr + 304;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_tma_empty = 1;
            #pragma unroll 1
            for (unsigned int tile = cluster_id; tile < num_tiles; tile += num_clusters) {
                int w_idx = tile / (unsigned int)tok_tiles;
                int tok_idx = tile - (unsigned int)(w_idx * tok_tiles);
                int a_rows = tok_idx * 128 + cta_rank * 64;
                int b_rows = w_idx * 64 + cta_rank * 32;
                int a_atom = a_rows / 128;
                int b_atom = w_idx * 64 / 128;
                #pragma unroll 1
                for (unsigned int k_tile = 0; k_tile < K_tiles; k_tile++) {
                    mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(26624)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 18432, (&A), 0, a_rows, k_tile, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_b_addr + load_stage * 18432), "l"((&B)), "r"(0), "r"(b_rows), "r"(k_tile),
                               "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            tma_3d_gmem2smem_cta2(smem_sfa_addr + load_stage * 18432, (&SFA), 0, 4 * k_tile, a_atom, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        }
                    }
                    load_stage += 1;
                    if (load_stage == 11) { load_stage = 0; _phase_tma_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int completed = 0;
            unsigned int zero_w_m = 0;
            int zero_i_m = 0;
            if (cta_rank != 0) {
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(lane * 16 + 12288 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(lane * 16 + 12288 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(lane * 16 + 12288 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(lane * 16 + 12288 + 1536 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(18432 + lane * 16 + 12288 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(18432 + lane * 16 + 12288 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(18432 + lane * 16 + 12288 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(18432 + lane * 16 + 12288 + 1536 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(36864 + lane * 16 + 12288 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(36864 + lane * 16 + 12288 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(36864 + lane * 16 + 12288 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(36864 + lane * 16 + 12288 + 1536 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(55296 + lane * 16 + 12288 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(55296 + lane * 16 + 12288 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(55296 + lane * 16 + 12288 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(55296 + lane * 16 + 12288 + 1536 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(73728 + lane * 16 + 12288 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(73728 + lane * 16 + 12288 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(73728 + lane * 16 + 12288 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(73728 + lane * 16 + 12288 + 1536 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(92160 + lane * 16 + 12288 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(92160 + lane * 16 + 12288 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(92160 + lane * 16 + 12288 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(92160 + lane * 16 + 12288 + 1536 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(110592 + lane * 16 + 12288 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(110592 + lane * 16 + 12288 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(110592 + lane * 16 + 12288 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(110592 + lane * 16 + 12288 + 1536 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(129024 + lane * 16 + 12288 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(129024 + lane * 16 + 12288 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(129024 + lane * 16 + 12288 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(129024 + lane * 16 + 12288 + 1536 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(147456 + lane * 16 + 12288 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(147456 + lane * 16 + 12288 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(147456 + lane * 16 + 12288 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(147456 + lane * 16 + 12288 + 1536 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(165888 + lane * 16 + 12288 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(165888 + lane * 16 + 12288 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(165888 + lane * 16 + 12288 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(165888 + lane * 16 + 12288 + 1536 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(184320 + lane * 16 + 12288 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(184320 + lane * 16 + 12288 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(184320 + lane * 16 + 12288 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
                asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(184320 + lane * 16 + 12288 + 1536 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            }
            {
                uint32_t _ival_0 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_0 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_0), "r"(_ival_0) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_1 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_1 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_1), "r"(_ival_1) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_2 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_2 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 1024 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_2), "r"(_ival_2) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_3 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_3 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 1024 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_3), "r"(_ival_3) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 1024 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_4 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_4 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 2048 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_4), "r"(_ival_4) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 2048 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_5 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_5 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 2048 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_5), "r"(_ival_5) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 2048 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_6 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_6 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 3072 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_6), "r"(_ival_6) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 3072 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_7 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_7 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 3072 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_7), "r"(_ival_7) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(lane * 16 + 14336 + 3072 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_8 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_8 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_8), "r"(_ival_8) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_9 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_9 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_9), "r"(_ival_9) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_10 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_10 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 1024 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_10), "r"(_ival_10) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_11 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_11 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 1024 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_11), "r"(_ival_11) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 1024 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_12 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_12 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 2048 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_12), "r"(_ival_12) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 2048 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_13 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_13 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 2048 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_13), "r"(_ival_13) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 2048 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_14 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_14 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 3072 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_14), "r"(_ival_14) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 3072 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_15 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_15 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 3072 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_15), "r"(_ival_15) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(18432 + lane * 16 + 14336 + 3072 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_16 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_16 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_16), "r"(_ival_16) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_17 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_17 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_17), "r"(_ival_17) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_18 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_18 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 1024 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_18), "r"(_ival_18) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_19 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_19 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 1024 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_19), "r"(_ival_19) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 1024 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_20 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_20 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 2048 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_20), "r"(_ival_20) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 2048 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_21 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_21 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 2048 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_21), "r"(_ival_21) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 2048 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_22 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_22 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 3072 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_22), "r"(_ival_22) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 3072 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_23 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_23 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 3072 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_23), "r"(_ival_23) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(36864 + lane * 16 + 14336 + 3072 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_24 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_24 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_24), "r"(_ival_24) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_25 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_25 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_25), "r"(_ival_25) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_26 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_26 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 1024 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_26), "r"(_ival_26) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_27 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_27 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 1024 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_27), "r"(_ival_27) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 1024 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_28 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_28 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 2048 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_28), "r"(_ival_28) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 2048 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_29 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_29 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 2048 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_29), "r"(_ival_29) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 2048 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_30 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_30 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 3072 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_30), "r"(_ival_30) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 3072 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_31 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_31 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 3072 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_31), "r"(_ival_31) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(55296 + lane * 16 + 14336 + 3072 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_32 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_32 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_32), "r"(_ival_32) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_33 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_33 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_33), "r"(_ival_33) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_34 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_34 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 1024 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_34), "r"(_ival_34) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_35 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_35 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 1024 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_35), "r"(_ival_35) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 1024 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_36 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_36 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 2048 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_36), "r"(_ival_36) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 2048 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_37 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_37 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 2048 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_37), "r"(_ival_37) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 2048 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_38 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_38 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 3072 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_38), "r"(_ival_38) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 3072 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_39 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_39 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 3072 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_39), "r"(_ival_39) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(73728 + lane * 16 + 14336 + 3072 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_40 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_40 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_40), "r"(_ival_40) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_41 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_41 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_41), "r"(_ival_41) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_42 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_42 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 1024 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_42), "r"(_ival_42) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_43 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_43 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 1024 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_43), "r"(_ival_43) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 1024 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_44 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_44 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 2048 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_44), "r"(_ival_44) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 2048 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_45 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_45 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 2048 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_45), "r"(_ival_45) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 2048 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_46 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_46 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 3072 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_46), "r"(_ival_46) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 3072 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_47 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_47 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 3072 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_47), "r"(_ival_47) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(92160 + lane * 16 + 14336 + 3072 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_48 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_48 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_48), "r"(_ival_48) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_49 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_49 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_49), "r"(_ival_49) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_50 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_50 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 1024 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_50), "r"(_ival_50) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_51 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_51 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 1024 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_51), "r"(_ival_51) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 1024 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_52 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_52 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 2048 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_52), "r"(_ival_52) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 2048 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_53 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_53 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 2048 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_53), "r"(_ival_53) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 2048 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_54 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_54 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 3072 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_54), "r"(_ival_54) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 3072 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_55 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_55 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 3072 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_55), "r"(_ival_55) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(110592 + lane * 16 + 14336 + 3072 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_56 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_56 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_56), "r"(_ival_56) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_57 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_57 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_57), "r"(_ival_57) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_58 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_58 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 1024 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_58), "r"(_ival_58) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_59 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_59 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 1024 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_59), "r"(_ival_59) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 1024 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_60 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_60 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 2048 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_60), "r"(_ival_60) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 2048 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_61 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_61 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 2048 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_61), "r"(_ival_61) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 2048 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_62 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_62 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 3072 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_62), "r"(_ival_62) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 3072 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_63 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_63 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 3072 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_63), "r"(_ival_63) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(129024 + lane * 16 + 14336 + 3072 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_64 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_64 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_64), "r"(_ival_64) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_65 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_65 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_65), "r"(_ival_65) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_66 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_66 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 1024 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_66), "r"(_ival_66) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_67 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_67 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 1024 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_67), "r"(_ival_67) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 1024 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_68 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_68 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 2048 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_68), "r"(_ival_68) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 2048 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_69 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_69 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 2048 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_69), "r"(_ival_69) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 2048 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_70 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_70 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 3072 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_70), "r"(_ival_70) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 3072 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_71 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_71 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 3072 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_71), "r"(_ival_71) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(147456 + lane * 16 + 14336 + 3072 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_72 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_72 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_72), "r"(_ival_72) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_73 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_73 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_73), "r"(_ival_73) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_74 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_74 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 1024 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_74), "r"(_ival_74) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_75 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_75 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 1024 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_75), "r"(_ival_75) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 1024 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_76 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_76 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 2048 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_76), "r"(_ival_76) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 2048 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_77 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_77 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 2048 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_77), "r"(_ival_77) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 2048 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_78 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_78 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 3072 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_78), "r"(_ival_78) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 3072 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_79 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_79 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 3072 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_79), "r"(_ival_79) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(165888 + lane * 16 + 14336 + 3072 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_80 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_80 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_80), "r"(_ival_80) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_81 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_81 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_81), "r"(_ival_81) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_82 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_82 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 1024 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_82), "r"(_ival_82) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 1024 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_83 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_83 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 1024 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_83), "r"(_ival_83) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 1024 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_84 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_84 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 2048 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_84), "r"(_ival_84) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 2048 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_85 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_85 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 2048 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_85), "r"(_ival_85) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 2048 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_86 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_86 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 3072 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_86), "r"(_ival_86) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 3072 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            {
                uint32_t _ival_87 = static_cast<uint32_t>(zero_i_m);
                uint32_t _addr_87 = static_cast<uint32_t>(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 3072 + 512 + 4));
                asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_87), "r"(_ival_87) : "memory");
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(smem_w_addr + (unsigned int)(184320 + lane * 16 + 14336 + 3072 + 512 + 8)), "r"(zero_w_m), "r"(zero_w_m) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            unsigned int _phase_sf_done = 0;
            if (cta_rank != 0) {
                unsigned int pub_stage = 0;
                #pragma unroll 1
                for (unsigned int tile_pb = cluster_id; tile_pb < num_tiles; tile_pb += num_clusters) {
                    #pragma unroll 1
                    for (unsigned int k_pb = 0; k_pb < K_tiles; k_pb++) {
                        mbarrier_wait(sf_done_addr + (pub_stage) * 8, _phase_sf_done);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(tma_full_addr + pub_stage * 8), "r"(0) : "memory");
                        }
                        pub_stage += 1;
                        if (pub_stage == 11) { pub_stage = 0; _phase_sf_done ^= 1; }
                    }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int tile_1 = cluster_id; tile_1 < num_tiles; tile_1 += num_clusters) {
                    int acc_base = (int)acc_stage * 64;
                    int w_idx_mma = tile_1 / (unsigned int)tok_tiles;
                    int sfb_shift = 0;
                    mbarrier_wait(acc_empty_addr + (acc_stage) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (unsigned int k_tile_1 = 0; k_tile_1 < K_tiles; k_tile_1++) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int sfa_col = (int)mma_stage * 16;
                        int sfb_col = (int)mma_stage * 16;
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfa + mma_stage * 16, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 1152)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 1152 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 1152 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + mma_stage * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 1152 + 96)));
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        if (elect_sync()) {
                            {
                                uint64_t _tcgen05_cp_desc_88 = ((((uint64_t)(smem_sfb_cp_addr + mma_stage * 18432)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_tmem_sfb + sfb_col)), "l"(_tcgen05_cp_desc_88)
                                    : "memory");
                            }
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        if (elect_sync()) {
                            {
                                uint64_t _tcgen05_cp_desc_89 = ((((uint64_t)(smem_sfb_cp_addr + mma_stage * 18432 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_tmem_sfb + (sfb_col + 4))), "l"(_tcgen05_cp_desc_89)
                                    : "memory");
                            }
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        if (elect_sync()) {
                            {
                                uint64_t _tcgen05_cp_desc_90 = ((((uint64_t)(smem_sfb_cp_addr + mma_stage * 18432 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_tmem_sfb + (sfb_col + 8))), "l"(_tcgen05_cp_desc_90)
                                    : "memory");
                            }
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        if (elect_sync()) {
                            {
                                uint64_t _tcgen05_cp_desc_91 = ((((uint64_t)(smem_sfb_cp_addr + mma_stage * 18432 + 3072)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_tmem_sfb + (sfb_col + 12))), "l"(_tcgen05_cp_desc_91)
                                    : "memory");
                            }
                        }
                        int init_flag = ((k_tile_1 == 0) ? 1 : 0);
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 1152;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 1152;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa + sfa_col + 0, tmem_tmem_sfb + (sfb_col + sfb_shift) + 0, ((((1) ? init_flag : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 1152;
                        int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 1152;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa + (sfa_col + 4) + 0, tmem_tmem_sfb + (sfb_col + 4 + sfb_shift) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 1152;
                        int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 1152;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa + (sfa_col + 8) + 0, tmem_tmem_sfb + (sfb_col + 8 + sfb_shift) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 1152;
                        int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 1152;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x8100480U, tmem_tmem_sfa + (sfa_col + 12) + 0, tmem_tmem_sfb + (sfb_col + 12 + sfb_shift) + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                            }
                        }
                        elect_commit_cg2_multicast(tma_empty_addr + (mma_stage) * 8, (uint16_t)(3));
                        mma_stage += 1;
                        if (mma_stage == 11) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(acc_full_addr + (acc_stage) * 8, (uint16_t)(3));
                    acc_stage += 1;
                    if (acc_stage == 2) { acc_stage = 0; _phase_acc_empty ^= 1; }
                    completed += 1;
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                if (completed > 0) {
                    mbarrier_wait(acc_empty_addr + ((completed - 1) % 2) * 8, (completed - 1) / 2 & 1);
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
        { // prefetch_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int pf_stage = 0;
            unsigned int _phase_tma_empty_1 = 1;
            #pragma unroll 1
            for (unsigned int tile_pf = cluster_id; tile_pf < num_tiles; tile_pf += num_clusters) {
                int w_idx_pf = tile_pf / (unsigned int)tok_tiles;
                int tok_idx_pf = tile_pf - (unsigned int)(w_idx_pf * tok_tiles);
                int a_atom_pf = (tok_idx_pf * 128 + cta_rank * 64) / 128;
                int b_atom_pf = w_idx_pf * 64 / 128;
                int sfb_rg0_pf = w_idx_pf * 64 % 128 / 32;
                #pragma unroll 1
                for (unsigned int k_tile_pf = 0; k_tile_pf < K_tiles; k_tile_pf++) {
                    mbarrier_wait(tma_empty_addr + (pf_stage) * 8, _phase_tma_empty_1);
                    if (cta_rank != 0) {
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                            :: "r"(smem_sfa_addr + pf_stage * 18432 + (unsigned int)((int)lane * 16)), "l"(SFA_RAW + (a_atom_pf * (K_tiles * 2048) + 4 * (int)k_tile_pf * 512 + (int)lane * 16 + 8)));
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                            :: "r"(smem_sfa_addr + pf_stage * 18432 + (unsigned int)(512 + (int)lane * 16)), "l"(SFA_RAW + (a_atom_pf * (K_tiles * 2048) + 4 * (int)k_tile_pf * 512 + (int)lane * 16 + 520)));
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                            :: "r"(smem_sfa_addr + pf_stage * 18432 + (unsigned int)(1024 + (int)lane * 16)), "l"(SFA_RAW + (a_atom_pf * (K_tiles * 2048) + 4 * (int)k_tile_pf * 512 + (int)lane * 16 + 1032)));
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 8;"
                            :: "r"(smem_sfa_addr + pf_stage * 18432 + (unsigned int)(1536 + (int)lane * 16)), "l"(SFA_RAW + (a_atom_pf * (K_tiles * 2048) + 4 * (int)k_tile_pf * 512 + (int)lane * 16 + 1544)));
                    }
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                        :: "r"(smem_sfb_cp_addr + pf_stage * 18432 + (unsigned int)((int)lane * 16)), "l"(SFB_RAW + (b_atom_pf * (K_tiles * 2048) + 4 * (int)k_tile_pf * 512 + (int)lane * 16 + sfb_rg0_pf * 4)));
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                        :: "r"(smem_sfb_cp_addr + pf_stage * 18432 + (unsigned int)(512 + (int)lane * 16)), "l"(SFB_RAW + (b_atom_pf * (K_tiles * 2048) + 4 * (int)k_tile_pf * 512 + (int)lane * 16 + sfb_rg0_pf * 4 + 4)));
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                        :: "r"(smem_sfb_cp_addr + pf_stage * 18432 + (unsigned int)(1024 + (int)lane * 16)), "l"(SFB_RAW + (b_atom_pf * (K_tiles * 2048) + 4 * (int)k_tile_pf * 512 + (int)lane * 16 + sfb_rg0_pf * 4 + 512)));
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                        :: "r"(smem_sfb_cp_addr + pf_stage * 18432 + (unsigned int)(1536 + (int)lane * 16)), "l"(SFB_RAW + (b_atom_pf * (K_tiles * 2048) + 4 * (int)k_tile_pf * 512 + (int)lane * 16 + sfb_rg0_pf * 4 + 516)));
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                        :: "r"(smem_sfb_cp_addr + pf_stage * 18432 + (unsigned int)(2048 + (int)lane * 16)), "l"(SFB_RAW + (b_atom_pf * (K_tiles * 2048) + 4 * (int)k_tile_pf * 512 + (int)lane * 16 + sfb_rg0_pf * 4 + 1024)));
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                        :: "r"(smem_sfb_cp_addr + pf_stage * 18432 + (unsigned int)(2560 + (int)lane * 16)), "l"(SFB_RAW + (b_atom_pf * (K_tiles * 2048) + 4 * (int)k_tile_pf * 512 + (int)lane * 16 + sfb_rg0_pf * 4 + 1028)));
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                        :: "r"(smem_sfb_cp_addr + pf_stage * 18432 + (unsigned int)(3072 + (int)lane * 16)), "l"(SFB_RAW + (b_atom_pf * (K_tiles * 2048) + 4 * (int)k_tile_pf * 512 + (int)lane * 16 + sfb_rg0_pf * 4 + 1536)));
                    asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                        :: "r"(smem_sfb_cp_addr + pf_stage * 18432 + (unsigned int)(3584 + (int)lane * 16)), "l"(SFB_RAW + (b_atom_pf * (K_tiles * 2048) + 4 * (int)k_tile_pf * 512 + (int)lane * 16 + sfb_rg0_pf * 4 + 1540)));
                    if (cta_rank != 0) {
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(sf_done_addr + (pf_stage) * 8) : "memory");
                    }
                    if (cta_rank == 0) {
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(tma_full_addr + (pf_stage) * 8) : "memory");
                    }
                    pf_stage += 1;
                    if (pf_stage == 11) { pf_stage = 0; _phase_tma_empty_1 ^= 1; }
                }
            }
        }
    }
    // ---- Role: sched ----
    if (warp == 3) {
        // idle — no tasks assigned
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp - 4;
            int epi_row = epi_warp % 2 * 32 + lane;
            int epi_srow = epi_warp * 32 + lane;
            unsigned int acc_stage_1 = 0;
            unsigned int store_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int tile_2 = cluster_id; tile_2 < num_tiles; tile_2 += num_clusters) {
                int w_idx_1 = tile_2 / (unsigned int)tok_tiles;
                int tok_idx_1 = tile_2 - (unsigned int)(w_idx_1 * tok_tiles);
                int off_tok = tok_idx_1 * 128 + cta_rank * 64;
                int off_w = w_idx_1 * 64;
                int acc_base_1 = (int)acc_stage_1 * 64;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_base_1;
                int _min_0 = ((off_tok + epi_row) < (M - 1) ? (off_tok + epi_row) : (M - 1));
                int tok_row = _min_0;
                float alpha_row = alpha[tok_row];
                mbarrier_wait(acc_full_addr + (acc_stage_1) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int subtile = 0; subtile < 1; subtile++) {
                    int tmem_addr = lane_addr + subtile * 32;
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(tmem_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (subtile == 0) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((acc_empty_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
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
                                tma_store_2d((&out), off_w + (subtile - 1) * 32, off_tok, smem_out_addr + store_stage * 8192);
                                tma_store_2d((&out), off_w + 32 + (subtile - 1) * 32, off_tok, smem_out_addr + store_stage * 8192 + 4096);
                            }
                        }
                        if (warp == 4) {
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 2;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        store_stage = store_stage + 1;
                        if (store_stage == 3) {
                            store_stage = 0;
                        }
                    }
                    int out_stage_row = store_stage * 128 + (unsigned int)epi_srow;
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
                        tma_store_2d((&out), off_w, off_tok, smem_out_addr + store_stage * 8192);
                        tma_store_2d((&out), off_w + 32, off_tok, smem_out_addr + store_stage * 8192 + 4096);
                    }
                }
                if (warp == 4) {
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 2;");
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                store_stage = store_stage + 1;
                if (store_stage == 3) {
                    store_stage = 0;
                }
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_acc_full ^= 1; }
            }
            if (warp == 4) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    // Cleanup
}

} // extern "C"
