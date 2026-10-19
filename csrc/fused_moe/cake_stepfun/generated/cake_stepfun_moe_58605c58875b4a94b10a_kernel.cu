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
#include "cake_stepfun_moe_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_ACCUM_OFFSET 0
#define NUM_K_PIPE_STAGES 7
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_FAST_PIPE_STAGES 1
#define NUM_EXIT_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 115712
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_EPI_STAGING_OFF 173056
#define SMEM_EPI_STAGING_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_STRIDE 8192
#define SMEM_EPI_STAGING_U64_OFF 173056
#define SMEM_EPI_STAGING_U64_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_U64_STRIDE 8192
#define SMEM_WORK_RESPONSE_OFF 181248
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_RESPONSE_OFF 181296
#define SMEM_FAST_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_RESPONSE_STRIDE 64
#define SMEM_TOTAL 181376
#define THREADS 256
#define BLOCK_N 128
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_stepfun_moe_58605c58875b4a94b10a(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    #define a_full_addr (mbar_base + 0)
    #define b_full_addr (mbar_base + 56)
    #define k_done_addr (mbar_base + 112)
    #define mma_full_addr (mbar_base + 168)
    #define mma_free_addr (mbar_base + 184)
    #define work_full_addr (mbar_base + 200)
    #define work_empty_addr (mbar_base + 224)
    #define throttle_full_addr (mbar_base + 248)
    #define throttle_empty_addr (mbar_base + 272)
    #define fast_ready_addr (mbar_base + 296)
    #define exit_bar_addr (mbar_base + 304)

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
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_EPI_STAGING_OFF);
    const int epi_staging_addr = smem + SMEM_EPI_STAGING_OFF;
    unsigned long long* epi_staging_u64 = reinterpret_cast<unsigned long long*>(smem_raw + SMEM_EPI_STAGING_U64_OFF);
    const int epi_staging_u64_addr = smem + SMEM_EPI_STAGING_U64_OFF;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WORK_RESPONSE_OFF);
    const int work_response_addr = smem + SMEM_WORK_RESPONSE_OFF;
    unsigned int* fast_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_FAST_RESPONSE_OFF);
    const int fast_response_addr = smem + SMEM_FAST_RESPONSE_OFF;

    // Mbarrier init (11 pipeline groups, 0 ordered-sequence groups, 39 barriers)
    // Mbarriers at smem_raw[0..312)

    if (warp == 0) {
        // --- pipeline 'k_pipe' ---
        // a_full: 7 barriers, init_count=2
        // b_full: 7 barriers, init_count=2
        // k_done: 7 barriers, init_count=1
        // --- pipeline 'mma_pipe' ---
        // mma_full: 2 barriers, init_count=1
        // mma_free: 2 barriers, init_count=256
        // --- pipeline 'work_pipe' ---
        // work_full: 3 barriers, init_count=1
        // work_empty: 3 barriers, init_count=448
        // --- pipeline 'throttle_pipe' ---
        // throttle_full: 3 barriers, init_count=32
        // throttle_empty: 3 barriers, init_count=32
        // --- pipeline 'fast_pipe' ---
        // fast_ready: 1 barriers, init_count=1
        // --- pipeline 'exit_pipe' ---
        // exit_bar: 1 barriers, init_count=32
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(31), "r"((uint32_t)(448)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(28), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(25), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(23), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(14), "r"((uint32_t)(2)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        uint32_t _mbarrier_init_count_0_32 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(6), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(5), "r"((uint32_t)(32)));
        if (lane < 7) {
            mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 312);
    if (warp == 0) {
        int _tmem_hold = smem + 312;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    // Partial post-allocation TMEM rendezvous (192 threads on named barrier 6)
    if (warp == 0 || warp == 1 || warp == 2 || warp == 3 || warp == 4 || warp == 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(6), "r"(192) : "memory");
    }

    const int taddr = (warp == 0 || warp == 1 || warp == 2 || warp == 3 || warp == 4 || warp == 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count = num_non_exiting_ctas[0];
            const int warp_0 = warp;
            const int lane_1 = lane;
            unsigned int acc_stage = 0;
            unsigned int work_stage = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            int base_feature = warp_0 * 32 + lane_1 / 4 * 4;
            int base_token = lane_1 % 4 * 2;
            float converted[4];
            unsigned int packed[2];
            unsigned long long packed_word = 0;
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m / 2 * grid_n; _tile_iter++) {
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 128;
                if (tile_count > (int)n_tile) {
                    if (valid_rows > 0) {
                        int expert_e = tile_expert[n_tile];
                        float sc = scale_c[expert_e];
                        mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int acc_offset = acc_stage * 128;
                        float _tmem_load_0[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                            : "r"(taddr + (unsigned int)acc_offset));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_1[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                            : "r"(taddr + 1048576 + (unsigned int)acc_offset));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        int token = base_token;
                        float _mul_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(_tmem_load_0[0]), "f"(sc));
                        converted[0] = _mul_0;
                        float _mul_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_tmem_load_0[2]), "f"(sc));
                        converted[1] = _mul_1;
                        float _mul_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_tmem_load_1[0]), "f"(sc));
                        converted[2] = _mul_2;
                        float _mul_3;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_tmem_load_1[2]), "f"(sc));
                        converted[3] = _mul_3;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature = token % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token * 64 + (base_feature ^ swizzle_feature)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token * 64 + (base_feature - 64 ^ swizzle_feature)) / 4] = packed_word;
                        }
                        int token_0 = base_token + 1;
                        float _mul_4;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_tmem_load_0[1]), "f"(sc));
                        converted[0] = _mul_4;
                        float _mul_5;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_tmem_load_0[3]), "f"(sc));
                        converted[1] = _mul_5;
                        float _mul_6;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_tmem_load_1[1]), "f"(sc));
                        converted[2] = _mul_6;
                        float _mul_7;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_tmem_load_1[3]), "f"(sc));
                        converted[3] = _mul_7;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_1 = token_0 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_0 * 64 + (base_feature ^ swizzle_feature_1)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_0 * 64 + (base_feature - 64 ^ swizzle_feature_1)) / 4] = packed_word;
                        }
                        int token_2 = base_token + 8;
                        float _mul_8;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_tmem_load_0[4]), "f"(sc));
                        converted[0] = _mul_8;
                        float _mul_9;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_tmem_load_0[6]), "f"(sc));
                        converted[1] = _mul_9;
                        float _mul_10;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(_tmem_load_1[4]), "f"(sc));
                        converted[2] = _mul_10;
                        float _mul_11;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_tmem_load_1[6]), "f"(sc));
                        converted[3] = _mul_11;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_3 = token_2 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_2 * 64 + (base_feature ^ swizzle_feature_3)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_2 * 64 + (base_feature - 64 ^ swizzle_feature_3)) / 4] = packed_word;
                        }
                        int token_4 = base_token + 8 + 1;
                        float _mul_12;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(_tmem_load_0[5]), "f"(sc));
                        converted[0] = _mul_12;
                        float _mul_13;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_tmem_load_0[7]), "f"(sc));
                        converted[1] = _mul_13;
                        float _mul_14;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(_tmem_load_1[5]), "f"(sc));
                        converted[2] = _mul_14;
                        float _mul_15;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_tmem_load_1[7]), "f"(sc));
                        converted[3] = _mul_15;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_5 = token_4 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_4 * 64 + (base_feature ^ swizzle_feature_5)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_4 * 64 + (base_feature - 64 ^ swizzle_feature_5)) / 4] = packed_word;
                        }
                        int token_6 = base_token + 16;
                        float _mul_16;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(_tmem_load_0[8]), "f"(sc));
                        converted[0] = _mul_16;
                        float _mul_17;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_tmem_load_0[10]), "f"(sc));
                        converted[1] = _mul_17;
                        float _mul_18;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_tmem_load_1[8]), "f"(sc));
                        converted[2] = _mul_18;
                        float _mul_19;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_tmem_load_1[10]), "f"(sc));
                        converted[3] = _mul_19;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_7 = token_6 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_6 * 64 + (base_feature ^ swizzle_feature_7)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_6 * 64 + (base_feature - 64 ^ swizzle_feature_7)) / 4] = packed_word;
                        }
                        int token_8 = base_token + 16 + 1;
                        float _mul_20;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(_tmem_load_0[9]), "f"(sc));
                        converted[0] = _mul_20;
                        float _mul_21;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_tmem_load_0[11]), "f"(sc));
                        converted[1] = _mul_21;
                        float _mul_22;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_tmem_load_1[9]), "f"(sc));
                        converted[2] = _mul_22;
                        float _mul_23;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_tmem_load_1[11]), "f"(sc));
                        converted[3] = _mul_23;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_9 = token_8 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_8 * 64 + (base_feature ^ swizzle_feature_9)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_8 * 64 + (base_feature - 64 ^ swizzle_feature_9)) / 4] = packed_word;
                        }
                        int token_10 = base_token + 24;
                        float _mul_24;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(_tmem_load_0[12]), "f"(sc));
                        converted[0] = _mul_24;
                        float _mul_25;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(_tmem_load_0[14]), "f"(sc));
                        converted[1] = _mul_25;
                        float _mul_26;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(_tmem_load_1[12]), "f"(sc));
                        converted[2] = _mul_26;
                        float _mul_27;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(_tmem_load_1[14]), "f"(sc));
                        converted[3] = _mul_27;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_11 = token_10 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_10 * 64 + (base_feature ^ swizzle_feature_11)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_10 * 64 + (base_feature - 64 ^ swizzle_feature_11)) / 4] = packed_word;
                        }
                        int token_12 = base_token + 24 + 1;
                        float _mul_28;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(_tmem_load_0[13]), "f"(sc));
                        converted[0] = _mul_28;
                        float _mul_29;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(_tmem_load_0[15]), "f"(sc));
                        converted[1] = _mul_29;
                        float _mul_30;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(_tmem_load_1[13]), "f"(sc));
                        converted[2] = _mul_30;
                        float _mul_31;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(_tmem_load_1[15]), "f"(sc));
                        converted[3] = _mul_31;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_13 = token_12 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_12 * 64 + (base_feature ^ swizzle_feature_13)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_12 * 64 + (base_feature - 64 ^ swizzle_feature_13)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows = (128 - valid_rows % 128) % 128;
                                tma_store_4d((&C), m_tile * 128, padding_rows, 1073741824, n_tile * 128 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows, 1073741824, n_tile * 128 - (unsigned int)padding_rows + 1073741824, epi_staging_addr + 4096);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        int acc_offset_14 = acc_stage * 128 + 32;
                        float _tmem_load_2[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                            : "r"(taddr + (unsigned int)acc_offset_14));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_3[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                            : "r"(taddr + 1048576 + (unsigned int)acc_offset_14));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        int token_15 = base_token;
                        float _mul_32;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(_tmem_load_2[0]), "f"(sc));
                        converted[0] = _mul_32;
                        float _mul_33;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(_tmem_load_2[2]), "f"(sc));
                        converted[1] = _mul_33;
                        float _mul_34;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(_tmem_load_3[0]), "f"(sc));
                        converted[2] = _mul_34;
                        float _mul_35;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(_tmem_load_3[2]), "f"(sc));
                        converted[3] = _mul_35;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_16 = token_15 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_15 * 64 + (base_feature ^ swizzle_feature_16)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_15 * 64 + (base_feature - 64 ^ swizzle_feature_16)) / 4] = packed_word;
                        }
                        int token_17 = base_token + 1;
                        float _mul_36;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(_tmem_load_2[1]), "f"(sc));
                        converted[0] = _mul_36;
                        float _mul_37;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(_tmem_load_2[3]), "f"(sc));
                        converted[1] = _mul_37;
                        float _mul_38;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(_tmem_load_3[1]), "f"(sc));
                        converted[2] = _mul_38;
                        float _mul_39;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(_tmem_load_3[3]), "f"(sc));
                        converted[3] = _mul_39;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_18 = token_17 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_17 * 64 + (base_feature ^ swizzle_feature_18)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_17 * 64 + (base_feature - 64 ^ swizzle_feature_18)) / 4] = packed_word;
                        }
                        int token_19 = base_token + 8;
                        float _mul_40;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(_tmem_load_2[4]), "f"(sc));
                        converted[0] = _mul_40;
                        float _mul_41;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(_tmem_load_2[6]), "f"(sc));
                        converted[1] = _mul_41;
                        float _mul_42;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_42) : "f"(_tmem_load_3[4]), "f"(sc));
                        converted[2] = _mul_42;
                        float _mul_43;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_43) : "f"(_tmem_load_3[6]), "f"(sc));
                        converted[3] = _mul_43;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_20 = token_19 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_19 * 64 + (base_feature ^ swizzle_feature_20)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_19 * 64 + (base_feature - 64 ^ swizzle_feature_20)) / 4] = packed_word;
                        }
                        int token_21 = base_token + 8 + 1;
                        float _mul_44;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_44) : "f"(_tmem_load_2[5]), "f"(sc));
                        converted[0] = _mul_44;
                        float _mul_45;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_45) : "f"(_tmem_load_2[7]), "f"(sc));
                        converted[1] = _mul_45;
                        float _mul_46;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_46) : "f"(_tmem_load_3[5]), "f"(sc));
                        converted[2] = _mul_46;
                        float _mul_47;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_47) : "f"(_tmem_load_3[7]), "f"(sc));
                        converted[3] = _mul_47;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_22 = token_21 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_21 * 64 + (base_feature ^ swizzle_feature_22)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_21 * 64 + (base_feature - 64 ^ swizzle_feature_22)) / 4] = packed_word;
                        }
                        int token_23 = base_token + 16;
                        float _mul_48;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_48) : "f"(_tmem_load_2[8]), "f"(sc));
                        converted[0] = _mul_48;
                        float _mul_49;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_49) : "f"(_tmem_load_2[10]), "f"(sc));
                        converted[1] = _mul_49;
                        float _mul_50;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_50) : "f"(_tmem_load_3[8]), "f"(sc));
                        converted[2] = _mul_50;
                        float _mul_51;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_51) : "f"(_tmem_load_3[10]), "f"(sc));
                        converted[3] = _mul_51;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_24 = token_23 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_23 * 64 + (base_feature ^ swizzle_feature_24)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_23 * 64 + (base_feature - 64 ^ swizzle_feature_24)) / 4] = packed_word;
                        }
                        int token_25 = base_token + 16 + 1;
                        float _mul_52;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_52) : "f"(_tmem_load_2[9]), "f"(sc));
                        converted[0] = _mul_52;
                        float _mul_53;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_53) : "f"(_tmem_load_2[11]), "f"(sc));
                        converted[1] = _mul_53;
                        float _mul_54;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_54) : "f"(_tmem_load_3[9]), "f"(sc));
                        converted[2] = _mul_54;
                        float _mul_55;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_55) : "f"(_tmem_load_3[11]), "f"(sc));
                        converted[3] = _mul_55;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_26 = token_25 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_25 * 64 + (base_feature ^ swizzle_feature_26)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_25 * 64 + (base_feature - 64 ^ swizzle_feature_26)) / 4] = packed_word;
                        }
                        int token_27 = base_token + 24;
                        float _mul_56;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_56) : "f"(_tmem_load_2[12]), "f"(sc));
                        converted[0] = _mul_56;
                        float _mul_57;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_57) : "f"(_tmem_load_2[14]), "f"(sc));
                        converted[1] = _mul_57;
                        float _mul_58;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_58) : "f"(_tmem_load_3[12]), "f"(sc));
                        converted[2] = _mul_58;
                        float _mul_59;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_59) : "f"(_tmem_load_3[14]), "f"(sc));
                        converted[3] = _mul_59;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_28 = token_27 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_27 * 64 + (base_feature ^ swizzle_feature_28)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_27 * 64 + (base_feature - 64 ^ swizzle_feature_28)) / 4] = packed_word;
                        }
                        int token_29 = base_token + 24 + 1;
                        float _mul_60;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_60) : "f"(_tmem_load_2[13]), "f"(sc));
                        converted[0] = _mul_60;
                        float _mul_61;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_61) : "f"(_tmem_load_2[15]), "f"(sc));
                        converted[1] = _mul_61;
                        float _mul_62;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_62) : "f"(_tmem_load_3[13]), "f"(sc));
                        converted[2] = _mul_62;
                        float _mul_63;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_63) : "f"(_tmem_load_3[15]), "f"(sc));
                        converted[3] = _mul_63;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_30 = token_29 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_29 * 64 + (base_feature ^ swizzle_feature_30)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_29 * 64 + (base_feature - 64 ^ swizzle_feature_30)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows_1 = (128 - valid_rows % 128) % 128;
                                tma_store_4d((&C), m_tile * 128, padding_rows_1 + 32, 1073741824, n_tile * 128 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows_1 + 32, 1073741824, n_tile * 128 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr + 4096);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        int acc_offset_31 = acc_stage * 128 + 64;
                        float _tmem_load_4[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                            : "r"(taddr + (unsigned int)acc_offset_31));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_5[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                            : "r"(taddr + 1048576 + (unsigned int)acc_offset_31));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        int token_32 = base_token;
                        float _mul_64;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_64) : "f"(_tmem_load_4[0]), "f"(sc));
                        converted[0] = _mul_64;
                        float _mul_65;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_65) : "f"(_tmem_load_4[2]), "f"(sc));
                        converted[1] = _mul_65;
                        float _mul_66;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_66) : "f"(_tmem_load_5[0]), "f"(sc));
                        converted[2] = _mul_66;
                        float _mul_67;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_67) : "f"(_tmem_load_5[2]), "f"(sc));
                        converted[3] = _mul_67;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_33 = token_32 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_32 * 64 + (base_feature ^ swizzle_feature_33)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_32 * 64 + (base_feature - 64 ^ swizzle_feature_33)) / 4] = packed_word;
                        }
                        int token_34 = base_token + 1;
                        float _mul_68;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_68) : "f"(_tmem_load_4[1]), "f"(sc));
                        converted[0] = _mul_68;
                        float _mul_69;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_69) : "f"(_tmem_load_4[3]), "f"(sc));
                        converted[1] = _mul_69;
                        float _mul_70;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_70) : "f"(_tmem_load_5[1]), "f"(sc));
                        converted[2] = _mul_70;
                        float _mul_71;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_71) : "f"(_tmem_load_5[3]), "f"(sc));
                        converted[3] = _mul_71;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_35 = token_34 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_34 * 64 + (base_feature ^ swizzle_feature_35)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_34 * 64 + (base_feature - 64 ^ swizzle_feature_35)) / 4] = packed_word;
                        }
                        int token_36 = base_token + 8;
                        float _mul_72;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_72) : "f"(_tmem_load_4[4]), "f"(sc));
                        converted[0] = _mul_72;
                        float _mul_73;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_73) : "f"(_tmem_load_4[6]), "f"(sc));
                        converted[1] = _mul_73;
                        float _mul_74;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_74) : "f"(_tmem_load_5[4]), "f"(sc));
                        converted[2] = _mul_74;
                        float _mul_75;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_75) : "f"(_tmem_load_5[6]), "f"(sc));
                        converted[3] = _mul_75;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_37 = token_36 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_36 * 64 + (base_feature ^ swizzle_feature_37)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_36 * 64 + (base_feature - 64 ^ swizzle_feature_37)) / 4] = packed_word;
                        }
                        int token_38 = base_token + 8 + 1;
                        float _mul_76;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_76) : "f"(_tmem_load_4[5]), "f"(sc));
                        converted[0] = _mul_76;
                        float _mul_77;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_77) : "f"(_tmem_load_4[7]), "f"(sc));
                        converted[1] = _mul_77;
                        float _mul_78;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_78) : "f"(_tmem_load_5[5]), "f"(sc));
                        converted[2] = _mul_78;
                        float _mul_79;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_79) : "f"(_tmem_load_5[7]), "f"(sc));
                        converted[3] = _mul_79;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_39 = token_38 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_38 * 64 + (base_feature ^ swizzle_feature_39)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_38 * 64 + (base_feature - 64 ^ swizzle_feature_39)) / 4] = packed_word;
                        }
                        int token_40 = base_token + 16;
                        float _mul_80;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_80) : "f"(_tmem_load_4[8]), "f"(sc));
                        converted[0] = _mul_80;
                        float _mul_81;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_81) : "f"(_tmem_load_4[10]), "f"(sc));
                        converted[1] = _mul_81;
                        float _mul_82;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_82) : "f"(_tmem_load_5[8]), "f"(sc));
                        converted[2] = _mul_82;
                        float _mul_83;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_83) : "f"(_tmem_load_5[10]), "f"(sc));
                        converted[3] = _mul_83;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_41 = token_40 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_40 * 64 + (base_feature ^ swizzle_feature_41)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_40 * 64 + (base_feature - 64 ^ swizzle_feature_41)) / 4] = packed_word;
                        }
                        int token_42 = base_token + 16 + 1;
                        float _mul_84;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_84) : "f"(_tmem_load_4[9]), "f"(sc));
                        converted[0] = _mul_84;
                        float _mul_85;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_85) : "f"(_tmem_load_4[11]), "f"(sc));
                        converted[1] = _mul_85;
                        float _mul_86;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_86) : "f"(_tmem_load_5[9]), "f"(sc));
                        converted[2] = _mul_86;
                        float _mul_87;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_87) : "f"(_tmem_load_5[11]), "f"(sc));
                        converted[3] = _mul_87;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_43 = token_42 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_42 * 64 + (base_feature ^ swizzle_feature_43)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_42 * 64 + (base_feature - 64 ^ swizzle_feature_43)) / 4] = packed_word;
                        }
                        int token_44 = base_token + 24;
                        float _mul_88;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_88) : "f"(_tmem_load_4[12]), "f"(sc));
                        converted[0] = _mul_88;
                        float _mul_89;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_89) : "f"(_tmem_load_4[14]), "f"(sc));
                        converted[1] = _mul_89;
                        float _mul_90;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_90) : "f"(_tmem_load_5[12]), "f"(sc));
                        converted[2] = _mul_90;
                        float _mul_91;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_91) : "f"(_tmem_load_5[14]), "f"(sc));
                        converted[3] = _mul_91;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_45 = token_44 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_44 * 64 + (base_feature ^ swizzle_feature_45)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_44 * 64 + (base_feature - 64 ^ swizzle_feature_45)) / 4] = packed_word;
                        }
                        int token_46 = base_token + 24 + 1;
                        float _mul_92;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_92) : "f"(_tmem_load_4[13]), "f"(sc));
                        converted[0] = _mul_92;
                        float _mul_93;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_93) : "f"(_tmem_load_4[15]), "f"(sc));
                        converted[1] = _mul_93;
                        float _mul_94;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_94) : "f"(_tmem_load_5[13]), "f"(sc));
                        converted[2] = _mul_94;
                        float _mul_95;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_95) : "f"(_tmem_load_5[15]), "f"(sc));
                        converted[3] = _mul_95;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_47 = token_46 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_46 * 64 + (base_feature ^ swizzle_feature_47)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_46 * 64 + (base_feature - 64 ^ swizzle_feature_47)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows_2 = (128 - valid_rows % 128) % 128;
                                tma_store_4d((&C), m_tile * 128, padding_rows_2 + 64, 1073741824, n_tile * 128 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows_2 + 64, 1073741824, n_tile * 128 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr + 4096);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        int acc_offset_48 = acc_stage * 128 + 96;
                        float _tmem_load_6[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                            : "r"(taddr + (unsigned int)acc_offset_48));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_7[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                            : "r"(taddr + 1048576 + (unsigned int)acc_offset_48));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        int token_49 = base_token;
                        float _mul_96;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_96) : "f"(_tmem_load_6[0]), "f"(sc));
                        converted[0] = _mul_96;
                        float _mul_97;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_97) : "f"(_tmem_load_6[2]), "f"(sc));
                        converted[1] = _mul_97;
                        float _mul_98;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_98) : "f"(_tmem_load_7[0]), "f"(sc));
                        converted[2] = _mul_98;
                        float _mul_99;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_99) : "f"(_tmem_load_7[2]), "f"(sc));
                        converted[3] = _mul_99;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_50 = token_49 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_49 * 64 + (base_feature ^ swizzle_feature_50)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_49 * 64 + (base_feature - 64 ^ swizzle_feature_50)) / 4] = packed_word;
                        }
                        int token_51 = base_token + 1;
                        float _mul_100;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_100) : "f"(_tmem_load_6[1]), "f"(sc));
                        converted[0] = _mul_100;
                        float _mul_101;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_101) : "f"(_tmem_load_6[3]), "f"(sc));
                        converted[1] = _mul_101;
                        float _mul_102;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_102) : "f"(_tmem_load_7[1]), "f"(sc));
                        converted[2] = _mul_102;
                        float _mul_103;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_103) : "f"(_tmem_load_7[3]), "f"(sc));
                        converted[3] = _mul_103;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_52 = token_51 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_51 * 64 + (base_feature ^ swizzle_feature_52)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_51 * 64 + (base_feature - 64 ^ swizzle_feature_52)) / 4] = packed_word;
                        }
                        int token_53 = base_token + 8;
                        float _mul_104;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_104) : "f"(_tmem_load_6[4]), "f"(sc));
                        converted[0] = _mul_104;
                        float _mul_105;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_105) : "f"(_tmem_load_6[6]), "f"(sc));
                        converted[1] = _mul_105;
                        float _mul_106;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_106) : "f"(_tmem_load_7[4]), "f"(sc));
                        converted[2] = _mul_106;
                        float _mul_107;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_107) : "f"(_tmem_load_7[6]), "f"(sc));
                        converted[3] = _mul_107;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_54 = token_53 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_53 * 64 + (base_feature ^ swizzle_feature_54)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_53 * 64 + (base_feature - 64 ^ swizzle_feature_54)) / 4] = packed_word;
                        }
                        int token_55 = base_token + 8 + 1;
                        float _mul_108;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_108) : "f"(_tmem_load_6[5]), "f"(sc));
                        converted[0] = _mul_108;
                        float _mul_109;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_109) : "f"(_tmem_load_6[7]), "f"(sc));
                        converted[1] = _mul_109;
                        float _mul_110;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_110) : "f"(_tmem_load_7[5]), "f"(sc));
                        converted[2] = _mul_110;
                        float _mul_111;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_111) : "f"(_tmem_load_7[7]), "f"(sc));
                        converted[3] = _mul_111;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_56 = token_55 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_55 * 64 + (base_feature ^ swizzle_feature_56)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_55 * 64 + (base_feature - 64 ^ swizzle_feature_56)) / 4] = packed_word;
                        }
                        int token_57 = base_token + 16;
                        float _mul_112;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_112) : "f"(_tmem_load_6[8]), "f"(sc));
                        converted[0] = _mul_112;
                        float _mul_113;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_113) : "f"(_tmem_load_6[10]), "f"(sc));
                        converted[1] = _mul_113;
                        float _mul_114;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_114) : "f"(_tmem_load_7[8]), "f"(sc));
                        converted[2] = _mul_114;
                        float _mul_115;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_115) : "f"(_tmem_load_7[10]), "f"(sc));
                        converted[3] = _mul_115;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_58 = token_57 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_57 * 64 + (base_feature ^ swizzle_feature_58)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_57 * 64 + (base_feature - 64 ^ swizzle_feature_58)) / 4] = packed_word;
                        }
                        int token_59 = base_token + 16 + 1;
                        float _mul_116;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_116) : "f"(_tmem_load_6[9]), "f"(sc));
                        converted[0] = _mul_116;
                        float _mul_117;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_117) : "f"(_tmem_load_6[11]), "f"(sc));
                        converted[1] = _mul_117;
                        float _mul_118;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_118) : "f"(_tmem_load_7[9]), "f"(sc));
                        converted[2] = _mul_118;
                        float _mul_119;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_119) : "f"(_tmem_load_7[11]), "f"(sc));
                        converted[3] = _mul_119;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_60 = token_59 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_59 * 64 + (base_feature ^ swizzle_feature_60)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_59 * 64 + (base_feature - 64 ^ swizzle_feature_60)) / 4] = packed_word;
                        }
                        int token_61 = base_token + 24;
                        float _mul_120;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_120) : "f"(_tmem_load_6[12]), "f"(sc));
                        converted[0] = _mul_120;
                        float _mul_121;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_121) : "f"(_tmem_load_6[14]), "f"(sc));
                        converted[1] = _mul_121;
                        float _mul_122;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_122) : "f"(_tmem_load_7[12]), "f"(sc));
                        converted[2] = _mul_122;
                        float _mul_123;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_123) : "f"(_tmem_load_7[14]), "f"(sc));
                        converted[3] = _mul_123;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_62 = token_61 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_61 * 64 + (base_feature ^ swizzle_feature_62)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_61 * 64 + (base_feature - 64 ^ swizzle_feature_62)) / 4] = packed_word;
                        }
                        int token_63 = base_token + 24 + 1;
                        float _mul_124;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_124) : "f"(_tmem_load_6[13]), "f"(sc));
                        converted[0] = _mul_124;
                        float _mul_125;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_125) : "f"(_tmem_load_6[15]), "f"(sc));
                        converted[1] = _mul_125;
                        float _mul_126;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_126) : "f"(_tmem_load_7[13]), "f"(sc));
                        converted[2] = _mul_126;
                        float _mul_127;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_127) : "f"(_tmem_load_7[15]), "f"(sc));
                        converted[3] = _mul_127;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_64 = token_63 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_63 * 64 + (base_feature ^ swizzle_feature_64)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(2048 + token_63 * 64 + (base_feature - 64 ^ swizzle_feature_64)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows_3 = (128 - valid_rows % 128) % 128;
                                tma_store_4d((&C), m_tile * 128, padding_rows_3 + 96, 1073741824, n_tile * 128 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows_3 + 96, 1073741824, n_tile * 128 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr + 4096);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((mma_free_addr + (acc_stage) * 8) & 0xFEFFFFFF) : "memory");
                        acc_stage += 1;
                        if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                unsigned int valid = 0;
                unsigned int next_x = 0;
                unsigned int next_y = 0;
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
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                valid = _clc_valid_3;
                uint32_t _clc_ctaid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_6)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                next_x = _clc_ctaid_6;
                uint32_t _clc_ctaid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_7)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                next_y = _clc_ctaid_7;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                work_stage += 1;
                if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                unsigned int valid_0 = valid;
                m_tile = next_x + (unsigned int)cta_rank;
                n_tile = next_y;
                if (valid_0 == 0) {
                    break;
                }
            }
            asm volatile("barrier.sync 7, 128;" ::: "memory");
            if (warp_0 == 0) {
                if (cta_rank == 0) {
                    mbarrier_wait(exit_bar_addr, 0);
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(exit_bar_addr + 0 * 8), "r"(1) : "memory");
                } else {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(exit_bar_addr + 0 * 8), "r"(0) : "memory");
                    mbarrier_wait(exit_bar_addr, 0);
                }
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int tile_count_1 = num_non_exiting_ctas[0];
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int stage = 0;
                unsigned int acc_stage_1 = 0;
                unsigned int work_stage_1 = 0;
                unsigned int m_tile_1 = blockIdx.x;
                unsigned int n_tile_1 = blockIdx.y;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m / 2 * grid_n; _tile_iter_1++) {
                    if (tile_count_1 > (int)n_tile_1) {
                        int valid_rows_m = (unsigned int)tile_mn_limit[n_tile_1] - n_tile_1 * 128;
                        if (valid_rows_m > 0) {
                            mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                            #pragma unroll 1
                            for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                                mbarrier_wait(a_full_addr + (stage) * 8, _phase_a_full);
                                mbarrier_wait(b_full_addr + (stage) * 8, _phase_b_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage) * 1024;
                                int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage) * 512;
                                mma_ss_step_cg2(_mma_a_lo_0, _mma_b_lo_0, (tmem_accum + (acc_stage_1 * 128)), 270532624, ((((1) ? ((iter_k == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                                int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage) * 1024;
                                int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage) * 512;
                                mma_ss_step_cg2(_mma_a_lo_1, _mma_b_lo_1, (tmem_accum + (acc_stage_1 * 128)), 270532624, ((((0) ? ((iter_k == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                                int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage) * 1024;
                                int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage) * 512;
                                mma_ss_step_cg2(_mma_a_lo_2, _mma_b_lo_2, (tmem_accum + (acc_stage_1 * 128)), 270532624, ((((0) ? ((iter_k == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                                int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage) * 1024;
                                int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage) * 512;
                                mma_ss_step_cg2(_mma_a_lo_3, _mma_b_lo_3, (tmem_accum + (acc_stage_1 * 128)), 270532624, ((((0) ? ((iter_k == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                                elect_commit_cg2_multicast(k_done_addr + (stage) * 8, (uint16_t)(3));
                                if (iter_k + 1 == K_tiles) {
                                    elect_commit_cg2_multicast(mma_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                                }
                                stage += 1;
                                if (stage == 7) { stage = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; }
                            }
                            acc_stage_1 += 1;
                            if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    unsigned int valid_1 = 0;
                    unsigned int next_x_1 = 0;
                    unsigned int next_y_1 = 0;
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
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    valid_1 = _clc_valid_2;
                    uint32_t _clc_ctaid_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_4)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    next_x_1 = _clc_ctaid_4;
                    uint32_t _clc_ctaid_5 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_5)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    next_y_1 = _clc_ctaid_5;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    unsigned int valid_0_1 = valid_1;
                    m_tile_1 = next_x_1 + (unsigned int)cta_rank;
                    n_tile_1 = next_y_1;
                    if (valid_0_1 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 5) {
        { // load_a_main
            int tile_count_2 = num_non_exiting_ctas[0];
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m / 2 * grid_n; _tile_iter_2++) {
                if (tile_count_2 > (int)n_tile_2) {
                    int valid_rows_a = (unsigned int)tile_mn_limit[n_tile_2] - n_tile_2 * 128;
                    if (valid_rows_a > 0) {
                        int expert = tile_expert[n_tile_2];
                        if (cta_rank == 0) {
                            mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                            mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                            throttle_stage += 1;
                            if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                        }
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                            mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done);
                            if (elect_sync()) {
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(smem_a_addr + stage_1 * 16384), "l"((&A)), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k_1), "r"(expert),
                                       "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask)) : "memory");
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            }
                            stage_1 += 1;
                            if (stage_1 == 7) { stage_1 = 0; _phase_k_done ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                unsigned int valid_2 = 0;
                unsigned int next_x_2 = 0;
                unsigned int next_y_2 = 0;
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                valid_2 = _clc_valid_0;
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_0)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                next_x_2 = _clc_ctaid_0;
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_1)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                next_y_2 = _clc_ctaid_1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                unsigned int valid_0_2 = valid_2;
                m_tile_2 = next_x_2 + (unsigned int)cta_rank;
                n_tile_2 = next_y_2;
                if (valid_0_2 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 6) {
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_3 = num_non_exiting_ctas[0];
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            unsigned int cta_mask_1 = 1 << cta_rank;
            unsigned int _phase_k_done_1 = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m / 2 * grid_n; _tile_iter_3++) {
                int valid_rows_1 = (unsigned int)tile_mn_limit[n_tile_3] - n_tile_3 * 128;
                if (tile_count_3 > (int)n_tile_3) {
                    if (valid_rows_1 > 0) {
                        int padding_rows_4 = (128 - valid_rows_1 % 128) % 128;
                        int box_row0 = cta_rank * 64 + padding_rows_4;
                        #pragma unroll 1
                        for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                            mbarrier_wait(k_done_addr + (stage_2) * 8, _phase_k_done_1);
                            if (elect_sync()) {
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(smem_b_addr + stage_2 * 8192), "l"((&B)), "r"(iter_k_2 * 128), "r"(box_row0), "r"(1073741824), "r"(n_tile_3 * 128 - (unsigned int)padding_rows_4 + 1073741824),
                                       "r"(((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)) : "memory");
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
                            }
                            stage_2 += 1;
                            if (stage_2 == 7) { stage_2 = 0; _phase_k_done_1 ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                unsigned int valid_3 = 0;
                unsigned int next_x_3 = 0;
                unsigned int next_y_3 = 0;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                valid_3 = _clc_valid_1;
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_x_3 = _clc_ctaid_2;
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_y_3 = _clc_ctaid_3;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_3 * 8), "r"(0) : "memory");
                work_stage_3 += 1;
                if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                unsigned int valid_0_3 = valid_3;
                m_tile_3 = next_x_3 + (unsigned int)cta_rank;
                n_tile_3 = next_y_3;
                if (valid_0_3 == 0) {
                    break;
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: work_id ----
    if (warp == 7) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_4 = num_non_exiting_ctas[0];
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int fast_stage = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_fast_ready = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < grid_m / 2 * grid_n; _tile_iter_4++) {
                    if (tile_count_4 > (int)n_tile_4) {
                        mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                        mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                        throttle_stage_1 += 1;
                        if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                    } else {
                        #pragma unroll 1
                        for (unsigned int _drain_iter = 0; _drain_iter < grid_m / 2 * grid_n; _drain_iter++) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(fast_ready_addr + (fast_stage) * 8, 64);
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_response_addr + fast_stage * 64 + 0 * 16), "r"(fast_ready_addr + fast_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_response_addr + fast_stage * 64 + 1 * 16), "r"(fast_ready_addr + fast_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_response_addr + fast_stage * 64 + 2 * 16), "r"(fast_ready_addr + fast_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_response_addr + fast_stage * 64 + 3 * 16), "r"(fast_ready_addr + fast_stage * 8)
                                    : "memory");
                            }
                            mbarrier_wait(fast_ready_addr + (fast_stage) * 8, _phase_fast_ready);
                            unsigned int canceled = 0;
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
                                : "r"(fast_response_addr + fast_stage * 64 + 0 * 16)
                                : "memory");
                            canceled += _clc_valid_4;
                            uint32_t _clc_valid_5 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "selp.u32 %0, 1, 0, p1;\n\t"
                                "}\n"
                                : "=r"(_clc_valid_5)
                                : "r"(fast_response_addr + fast_stage * 64 + 1 * 16)
                                : "memory");
                            canceled += _clc_valid_5;
                            uint32_t _clc_valid_6 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "selp.u32 %0, 1, 0, p1;\n\t"
                                "}\n"
                                : "=r"(_clc_valid_6)
                                : "r"(fast_response_addr + fast_stage * 64 + 2 * 16)
                                : "memory");
                            canceled += _clc_valid_6;
                            uint32_t _clc_valid_7 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "selp.u32 %0, 1, 0, p1;\n\t"
                                "}\n"
                                : "=r"(_clc_valid_7)
                                : "r"(fast_response_addr + fast_stage * 64 + 3 * 16)
                                : "memory");
                            canceled += _clc_valid_7;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            _phase_fast_ready ^= 1;
                            if (canceled == 0) {
                                break;
                            }
                        }
                    }
                    mbarrier_wait(work_empty_addr + (work_stage_4) * 8, _phase_work_empty);
                    if (lane < 2) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage_4 * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage_4 * 16 + 0 * 16), "r"(work_full_addr + work_stage_4 * 8)
                            : "memory");
                    }
                    mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
                    unsigned int valid_4 = 0;
                    unsigned int next_x_4 = 0;
                    unsigned int next_y_4 = 0;
                    uint32_t _clc_valid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_8)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    valid_4 = _clc_valid_8;
                    uint32_t _clc_ctaid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_8)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    next_x_4 = _clc_ctaid_8;
                    uint32_t _clc_ctaid_9 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_9)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    next_y_4 = _clc_ctaid_9;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_4 * 8), "r"(0) : "memory");
                    work_stage_4 += 1;
                    if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                    unsigned int valid_0_4 = valid_4;
                    m_tile_4 = next_x_4 + (unsigned int)cta_rank;
                    n_tile_4 = next_y_4;
                    if (valid_0_4 == 0) {
                        break;
                    }
                }
                #pragma unroll 1
                for (unsigned int _tail_iter = 0; _tail_iter < 3; _tail_iter++) {
                    mbarrier_wait(work_empty_addr + (work_stage_4) * 8, _phase_work_empty);
                    work_stage_4 += 1;
                    if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
