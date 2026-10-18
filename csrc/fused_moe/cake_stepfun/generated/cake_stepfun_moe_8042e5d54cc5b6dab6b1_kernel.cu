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
#define TMEM_NCOLS 496
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 448
#define TMEM_SFB_OFFSET 464
#define NUM_K_PIPE_STAGES 3
#define NUM_MMA_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_FAST_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 50176
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_EPI_STAGING_OFF 148480
#define SMEM_EPI_STAGING_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_STRIDE 16384
#define SMEM_EPI_STAGING_U64_OFF 148480
#define SMEM_EPI_STAGING_U64_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_U64_STRIDE 16384
#define SMEM_TOK_STAGE_OFF 164864
#define SMEM_TOK_STAGE_STAGE_BYTES 1024
#define SMEM_TOK_STAGE_STRIDE 1024
#define SMEM_SMEM_SFA_OFF 165888
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 2048
#define SMEM_SMEM_SFB_OFF 172032
#define SMEM_SMEM_SFB_STAGE_BYTES 4096
#define SMEM_SMEM_SFB_STRIDE 4096
#define SMEM_WORK_RESPONSE_OFF 184320
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_RESPONSE_OFF 184368
#define SMEM_FAST_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_RESPONSE_STRIDE 64
#define SMEM_TOTAL 184448
#define THREADS 384
#define BLOCK_N 256
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_stepfun_moe_8042e5d54cc5b6dab6b1(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, int M_out, int K, int grid_m, int grid_n, int K_tiles, float* __restrict__ token_sf)
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
    #define b_full_addr (mbar_base + 24)
    #define sfa_full_addr (mbar_base + 48)
    #define sfb_full_addr (mbar_base + 72)
    #define a_empty_addr (mbar_base + 96)
    #define b_empty_addr (mbar_base + 120)
    #define sfa_empty_addr (mbar_base + 144)
    #define sfb_empty_addr (mbar_base + 168)
    #define mma_full_addr (mbar_base + 192)
    #define mma_free_addr (mbar_base + 200)
    #define work_full_addr (mbar_base + 208)
    #define work_empty_addr (mbar_base + 232)
    #define throttle_full_addr (mbar_base + 256)
    #define throttle_empty_addr (mbar_base + 280)
    #define fast_ready_addr (mbar_base + 304)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_A_OFF);
    const int smem_a_addr = smem + SMEM_SMEM_A_OFF;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_B_OFF);
    const int smem_b_addr = smem + SMEM_SMEM_B_OFF;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_EPI_STAGING_OFF);
    const int epi_staging_addr = smem + SMEM_EPI_STAGING_OFF;
    unsigned long long* epi_staging_u64 = reinterpret_cast<unsigned long long*>(smem_raw + SMEM_EPI_STAGING_U64_OFF);
    const int epi_staging_u64_addr = smem + SMEM_EPI_STAGING_U64_OFF;
    float* tok_stage = reinterpret_cast<float*>(smem_raw + SMEM_TOK_STAGE_OFF);
    const int tok_stage_addr = smem + SMEM_TOK_STAGE_OFF;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFA_OFF);
    const int smem_sfa_addr = smem + SMEM_SMEM_SFA_OFF;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_SFB_OFF);
    const int smem_sfb_addr = smem + SMEM_SMEM_SFB_OFF;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WORK_RESPONSE_OFF);
    const int work_response_addr = smem + SMEM_WORK_RESPONSE_OFF;
    unsigned int* fast_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_FAST_RESPONSE_OFF);
    const int fast_response_addr = smem + SMEM_FAST_RESPONSE_OFF;

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 39 barriers)
    // Mbarriers at smem_raw[0..312)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // b_full: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // sfa_full: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // sfb_full: 3 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // a_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // b_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // sfa_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // sfb_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // mma_free: 1 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // work_empty: 3 barriers, init_count=384
            mbarrier_init(smem + 232, 384);
            mbarrier_init(smem + 240, 384);
            mbarrier_init(smem + 248, 384);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 256, 32);
            mbarrier_init(smem + 264, 32);
            mbarrier_init(smem + 272, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 280, 32);
            mbarrier_init(smem + 288, 32);
            mbarrier_init(smem + 296, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 496 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 312);
    if (warp == 0) {
        int _tmem_hold = smem + 312;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sfa = taddr + 448;
    const int tmem_sfb = taddr + 464;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count = num_non_exiting_ctas[0];
            const int warp_0 = warp;
            const int lane_1 = lane;
            int epi_thread = warp_0 * 32 + lane_1;
            unsigned int work_stage = 0;
            int epilogue_local_idx = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            int base_feature = warp_0 * 32 + lane_1 / 4 * 4;
            int base_token = lane_1 % 4 * 2;
            int row_addr = warp_0 * 32 << 16;
            float converted[4];
            unsigned int packed[2];
            unsigned long long packed_word = 0;
            float frag[64];
            unsigned int _phase_mma_full_0 = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m * grid_n; _tile_iter++) {
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 256;
                if (tile_count > (int)n_tile) {
                    if (valid_rows > 0) {
                        int expert_e = tile_expert[n_tile];
                        float sc = scale_c[expert_e];
                        int tok_base = (int)n_tile * 256;
                        if (epi_thread + tok_base < tile_mn_limit[n_tile]) {
                            tok_stage[epi_thread] = token_sf[tok_base + epi_thread];
                        }
                        if (epi_thread + 128 + tok_base < tile_mn_limit[n_tile]) {
                            tok_stage[epi_thread + 128] = token_sf[tok_base + 128 + epi_thread];
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        mbarrier_wait(mma_full_addr, _phase_mma_full_0);
                        _phase_mma_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int token_block = 0;
                        if (epilogue_local_idx == 0) {
                            token_block = 3;
                        }
                        int acc_col = epilogue_local_idx * 192 + token_block * 64;
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                            : "r"(taddr + (unsigned int)row_addr + (unsigned int)acc_col));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                            : "r"(taddr + (unsigned int)row_addr + 1048576 + (unsigned int)acc_col));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        for (int i = 0; i < 32; i++) {
                            frag[i] = _tmem_load_0[i];
                            frag[32 + i] = _tmem_load_1[i];
                        }
                        {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            mbarrier_arrive(mma_free_addr);
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token = base_token;
                        float tok = tok_stage[token_block * 64 + token];
                        float _mul_0;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(frag[0]), "f"(tok));
                        float _mul_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_mul_0), "f"(sc));
                        converted[0] = _mul_1;
                        float _mul_2;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(frag[2]), "f"(tok));
                        float _mul_3;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_mul_2), "f"(sc));
                        converted[1] = _mul_3;
                        float _mul_4;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(frag[32]), "f"(tok));
                        float _mul_5;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_mul_4), "f"(sc));
                        converted[2] = _mul_5;
                        float _mul_6;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(frag[34]), "f"(tok));
                        float _mul_7;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_mul_6), "f"(sc));
                        converted[3] = _mul_7;
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
                            epi_staging_u64[(4096 + token * 64 + (base_feature - 64 ^ swizzle_feature)) / 4] = packed_word;
                        }
                        int token_0 = base_token + 1;
                        float tok_1 = tok_stage[token_block * 64 + token_0];
                        float _mul_8;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(frag[1]), "f"(tok_1));
                        float _mul_9;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(_mul_8), "f"(sc));
                        converted[0] = _mul_9;
                        float _mul_10;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(frag[3]), "f"(tok_1));
                        float _mul_11;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(_mul_10), "f"(sc));
                        converted[1] = _mul_11;
                        float _mul_12;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(frag[33]), "f"(tok_1));
                        float _mul_13;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(_mul_12), "f"(sc));
                        converted[2] = _mul_13;
                        float _mul_14;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(frag[35]), "f"(tok_1));
                        float _mul_15;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(_mul_14), "f"(sc));
                        converted[3] = _mul_15;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_2 = token_0 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_0 * 64 + (base_feature ^ swizzle_feature_2)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_0 * 64 + (base_feature - 64 ^ swizzle_feature_2)) / 4] = packed_word;
                        }
                        int token_3 = base_token + 8;
                        float tok_4 = tok_stage[token_block * 64 + token_3];
                        float _mul_16;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(frag[4]), "f"(tok_4));
                        float _mul_17;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_mul_16), "f"(sc));
                        converted[0] = _mul_17;
                        float _mul_18;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(frag[6]), "f"(tok_4));
                        float _mul_19;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_mul_18), "f"(sc));
                        converted[1] = _mul_19;
                        float _mul_20;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(frag[36]), "f"(tok_4));
                        float _mul_21;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_mul_20), "f"(sc));
                        converted[2] = _mul_21;
                        float _mul_22;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(frag[38]), "f"(tok_4));
                        float _mul_23;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_mul_22), "f"(sc));
                        converted[3] = _mul_23;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_5 = token_3 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_3 * 64 + (base_feature ^ swizzle_feature_5)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_3 * 64 + (base_feature - 64 ^ swizzle_feature_5)) / 4] = packed_word;
                        }
                        int token_6 = base_token + 8 + 1;
                        float tok_7 = tok_stage[token_block * 64 + token_6];
                        float _mul_24;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(frag[5]), "f"(tok_7));
                        float _mul_25;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(_mul_24), "f"(sc));
                        converted[0] = _mul_25;
                        float _mul_26;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(frag[7]), "f"(tok_7));
                        float _mul_27;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(_mul_26), "f"(sc));
                        converted[1] = _mul_27;
                        float _mul_28;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(frag[37]), "f"(tok_7));
                        float _mul_29;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(_mul_28), "f"(sc));
                        converted[2] = _mul_29;
                        float _mul_30;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(frag[39]), "f"(tok_7));
                        float _mul_31;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(_mul_30), "f"(sc));
                        converted[3] = _mul_31;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_8 = token_6 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_6 * 64 + (base_feature ^ swizzle_feature_8)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_6 * 64 + (base_feature - 64 ^ swizzle_feature_8)) / 4] = packed_word;
                        }
                        int token_9 = base_token + 16;
                        float tok_10 = tok_stage[token_block * 64 + token_9];
                        float _mul_32;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(frag[8]), "f"(tok_10));
                        float _mul_33;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(_mul_32), "f"(sc));
                        converted[0] = _mul_33;
                        float _mul_34;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(frag[10]), "f"(tok_10));
                        float _mul_35;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(_mul_34), "f"(sc));
                        converted[1] = _mul_35;
                        float _mul_36;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(frag[40]), "f"(tok_10));
                        float _mul_37;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(_mul_36), "f"(sc));
                        converted[2] = _mul_37;
                        float _mul_38;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(frag[42]), "f"(tok_10));
                        float _mul_39;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(_mul_38), "f"(sc));
                        converted[3] = _mul_39;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_11 = token_9 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_9 * 64 + (base_feature ^ swizzle_feature_11)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_9 * 64 + (base_feature - 64 ^ swizzle_feature_11)) / 4] = packed_word;
                        }
                        int token_12 = base_token + 16 + 1;
                        float tok_13 = tok_stage[token_block * 64 + token_12];
                        float _mul_40;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(frag[9]), "f"(tok_13));
                        float _mul_41;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(_mul_40), "f"(sc));
                        converted[0] = _mul_41;
                        float _mul_42;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_42) : "f"(frag[11]), "f"(tok_13));
                        float _mul_43;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_43) : "f"(_mul_42), "f"(sc));
                        converted[1] = _mul_43;
                        float _mul_44;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_44) : "f"(frag[41]), "f"(tok_13));
                        float _mul_45;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_45) : "f"(_mul_44), "f"(sc));
                        converted[2] = _mul_45;
                        float _mul_46;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_46) : "f"(frag[43]), "f"(tok_13));
                        float _mul_47;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_47) : "f"(_mul_46), "f"(sc));
                        converted[3] = _mul_47;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_14 = token_12 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_12 * 64 + (base_feature ^ swizzle_feature_14)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_12 * 64 + (base_feature - 64 ^ swizzle_feature_14)) / 4] = packed_word;
                        }
                        int token_15 = base_token + 24;
                        float tok_16 = tok_stage[token_block * 64 + token_15];
                        float _mul_48;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_48) : "f"(frag[12]), "f"(tok_16));
                        float _mul_49;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_49) : "f"(_mul_48), "f"(sc));
                        converted[0] = _mul_49;
                        float _mul_50;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_50) : "f"(frag[14]), "f"(tok_16));
                        float _mul_51;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_51) : "f"(_mul_50), "f"(sc));
                        converted[1] = _mul_51;
                        float _mul_52;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_52) : "f"(frag[44]), "f"(tok_16));
                        float _mul_53;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_53) : "f"(_mul_52), "f"(sc));
                        converted[2] = _mul_53;
                        float _mul_54;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_54) : "f"(frag[46]), "f"(tok_16));
                        float _mul_55;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_55) : "f"(_mul_54), "f"(sc));
                        converted[3] = _mul_55;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_17 = token_15 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_15 * 64 + (base_feature ^ swizzle_feature_17)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_15 * 64 + (base_feature - 64 ^ swizzle_feature_17)) / 4] = packed_word;
                        }
                        int token_18 = base_token + 24 + 1;
                        float tok_19 = tok_stage[token_block * 64 + token_18];
                        float _mul_56;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_56) : "f"(frag[13]), "f"(tok_19));
                        float _mul_57;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_57) : "f"(_mul_56), "f"(sc));
                        converted[0] = _mul_57;
                        float _mul_58;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_58) : "f"(frag[15]), "f"(tok_19));
                        float _mul_59;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_59) : "f"(_mul_58), "f"(sc));
                        converted[1] = _mul_59;
                        float _mul_60;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_60) : "f"(frag[45]), "f"(tok_19));
                        float _mul_61;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_61) : "f"(_mul_60), "f"(sc));
                        converted[2] = _mul_61;
                        float _mul_62;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_62) : "f"(frag[47]), "f"(tok_19));
                        float _mul_63;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_63) : "f"(_mul_62), "f"(sc));
                        converted[3] = _mul_63;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_20 = token_18 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_18 * 64 + (base_feature ^ swizzle_feature_20)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_18 * 64 + (base_feature - 64 ^ swizzle_feature_20)) / 4] = packed_word;
                        }
                        int token_21 = base_token + 32;
                        float tok_22 = tok_stage[token_block * 64 + token_21];
                        float _mul_64;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_64) : "f"(frag[16]), "f"(tok_22));
                        float _mul_65;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_65) : "f"(_mul_64), "f"(sc));
                        converted[0] = _mul_65;
                        float _mul_66;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_66) : "f"(frag[18]), "f"(tok_22));
                        float _mul_67;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_67) : "f"(_mul_66), "f"(sc));
                        converted[1] = _mul_67;
                        float _mul_68;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_68) : "f"(frag[48]), "f"(tok_22));
                        float _mul_69;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_69) : "f"(_mul_68), "f"(sc));
                        converted[2] = _mul_69;
                        float _mul_70;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_70) : "f"(frag[50]), "f"(tok_22));
                        float _mul_71;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_71) : "f"(_mul_70), "f"(sc));
                        converted[3] = _mul_71;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_23 = token_21 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_21 * 64 + (base_feature ^ swizzle_feature_23)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_21 * 64 + (base_feature - 64 ^ swizzle_feature_23)) / 4] = packed_word;
                        }
                        int token_24 = base_token + 32 + 1;
                        float tok_25 = tok_stage[token_block * 64 + token_24];
                        float _mul_72;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_72) : "f"(frag[17]), "f"(tok_25));
                        float _mul_73;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_73) : "f"(_mul_72), "f"(sc));
                        converted[0] = _mul_73;
                        float _mul_74;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_74) : "f"(frag[19]), "f"(tok_25));
                        float _mul_75;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_75) : "f"(_mul_74), "f"(sc));
                        converted[1] = _mul_75;
                        float _mul_76;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_76) : "f"(frag[49]), "f"(tok_25));
                        float _mul_77;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_77) : "f"(_mul_76), "f"(sc));
                        converted[2] = _mul_77;
                        float _mul_78;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_78) : "f"(frag[51]), "f"(tok_25));
                        float _mul_79;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_79) : "f"(_mul_78), "f"(sc));
                        converted[3] = _mul_79;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_26 = token_24 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_24 * 64 + (base_feature ^ swizzle_feature_26)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_24 * 64 + (base_feature - 64 ^ swizzle_feature_26)) / 4] = packed_word;
                        }
                        int token_27 = base_token + 40;
                        float tok_28 = tok_stage[token_block * 64 + token_27];
                        float _mul_80;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_80) : "f"(frag[20]), "f"(tok_28));
                        float _mul_81;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_81) : "f"(_mul_80), "f"(sc));
                        converted[0] = _mul_81;
                        float _mul_82;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_82) : "f"(frag[22]), "f"(tok_28));
                        float _mul_83;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_83) : "f"(_mul_82), "f"(sc));
                        converted[1] = _mul_83;
                        float _mul_84;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_84) : "f"(frag[52]), "f"(tok_28));
                        float _mul_85;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_85) : "f"(_mul_84), "f"(sc));
                        converted[2] = _mul_85;
                        float _mul_86;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_86) : "f"(frag[54]), "f"(tok_28));
                        float _mul_87;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_87) : "f"(_mul_86), "f"(sc));
                        converted[3] = _mul_87;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_29 = token_27 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_27 * 64 + (base_feature ^ swizzle_feature_29)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_27 * 64 + (base_feature - 64 ^ swizzle_feature_29)) / 4] = packed_word;
                        }
                        int token_30 = base_token + 40 + 1;
                        float tok_31 = tok_stage[token_block * 64 + token_30];
                        float _mul_88;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_88) : "f"(frag[21]), "f"(tok_31));
                        float _mul_89;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_89) : "f"(_mul_88), "f"(sc));
                        converted[0] = _mul_89;
                        float _mul_90;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_90) : "f"(frag[23]), "f"(tok_31));
                        float _mul_91;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_91) : "f"(_mul_90), "f"(sc));
                        converted[1] = _mul_91;
                        float _mul_92;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_92) : "f"(frag[53]), "f"(tok_31));
                        float _mul_93;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_93) : "f"(_mul_92), "f"(sc));
                        converted[2] = _mul_93;
                        float _mul_94;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_94) : "f"(frag[55]), "f"(tok_31));
                        float _mul_95;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_95) : "f"(_mul_94), "f"(sc));
                        converted[3] = _mul_95;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_32 = token_30 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_30 * 64 + (base_feature ^ swizzle_feature_32)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_30 * 64 + (base_feature - 64 ^ swizzle_feature_32)) / 4] = packed_word;
                        }
                        int token_33 = base_token + 48;
                        float tok_34 = tok_stage[token_block * 64 + token_33];
                        float _mul_96;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_96) : "f"(frag[24]), "f"(tok_34));
                        float _mul_97;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_97) : "f"(_mul_96), "f"(sc));
                        converted[0] = _mul_97;
                        float _mul_98;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_98) : "f"(frag[26]), "f"(tok_34));
                        float _mul_99;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_99) : "f"(_mul_98), "f"(sc));
                        converted[1] = _mul_99;
                        float _mul_100;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_100) : "f"(frag[56]), "f"(tok_34));
                        float _mul_101;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_101) : "f"(_mul_100), "f"(sc));
                        converted[2] = _mul_101;
                        float _mul_102;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_102) : "f"(frag[58]), "f"(tok_34));
                        float _mul_103;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_103) : "f"(_mul_102), "f"(sc));
                        converted[3] = _mul_103;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_35 = token_33 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_33 * 64 + (base_feature ^ swizzle_feature_35)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_33 * 64 + (base_feature - 64 ^ swizzle_feature_35)) / 4] = packed_word;
                        }
                        int token_36 = base_token + 48 + 1;
                        float tok_37 = tok_stage[token_block * 64 + token_36];
                        float _mul_104;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_104) : "f"(frag[25]), "f"(tok_37));
                        float _mul_105;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_105) : "f"(_mul_104), "f"(sc));
                        converted[0] = _mul_105;
                        float _mul_106;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_106) : "f"(frag[27]), "f"(tok_37));
                        float _mul_107;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_107) : "f"(_mul_106), "f"(sc));
                        converted[1] = _mul_107;
                        float _mul_108;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_108) : "f"(frag[57]), "f"(tok_37));
                        float _mul_109;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_109) : "f"(_mul_108), "f"(sc));
                        converted[2] = _mul_109;
                        float _mul_110;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_110) : "f"(frag[59]), "f"(tok_37));
                        float _mul_111;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_111) : "f"(_mul_110), "f"(sc));
                        converted[3] = _mul_111;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_38 = token_36 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_36 * 64 + (base_feature ^ swizzle_feature_38)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_36 * 64 + (base_feature - 64 ^ swizzle_feature_38)) / 4] = packed_word;
                        }
                        int token_39 = base_token + 56;
                        float tok_40 = tok_stage[token_block * 64 + token_39];
                        float _mul_112;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_112) : "f"(frag[28]), "f"(tok_40));
                        float _mul_113;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_113) : "f"(_mul_112), "f"(sc));
                        converted[0] = _mul_113;
                        float _mul_114;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_114) : "f"(frag[30]), "f"(tok_40));
                        float _mul_115;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_115) : "f"(_mul_114), "f"(sc));
                        converted[1] = _mul_115;
                        float _mul_116;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_116) : "f"(frag[60]), "f"(tok_40));
                        float _mul_117;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_117) : "f"(_mul_116), "f"(sc));
                        converted[2] = _mul_117;
                        float _mul_118;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_118) : "f"(frag[62]), "f"(tok_40));
                        float _mul_119;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_119) : "f"(_mul_118), "f"(sc));
                        converted[3] = _mul_119;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_41 = token_39 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_39 * 64 + (base_feature ^ swizzle_feature_41)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_39 * 64 + (base_feature - 64 ^ swizzle_feature_41)) / 4] = packed_word;
                        }
                        int token_42 = base_token + 56 + 1;
                        float tok_43 = tok_stage[token_block * 64 + token_42];
                        float _mul_120;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_120) : "f"(frag[29]), "f"(tok_43));
                        float _mul_121;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_121) : "f"(_mul_120), "f"(sc));
                        converted[0] = _mul_121;
                        float _mul_122;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_122) : "f"(frag[31]), "f"(tok_43));
                        float _mul_123;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_123) : "f"(_mul_122), "f"(sc));
                        converted[1] = _mul_123;
                        float _mul_124;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_124) : "f"(frag[61]), "f"(tok_43));
                        float _mul_125;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_125) : "f"(_mul_124), "f"(sc));
                        converted[2] = _mul_125;
                        float _mul_126;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_126) : "f"(frag[63]), "f"(tok_43));
                        float _mul_127;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_127) : "f"(_mul_126), "f"(sc));
                        converted[3] = _mul_127;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_44 = token_42 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_42 * 64 + (base_feature ^ swizzle_feature_44)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_42 * 64 + (base_feature - 64 ^ swizzle_feature_44)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows = (256 - valid_rows % 256) % 256;
                                tma_store_4d((&C), m_tile * 128, padding_rows + token_block * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows + token_block * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows + 1073741824, epi_staging_addr + 8192);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_block_45 = 1;
                        if (epilogue_local_idx == 0) {
                            token_block_45 = 0;
                        }
                        int acc_col_46 = epilogue_local_idx * 192 + token_block_45 * 64;
                        float _tmem_load_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                            : "r"(taddr + (unsigned int)row_addr + (unsigned int)acc_col_46));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_3[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                            : "r"(taddr + (unsigned int)row_addr + 1048576 + (unsigned int)acc_col_46));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        for (int i_1 = 0; i_1 < 32; i_1++) {
                            frag[i_1] = _tmem_load_2[i_1];
                            frag[32 + i_1] = _tmem_load_3[i_1];
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_47 = base_token;
                        float tok_48 = tok_stage[token_block_45 * 64 + token_47];
                        float _mul_128;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_128) : "f"(frag[0]), "f"(tok_48));
                        float _mul_129;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_129) : "f"(_mul_128), "f"(sc));
                        converted[0] = _mul_129;
                        float _mul_130;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_130) : "f"(frag[2]), "f"(tok_48));
                        float _mul_131;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_131) : "f"(_mul_130), "f"(sc));
                        converted[1] = _mul_131;
                        float _mul_132;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_132) : "f"(frag[32]), "f"(tok_48));
                        float _mul_133;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_133) : "f"(_mul_132), "f"(sc));
                        converted[2] = _mul_133;
                        float _mul_134;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_134) : "f"(frag[34]), "f"(tok_48));
                        float _mul_135;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_135) : "f"(_mul_134), "f"(sc));
                        converted[3] = _mul_135;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_49 = token_47 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_47 * 64 + (base_feature ^ swizzle_feature_49)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_47 * 64 + (base_feature - 64 ^ swizzle_feature_49)) / 4] = packed_word;
                        }
                        int token_50 = base_token + 1;
                        float tok_51 = tok_stage[token_block_45 * 64 + token_50];
                        float _mul_136;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_136) : "f"(frag[1]), "f"(tok_51));
                        float _mul_137;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_137) : "f"(_mul_136), "f"(sc));
                        converted[0] = _mul_137;
                        float _mul_138;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_138) : "f"(frag[3]), "f"(tok_51));
                        float _mul_139;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_139) : "f"(_mul_138), "f"(sc));
                        converted[1] = _mul_139;
                        float _mul_140;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_140) : "f"(frag[33]), "f"(tok_51));
                        float _mul_141;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_141) : "f"(_mul_140), "f"(sc));
                        converted[2] = _mul_141;
                        float _mul_142;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_142) : "f"(frag[35]), "f"(tok_51));
                        float _mul_143;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_143) : "f"(_mul_142), "f"(sc));
                        converted[3] = _mul_143;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_52 = token_50 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_50 * 64 + (base_feature ^ swizzle_feature_52)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_50 * 64 + (base_feature - 64 ^ swizzle_feature_52)) / 4] = packed_word;
                        }
                        int token_53 = base_token + 8;
                        float tok_54 = tok_stage[token_block_45 * 64 + token_53];
                        float _mul_144;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_144) : "f"(frag[4]), "f"(tok_54));
                        float _mul_145;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_145) : "f"(_mul_144), "f"(sc));
                        converted[0] = _mul_145;
                        float _mul_146;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_146) : "f"(frag[6]), "f"(tok_54));
                        float _mul_147;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_147) : "f"(_mul_146), "f"(sc));
                        converted[1] = _mul_147;
                        float _mul_148;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_148) : "f"(frag[36]), "f"(tok_54));
                        float _mul_149;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_149) : "f"(_mul_148), "f"(sc));
                        converted[2] = _mul_149;
                        float _mul_150;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_150) : "f"(frag[38]), "f"(tok_54));
                        float _mul_151;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_151) : "f"(_mul_150), "f"(sc));
                        converted[3] = _mul_151;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_55 = token_53 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_53 * 64 + (base_feature ^ swizzle_feature_55)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_53 * 64 + (base_feature - 64 ^ swizzle_feature_55)) / 4] = packed_word;
                        }
                        int token_56 = base_token + 8 + 1;
                        float tok_57 = tok_stage[token_block_45 * 64 + token_56];
                        float _mul_152;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_152) : "f"(frag[5]), "f"(tok_57));
                        float _mul_153;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_153) : "f"(_mul_152), "f"(sc));
                        converted[0] = _mul_153;
                        float _mul_154;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_154) : "f"(frag[7]), "f"(tok_57));
                        float _mul_155;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_155) : "f"(_mul_154), "f"(sc));
                        converted[1] = _mul_155;
                        float _mul_156;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_156) : "f"(frag[37]), "f"(tok_57));
                        float _mul_157;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_157) : "f"(_mul_156), "f"(sc));
                        converted[2] = _mul_157;
                        float _mul_158;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_158) : "f"(frag[39]), "f"(tok_57));
                        float _mul_159;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_159) : "f"(_mul_158), "f"(sc));
                        converted[3] = _mul_159;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_58 = token_56 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_56 * 64 + (base_feature ^ swizzle_feature_58)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_56 * 64 + (base_feature - 64 ^ swizzle_feature_58)) / 4] = packed_word;
                        }
                        int token_59 = base_token + 16;
                        float tok_60 = tok_stage[token_block_45 * 64 + token_59];
                        float _mul_160;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_160) : "f"(frag[8]), "f"(tok_60));
                        float _mul_161;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_161) : "f"(_mul_160), "f"(sc));
                        converted[0] = _mul_161;
                        float _mul_162;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_162) : "f"(frag[10]), "f"(tok_60));
                        float _mul_163;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_163) : "f"(_mul_162), "f"(sc));
                        converted[1] = _mul_163;
                        float _mul_164;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_164) : "f"(frag[40]), "f"(tok_60));
                        float _mul_165;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_165) : "f"(_mul_164), "f"(sc));
                        converted[2] = _mul_165;
                        float _mul_166;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_166) : "f"(frag[42]), "f"(tok_60));
                        float _mul_167;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_167) : "f"(_mul_166), "f"(sc));
                        converted[3] = _mul_167;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_61 = token_59 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_59 * 64 + (base_feature ^ swizzle_feature_61)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_59 * 64 + (base_feature - 64 ^ swizzle_feature_61)) / 4] = packed_word;
                        }
                        int token_62 = base_token + 16 + 1;
                        float tok_63 = tok_stage[token_block_45 * 64 + token_62];
                        float _mul_168;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_168) : "f"(frag[9]), "f"(tok_63));
                        float _mul_169;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_169) : "f"(_mul_168), "f"(sc));
                        converted[0] = _mul_169;
                        float _mul_170;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_170) : "f"(frag[11]), "f"(tok_63));
                        float _mul_171;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_171) : "f"(_mul_170), "f"(sc));
                        converted[1] = _mul_171;
                        float _mul_172;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_172) : "f"(frag[41]), "f"(tok_63));
                        float _mul_173;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_173) : "f"(_mul_172), "f"(sc));
                        converted[2] = _mul_173;
                        float _mul_174;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_174) : "f"(frag[43]), "f"(tok_63));
                        float _mul_175;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_175) : "f"(_mul_174), "f"(sc));
                        converted[3] = _mul_175;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_64 = token_62 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_62 * 64 + (base_feature ^ swizzle_feature_64)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_62 * 64 + (base_feature - 64 ^ swizzle_feature_64)) / 4] = packed_word;
                        }
                        int token_65 = base_token + 24;
                        float tok_66 = tok_stage[token_block_45 * 64 + token_65];
                        float _mul_176;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_176) : "f"(frag[12]), "f"(tok_66));
                        float _mul_177;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_177) : "f"(_mul_176), "f"(sc));
                        converted[0] = _mul_177;
                        float _mul_178;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_178) : "f"(frag[14]), "f"(tok_66));
                        float _mul_179;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_179) : "f"(_mul_178), "f"(sc));
                        converted[1] = _mul_179;
                        float _mul_180;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_180) : "f"(frag[44]), "f"(tok_66));
                        float _mul_181;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_181) : "f"(_mul_180), "f"(sc));
                        converted[2] = _mul_181;
                        float _mul_182;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_182) : "f"(frag[46]), "f"(tok_66));
                        float _mul_183;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_183) : "f"(_mul_182), "f"(sc));
                        converted[3] = _mul_183;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_67 = token_65 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_65 * 64 + (base_feature ^ swizzle_feature_67)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_65 * 64 + (base_feature - 64 ^ swizzle_feature_67)) / 4] = packed_word;
                        }
                        int token_68 = base_token + 24 + 1;
                        float tok_69 = tok_stage[token_block_45 * 64 + token_68];
                        float _mul_184;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_184) : "f"(frag[13]), "f"(tok_69));
                        float _mul_185;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_185) : "f"(_mul_184), "f"(sc));
                        converted[0] = _mul_185;
                        float _mul_186;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_186) : "f"(frag[15]), "f"(tok_69));
                        float _mul_187;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_187) : "f"(_mul_186), "f"(sc));
                        converted[1] = _mul_187;
                        float _mul_188;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_188) : "f"(frag[45]), "f"(tok_69));
                        float _mul_189;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_189) : "f"(_mul_188), "f"(sc));
                        converted[2] = _mul_189;
                        float _mul_190;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_190) : "f"(frag[47]), "f"(tok_69));
                        float _mul_191;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_191) : "f"(_mul_190), "f"(sc));
                        converted[3] = _mul_191;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_70 = token_68 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_68 * 64 + (base_feature ^ swizzle_feature_70)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_68 * 64 + (base_feature - 64 ^ swizzle_feature_70)) / 4] = packed_word;
                        }
                        int token_71 = base_token + 32;
                        float tok_72 = tok_stage[token_block_45 * 64 + token_71];
                        float _mul_192;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_192) : "f"(frag[16]), "f"(tok_72));
                        float _mul_193;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_193) : "f"(_mul_192), "f"(sc));
                        converted[0] = _mul_193;
                        float _mul_194;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_194) : "f"(frag[18]), "f"(tok_72));
                        float _mul_195;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_195) : "f"(_mul_194), "f"(sc));
                        converted[1] = _mul_195;
                        float _mul_196;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_196) : "f"(frag[48]), "f"(tok_72));
                        float _mul_197;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_197) : "f"(_mul_196), "f"(sc));
                        converted[2] = _mul_197;
                        float _mul_198;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_198) : "f"(frag[50]), "f"(tok_72));
                        float _mul_199;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_199) : "f"(_mul_198), "f"(sc));
                        converted[3] = _mul_199;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_73 = token_71 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_71 * 64 + (base_feature ^ swizzle_feature_73)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_71 * 64 + (base_feature - 64 ^ swizzle_feature_73)) / 4] = packed_word;
                        }
                        int token_74 = base_token + 32 + 1;
                        float tok_75 = tok_stage[token_block_45 * 64 + token_74];
                        float _mul_200;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_200) : "f"(frag[17]), "f"(tok_75));
                        float _mul_201;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_201) : "f"(_mul_200), "f"(sc));
                        converted[0] = _mul_201;
                        float _mul_202;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_202) : "f"(frag[19]), "f"(tok_75));
                        float _mul_203;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_203) : "f"(_mul_202), "f"(sc));
                        converted[1] = _mul_203;
                        float _mul_204;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_204) : "f"(frag[49]), "f"(tok_75));
                        float _mul_205;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_205) : "f"(_mul_204), "f"(sc));
                        converted[2] = _mul_205;
                        float _mul_206;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_206) : "f"(frag[51]), "f"(tok_75));
                        float _mul_207;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_207) : "f"(_mul_206), "f"(sc));
                        converted[3] = _mul_207;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_76 = token_74 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_74 * 64 + (base_feature ^ swizzle_feature_76)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_74 * 64 + (base_feature - 64 ^ swizzle_feature_76)) / 4] = packed_word;
                        }
                        int token_77 = base_token + 40;
                        float tok_78 = tok_stage[token_block_45 * 64 + token_77];
                        float _mul_208;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_208) : "f"(frag[20]), "f"(tok_78));
                        float _mul_209;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_209) : "f"(_mul_208), "f"(sc));
                        converted[0] = _mul_209;
                        float _mul_210;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_210) : "f"(frag[22]), "f"(tok_78));
                        float _mul_211;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_211) : "f"(_mul_210), "f"(sc));
                        converted[1] = _mul_211;
                        float _mul_212;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_212) : "f"(frag[52]), "f"(tok_78));
                        float _mul_213;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_213) : "f"(_mul_212), "f"(sc));
                        converted[2] = _mul_213;
                        float _mul_214;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_214) : "f"(frag[54]), "f"(tok_78));
                        float _mul_215;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_215) : "f"(_mul_214), "f"(sc));
                        converted[3] = _mul_215;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_79 = token_77 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_77 * 64 + (base_feature ^ swizzle_feature_79)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_77 * 64 + (base_feature - 64 ^ swizzle_feature_79)) / 4] = packed_word;
                        }
                        int token_80 = base_token + 40 + 1;
                        float tok_81 = tok_stage[token_block_45 * 64 + token_80];
                        float _mul_216;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_216) : "f"(frag[21]), "f"(tok_81));
                        float _mul_217;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_217) : "f"(_mul_216), "f"(sc));
                        converted[0] = _mul_217;
                        float _mul_218;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_218) : "f"(frag[23]), "f"(tok_81));
                        float _mul_219;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_219) : "f"(_mul_218), "f"(sc));
                        converted[1] = _mul_219;
                        float _mul_220;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_220) : "f"(frag[53]), "f"(tok_81));
                        float _mul_221;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_221) : "f"(_mul_220), "f"(sc));
                        converted[2] = _mul_221;
                        float _mul_222;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_222) : "f"(frag[55]), "f"(tok_81));
                        float _mul_223;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_223) : "f"(_mul_222), "f"(sc));
                        converted[3] = _mul_223;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_82 = token_80 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_80 * 64 + (base_feature ^ swizzle_feature_82)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_80 * 64 + (base_feature - 64 ^ swizzle_feature_82)) / 4] = packed_word;
                        }
                        int token_83 = base_token + 48;
                        float tok_84 = tok_stage[token_block_45 * 64 + token_83];
                        float _mul_224;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_224) : "f"(frag[24]), "f"(tok_84));
                        float _mul_225;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_225) : "f"(_mul_224), "f"(sc));
                        converted[0] = _mul_225;
                        float _mul_226;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_226) : "f"(frag[26]), "f"(tok_84));
                        float _mul_227;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_227) : "f"(_mul_226), "f"(sc));
                        converted[1] = _mul_227;
                        float _mul_228;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_228) : "f"(frag[56]), "f"(tok_84));
                        float _mul_229;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_229) : "f"(_mul_228), "f"(sc));
                        converted[2] = _mul_229;
                        float _mul_230;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_230) : "f"(frag[58]), "f"(tok_84));
                        float _mul_231;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_231) : "f"(_mul_230), "f"(sc));
                        converted[3] = _mul_231;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_85 = token_83 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_83 * 64 + (base_feature ^ swizzle_feature_85)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_83 * 64 + (base_feature - 64 ^ swizzle_feature_85)) / 4] = packed_word;
                        }
                        int token_86 = base_token + 48 + 1;
                        float tok_87 = tok_stage[token_block_45 * 64 + token_86];
                        float _mul_232;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_232) : "f"(frag[25]), "f"(tok_87));
                        float _mul_233;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_233) : "f"(_mul_232), "f"(sc));
                        converted[0] = _mul_233;
                        float _mul_234;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_234) : "f"(frag[27]), "f"(tok_87));
                        float _mul_235;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_235) : "f"(_mul_234), "f"(sc));
                        converted[1] = _mul_235;
                        float _mul_236;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_236) : "f"(frag[57]), "f"(tok_87));
                        float _mul_237;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_237) : "f"(_mul_236), "f"(sc));
                        converted[2] = _mul_237;
                        float _mul_238;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_238) : "f"(frag[59]), "f"(tok_87));
                        float _mul_239;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_239) : "f"(_mul_238), "f"(sc));
                        converted[3] = _mul_239;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_88 = token_86 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_86 * 64 + (base_feature ^ swizzle_feature_88)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_86 * 64 + (base_feature - 64 ^ swizzle_feature_88)) / 4] = packed_word;
                        }
                        int token_89 = base_token + 56;
                        float tok_90 = tok_stage[token_block_45 * 64 + token_89];
                        float _mul_240;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_240) : "f"(frag[28]), "f"(tok_90));
                        float _mul_241;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_241) : "f"(_mul_240), "f"(sc));
                        converted[0] = _mul_241;
                        float _mul_242;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_242) : "f"(frag[30]), "f"(tok_90));
                        float _mul_243;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_243) : "f"(_mul_242), "f"(sc));
                        converted[1] = _mul_243;
                        float _mul_244;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_244) : "f"(frag[60]), "f"(tok_90));
                        float _mul_245;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_245) : "f"(_mul_244), "f"(sc));
                        converted[2] = _mul_245;
                        float _mul_246;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_246) : "f"(frag[62]), "f"(tok_90));
                        float _mul_247;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_247) : "f"(_mul_246), "f"(sc));
                        converted[3] = _mul_247;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_91 = token_89 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_89 * 64 + (base_feature ^ swizzle_feature_91)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_89 * 64 + (base_feature - 64 ^ swizzle_feature_91)) / 4] = packed_word;
                        }
                        int token_92 = base_token + 56 + 1;
                        float tok_93 = tok_stage[token_block_45 * 64 + token_92];
                        float _mul_248;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_248) : "f"(frag[29]), "f"(tok_93));
                        float _mul_249;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_249) : "f"(_mul_248), "f"(sc));
                        converted[0] = _mul_249;
                        float _mul_250;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_250) : "f"(frag[31]), "f"(tok_93));
                        float _mul_251;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_251) : "f"(_mul_250), "f"(sc));
                        converted[1] = _mul_251;
                        float _mul_252;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_252) : "f"(frag[61]), "f"(tok_93));
                        float _mul_253;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_253) : "f"(_mul_252), "f"(sc));
                        converted[2] = _mul_253;
                        float _mul_254;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_254) : "f"(frag[63]), "f"(tok_93));
                        float _mul_255;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_255) : "f"(_mul_254), "f"(sc));
                        converted[3] = _mul_255;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_94 = token_92 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_92 * 64 + (base_feature ^ swizzle_feature_94)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_92 * 64 + (base_feature - 64 ^ swizzle_feature_94)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows_1 = (256 - valid_rows % 256) % 256;
                                tma_store_4d((&C), m_tile * 128, padding_rows_1 + token_block_45 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows_1 + token_block_45 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr + 8192);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_block_95 = 2;
                        if (epilogue_local_idx == 0) {
                            token_block_95 = 1;
                        }
                        int acc_col_96 = epilogue_local_idx * 192 + token_block_95 * 64;
                        float _tmem_load_4[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                            : "r"(taddr + (unsigned int)row_addr + (unsigned int)acc_col_96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_5[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[31]))
                            : "r"(taddr + (unsigned int)row_addr + 1048576 + (unsigned int)acc_col_96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        for (int i_2 = 0; i_2 < 32; i_2++) {
                            frag[i_2] = _tmem_load_4[i_2];
                            frag[32 + i_2] = _tmem_load_5[i_2];
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_97 = base_token;
                        float tok_98 = tok_stage[token_block_95 * 64 + token_97];
                        float _mul_256;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_256) : "f"(frag[0]), "f"(tok_98));
                        float _mul_257;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_257) : "f"(_mul_256), "f"(sc));
                        converted[0] = _mul_257;
                        float _mul_258;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_258) : "f"(frag[2]), "f"(tok_98));
                        float _mul_259;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_259) : "f"(_mul_258), "f"(sc));
                        converted[1] = _mul_259;
                        float _mul_260;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_260) : "f"(frag[32]), "f"(tok_98));
                        float _mul_261;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_261) : "f"(_mul_260), "f"(sc));
                        converted[2] = _mul_261;
                        float _mul_262;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_262) : "f"(frag[34]), "f"(tok_98));
                        float _mul_263;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_263) : "f"(_mul_262), "f"(sc));
                        converted[3] = _mul_263;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_99 = token_97 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_97 * 64 + (base_feature ^ swizzle_feature_99)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_97 * 64 + (base_feature - 64 ^ swizzle_feature_99)) / 4] = packed_word;
                        }
                        int token_100 = base_token + 1;
                        float tok_101 = tok_stage[token_block_95 * 64 + token_100];
                        float _mul_264;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_264) : "f"(frag[1]), "f"(tok_101));
                        float _mul_265;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_265) : "f"(_mul_264), "f"(sc));
                        converted[0] = _mul_265;
                        float _mul_266;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_266) : "f"(frag[3]), "f"(tok_101));
                        float _mul_267;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_267) : "f"(_mul_266), "f"(sc));
                        converted[1] = _mul_267;
                        float _mul_268;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_268) : "f"(frag[33]), "f"(tok_101));
                        float _mul_269;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_269) : "f"(_mul_268), "f"(sc));
                        converted[2] = _mul_269;
                        float _mul_270;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_270) : "f"(frag[35]), "f"(tok_101));
                        float _mul_271;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_271) : "f"(_mul_270), "f"(sc));
                        converted[3] = _mul_271;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_102 = token_100 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_100 * 64 + (base_feature ^ swizzle_feature_102)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_100 * 64 + (base_feature - 64 ^ swizzle_feature_102)) / 4] = packed_word;
                        }
                        int token_103 = base_token + 8;
                        float tok_104 = tok_stage[token_block_95 * 64 + token_103];
                        float _mul_272;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_272) : "f"(frag[4]), "f"(tok_104));
                        float _mul_273;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_273) : "f"(_mul_272), "f"(sc));
                        converted[0] = _mul_273;
                        float _mul_274;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_274) : "f"(frag[6]), "f"(tok_104));
                        float _mul_275;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_275) : "f"(_mul_274), "f"(sc));
                        converted[1] = _mul_275;
                        float _mul_276;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_276) : "f"(frag[36]), "f"(tok_104));
                        float _mul_277;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_277) : "f"(_mul_276), "f"(sc));
                        converted[2] = _mul_277;
                        float _mul_278;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_278) : "f"(frag[38]), "f"(tok_104));
                        float _mul_279;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_279) : "f"(_mul_278), "f"(sc));
                        converted[3] = _mul_279;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_105 = token_103 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_103 * 64 + (base_feature ^ swizzle_feature_105)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_103 * 64 + (base_feature - 64 ^ swizzle_feature_105)) / 4] = packed_word;
                        }
                        int token_106 = base_token + 8 + 1;
                        float tok_107 = tok_stage[token_block_95 * 64 + token_106];
                        float _mul_280;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_280) : "f"(frag[5]), "f"(tok_107));
                        float _mul_281;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_281) : "f"(_mul_280), "f"(sc));
                        converted[0] = _mul_281;
                        float _mul_282;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_282) : "f"(frag[7]), "f"(tok_107));
                        float _mul_283;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_283) : "f"(_mul_282), "f"(sc));
                        converted[1] = _mul_283;
                        float _mul_284;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_284) : "f"(frag[37]), "f"(tok_107));
                        float _mul_285;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_285) : "f"(_mul_284), "f"(sc));
                        converted[2] = _mul_285;
                        float _mul_286;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_286) : "f"(frag[39]), "f"(tok_107));
                        float _mul_287;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_287) : "f"(_mul_286), "f"(sc));
                        converted[3] = _mul_287;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_108 = token_106 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_106 * 64 + (base_feature ^ swizzle_feature_108)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_106 * 64 + (base_feature - 64 ^ swizzle_feature_108)) / 4] = packed_word;
                        }
                        int token_109 = base_token + 16;
                        float tok_110 = tok_stage[token_block_95 * 64 + token_109];
                        float _mul_288;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_288) : "f"(frag[8]), "f"(tok_110));
                        float _mul_289;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_289) : "f"(_mul_288), "f"(sc));
                        converted[0] = _mul_289;
                        float _mul_290;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_290) : "f"(frag[10]), "f"(tok_110));
                        float _mul_291;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_291) : "f"(_mul_290), "f"(sc));
                        converted[1] = _mul_291;
                        float _mul_292;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_292) : "f"(frag[40]), "f"(tok_110));
                        float _mul_293;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_293) : "f"(_mul_292), "f"(sc));
                        converted[2] = _mul_293;
                        float _mul_294;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_294) : "f"(frag[42]), "f"(tok_110));
                        float _mul_295;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_295) : "f"(_mul_294), "f"(sc));
                        converted[3] = _mul_295;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_111 = token_109 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_109 * 64 + (base_feature ^ swizzle_feature_111)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_109 * 64 + (base_feature - 64 ^ swizzle_feature_111)) / 4] = packed_word;
                        }
                        int token_112 = base_token + 16 + 1;
                        float tok_113 = tok_stage[token_block_95 * 64 + token_112];
                        float _mul_296;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_296) : "f"(frag[9]), "f"(tok_113));
                        float _mul_297;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_297) : "f"(_mul_296), "f"(sc));
                        converted[0] = _mul_297;
                        float _mul_298;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_298) : "f"(frag[11]), "f"(tok_113));
                        float _mul_299;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_299) : "f"(_mul_298), "f"(sc));
                        converted[1] = _mul_299;
                        float _mul_300;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_300) : "f"(frag[41]), "f"(tok_113));
                        float _mul_301;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_301) : "f"(_mul_300), "f"(sc));
                        converted[2] = _mul_301;
                        float _mul_302;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_302) : "f"(frag[43]), "f"(tok_113));
                        float _mul_303;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_303) : "f"(_mul_302), "f"(sc));
                        converted[3] = _mul_303;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_114 = token_112 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_112 * 64 + (base_feature ^ swizzle_feature_114)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_112 * 64 + (base_feature - 64 ^ swizzle_feature_114)) / 4] = packed_word;
                        }
                        int token_115 = base_token + 24;
                        float tok_116 = tok_stage[token_block_95 * 64 + token_115];
                        float _mul_304;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_304) : "f"(frag[12]), "f"(tok_116));
                        float _mul_305;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_305) : "f"(_mul_304), "f"(sc));
                        converted[0] = _mul_305;
                        float _mul_306;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_306) : "f"(frag[14]), "f"(tok_116));
                        float _mul_307;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_307) : "f"(_mul_306), "f"(sc));
                        converted[1] = _mul_307;
                        float _mul_308;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_308) : "f"(frag[44]), "f"(tok_116));
                        float _mul_309;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_309) : "f"(_mul_308), "f"(sc));
                        converted[2] = _mul_309;
                        float _mul_310;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_310) : "f"(frag[46]), "f"(tok_116));
                        float _mul_311;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_311) : "f"(_mul_310), "f"(sc));
                        converted[3] = _mul_311;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_117 = token_115 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_115 * 64 + (base_feature ^ swizzle_feature_117)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_115 * 64 + (base_feature - 64 ^ swizzle_feature_117)) / 4] = packed_word;
                        }
                        int token_118 = base_token + 24 + 1;
                        float tok_119 = tok_stage[token_block_95 * 64 + token_118];
                        float _mul_312;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_312) : "f"(frag[13]), "f"(tok_119));
                        float _mul_313;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_313) : "f"(_mul_312), "f"(sc));
                        converted[0] = _mul_313;
                        float _mul_314;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_314) : "f"(frag[15]), "f"(tok_119));
                        float _mul_315;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_315) : "f"(_mul_314), "f"(sc));
                        converted[1] = _mul_315;
                        float _mul_316;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_316) : "f"(frag[45]), "f"(tok_119));
                        float _mul_317;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_317) : "f"(_mul_316), "f"(sc));
                        converted[2] = _mul_317;
                        float _mul_318;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_318) : "f"(frag[47]), "f"(tok_119));
                        float _mul_319;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_319) : "f"(_mul_318), "f"(sc));
                        converted[3] = _mul_319;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_120 = token_118 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_118 * 64 + (base_feature ^ swizzle_feature_120)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_118 * 64 + (base_feature - 64 ^ swizzle_feature_120)) / 4] = packed_word;
                        }
                        int token_121 = base_token + 32;
                        float tok_122 = tok_stage[token_block_95 * 64 + token_121];
                        float _mul_320;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_320) : "f"(frag[16]), "f"(tok_122));
                        float _mul_321;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_321) : "f"(_mul_320), "f"(sc));
                        converted[0] = _mul_321;
                        float _mul_322;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_322) : "f"(frag[18]), "f"(tok_122));
                        float _mul_323;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_323) : "f"(_mul_322), "f"(sc));
                        converted[1] = _mul_323;
                        float _mul_324;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_324) : "f"(frag[48]), "f"(tok_122));
                        float _mul_325;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_325) : "f"(_mul_324), "f"(sc));
                        converted[2] = _mul_325;
                        float _mul_326;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_326) : "f"(frag[50]), "f"(tok_122));
                        float _mul_327;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_327) : "f"(_mul_326), "f"(sc));
                        converted[3] = _mul_327;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_123 = token_121 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_121 * 64 + (base_feature ^ swizzle_feature_123)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_121 * 64 + (base_feature - 64 ^ swizzle_feature_123)) / 4] = packed_word;
                        }
                        int token_124 = base_token + 32 + 1;
                        float tok_125 = tok_stage[token_block_95 * 64 + token_124];
                        float _mul_328;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_328) : "f"(frag[17]), "f"(tok_125));
                        float _mul_329;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_329) : "f"(_mul_328), "f"(sc));
                        converted[0] = _mul_329;
                        float _mul_330;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_330) : "f"(frag[19]), "f"(tok_125));
                        float _mul_331;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_331) : "f"(_mul_330), "f"(sc));
                        converted[1] = _mul_331;
                        float _mul_332;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_332) : "f"(frag[49]), "f"(tok_125));
                        float _mul_333;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_333) : "f"(_mul_332), "f"(sc));
                        converted[2] = _mul_333;
                        float _mul_334;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_334) : "f"(frag[51]), "f"(tok_125));
                        float _mul_335;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_335) : "f"(_mul_334), "f"(sc));
                        converted[3] = _mul_335;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_126 = token_124 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_124 * 64 + (base_feature ^ swizzle_feature_126)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_124 * 64 + (base_feature - 64 ^ swizzle_feature_126)) / 4] = packed_word;
                        }
                        int token_127 = base_token + 40;
                        float tok_128 = tok_stage[token_block_95 * 64 + token_127];
                        float _mul_336;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_336) : "f"(frag[20]), "f"(tok_128));
                        float _mul_337;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_337) : "f"(_mul_336), "f"(sc));
                        converted[0] = _mul_337;
                        float _mul_338;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_338) : "f"(frag[22]), "f"(tok_128));
                        float _mul_339;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_339) : "f"(_mul_338), "f"(sc));
                        converted[1] = _mul_339;
                        float _mul_340;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_340) : "f"(frag[52]), "f"(tok_128));
                        float _mul_341;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_341) : "f"(_mul_340), "f"(sc));
                        converted[2] = _mul_341;
                        float _mul_342;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_342) : "f"(frag[54]), "f"(tok_128));
                        float _mul_343;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_343) : "f"(_mul_342), "f"(sc));
                        converted[3] = _mul_343;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_129 = token_127 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_127 * 64 + (base_feature ^ swizzle_feature_129)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_127 * 64 + (base_feature - 64 ^ swizzle_feature_129)) / 4] = packed_word;
                        }
                        int token_130 = base_token + 40 + 1;
                        float tok_131 = tok_stage[token_block_95 * 64 + token_130];
                        float _mul_344;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_344) : "f"(frag[21]), "f"(tok_131));
                        float _mul_345;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_345) : "f"(_mul_344), "f"(sc));
                        converted[0] = _mul_345;
                        float _mul_346;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_346) : "f"(frag[23]), "f"(tok_131));
                        float _mul_347;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_347) : "f"(_mul_346), "f"(sc));
                        converted[1] = _mul_347;
                        float _mul_348;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_348) : "f"(frag[53]), "f"(tok_131));
                        float _mul_349;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_349) : "f"(_mul_348), "f"(sc));
                        converted[2] = _mul_349;
                        float _mul_350;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_350) : "f"(frag[55]), "f"(tok_131));
                        float _mul_351;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_351) : "f"(_mul_350), "f"(sc));
                        converted[3] = _mul_351;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_132 = token_130 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_130 * 64 + (base_feature ^ swizzle_feature_132)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_130 * 64 + (base_feature - 64 ^ swizzle_feature_132)) / 4] = packed_word;
                        }
                        int token_133 = base_token + 48;
                        float tok_134 = tok_stage[token_block_95 * 64 + token_133];
                        float _mul_352;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_352) : "f"(frag[24]), "f"(tok_134));
                        float _mul_353;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_353) : "f"(_mul_352), "f"(sc));
                        converted[0] = _mul_353;
                        float _mul_354;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_354) : "f"(frag[26]), "f"(tok_134));
                        float _mul_355;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_355) : "f"(_mul_354), "f"(sc));
                        converted[1] = _mul_355;
                        float _mul_356;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_356) : "f"(frag[56]), "f"(tok_134));
                        float _mul_357;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_357) : "f"(_mul_356), "f"(sc));
                        converted[2] = _mul_357;
                        float _mul_358;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_358) : "f"(frag[58]), "f"(tok_134));
                        float _mul_359;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_359) : "f"(_mul_358), "f"(sc));
                        converted[3] = _mul_359;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_135 = token_133 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_133 * 64 + (base_feature ^ swizzle_feature_135)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_133 * 64 + (base_feature - 64 ^ swizzle_feature_135)) / 4] = packed_word;
                        }
                        int token_136 = base_token + 48 + 1;
                        float tok_137 = tok_stage[token_block_95 * 64 + token_136];
                        float _mul_360;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_360) : "f"(frag[25]), "f"(tok_137));
                        float _mul_361;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_361) : "f"(_mul_360), "f"(sc));
                        converted[0] = _mul_361;
                        float _mul_362;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_362) : "f"(frag[27]), "f"(tok_137));
                        float _mul_363;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_363) : "f"(_mul_362), "f"(sc));
                        converted[1] = _mul_363;
                        float _mul_364;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_364) : "f"(frag[57]), "f"(tok_137));
                        float _mul_365;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_365) : "f"(_mul_364), "f"(sc));
                        converted[2] = _mul_365;
                        float _mul_366;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_366) : "f"(frag[59]), "f"(tok_137));
                        float _mul_367;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_367) : "f"(_mul_366), "f"(sc));
                        converted[3] = _mul_367;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_138 = token_136 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_136 * 64 + (base_feature ^ swizzle_feature_138)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_136 * 64 + (base_feature - 64 ^ swizzle_feature_138)) / 4] = packed_word;
                        }
                        int token_139 = base_token + 56;
                        float tok_140 = tok_stage[token_block_95 * 64 + token_139];
                        float _mul_368;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_368) : "f"(frag[28]), "f"(tok_140));
                        float _mul_369;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_369) : "f"(_mul_368), "f"(sc));
                        converted[0] = _mul_369;
                        float _mul_370;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_370) : "f"(frag[30]), "f"(tok_140));
                        float _mul_371;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_371) : "f"(_mul_370), "f"(sc));
                        converted[1] = _mul_371;
                        float _mul_372;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_372) : "f"(frag[60]), "f"(tok_140));
                        float _mul_373;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_373) : "f"(_mul_372), "f"(sc));
                        converted[2] = _mul_373;
                        float _mul_374;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_374) : "f"(frag[62]), "f"(tok_140));
                        float _mul_375;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_375) : "f"(_mul_374), "f"(sc));
                        converted[3] = _mul_375;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_141 = token_139 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_139 * 64 + (base_feature ^ swizzle_feature_141)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_139 * 64 + (base_feature - 64 ^ swizzle_feature_141)) / 4] = packed_word;
                        }
                        int token_142 = base_token + 56 + 1;
                        float tok_143 = tok_stage[token_block_95 * 64 + token_142];
                        float _mul_376;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_376) : "f"(frag[29]), "f"(tok_143));
                        float _mul_377;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_377) : "f"(_mul_376), "f"(sc));
                        converted[0] = _mul_377;
                        float _mul_378;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_378) : "f"(frag[31]), "f"(tok_143));
                        float _mul_379;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_379) : "f"(_mul_378), "f"(sc));
                        converted[1] = _mul_379;
                        float _mul_380;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_380) : "f"(frag[61]), "f"(tok_143));
                        float _mul_381;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_381) : "f"(_mul_380), "f"(sc));
                        converted[2] = _mul_381;
                        float _mul_382;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_382) : "f"(frag[63]), "f"(tok_143));
                        float _mul_383;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_383) : "f"(_mul_382), "f"(sc));
                        converted[3] = _mul_383;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_144 = token_142 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_142 * 64 + (base_feature ^ swizzle_feature_144)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_142 * 64 + (base_feature - 64 ^ swizzle_feature_144)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows_2 = (256 - valid_rows % 256) % 256;
                                tma_store_4d((&C), m_tile * 128, padding_rows_2 + token_block_95 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows_2 + token_block_95 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr + 8192);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_block_145 = 3;
                        if (epilogue_local_idx == 0) {
                            token_block_145 = 2;
                        }
                        int acc_col_146 = epilogue_local_idx * 192 + token_block_145 * 64;
                        float _tmem_load_6[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[31]))
                            : "r"(taddr + (unsigned int)row_addr + (unsigned int)acc_col_146));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_7[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[31]))
                            : "r"(taddr + (unsigned int)row_addr + 1048576 + (unsigned int)acc_col_146));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        for (int i_3 = 0; i_3 < 32; i_3++) {
                            frag[i_3] = _tmem_load_6[i_3];
                            frag[32 + i_3] = _tmem_load_7[i_3];
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_147 = base_token;
                        float tok_148 = tok_stage[token_block_145 * 64 + token_147];
                        float _mul_384;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_384) : "f"(frag[0]), "f"(tok_148));
                        float _mul_385;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_385) : "f"(_mul_384), "f"(sc));
                        converted[0] = _mul_385;
                        float _mul_386;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_386) : "f"(frag[2]), "f"(tok_148));
                        float _mul_387;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_387) : "f"(_mul_386), "f"(sc));
                        converted[1] = _mul_387;
                        float _mul_388;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_388) : "f"(frag[32]), "f"(tok_148));
                        float _mul_389;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_389) : "f"(_mul_388), "f"(sc));
                        converted[2] = _mul_389;
                        float _mul_390;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_390) : "f"(frag[34]), "f"(tok_148));
                        float _mul_391;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_391) : "f"(_mul_390), "f"(sc));
                        converted[3] = _mul_391;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_149 = token_147 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_147 * 64 + (base_feature ^ swizzle_feature_149)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_147 * 64 + (base_feature - 64 ^ swizzle_feature_149)) / 4] = packed_word;
                        }
                        int token_150 = base_token + 1;
                        float tok_151 = tok_stage[token_block_145 * 64 + token_150];
                        float _mul_392;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_392) : "f"(frag[1]), "f"(tok_151));
                        float _mul_393;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_393) : "f"(_mul_392), "f"(sc));
                        converted[0] = _mul_393;
                        float _mul_394;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_394) : "f"(frag[3]), "f"(tok_151));
                        float _mul_395;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_395) : "f"(_mul_394), "f"(sc));
                        converted[1] = _mul_395;
                        float _mul_396;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_396) : "f"(frag[33]), "f"(tok_151));
                        float _mul_397;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_397) : "f"(_mul_396), "f"(sc));
                        converted[2] = _mul_397;
                        float _mul_398;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_398) : "f"(frag[35]), "f"(tok_151));
                        float _mul_399;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_399) : "f"(_mul_398), "f"(sc));
                        converted[3] = _mul_399;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_152 = token_150 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_150 * 64 + (base_feature ^ swizzle_feature_152)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_150 * 64 + (base_feature - 64 ^ swizzle_feature_152)) / 4] = packed_word;
                        }
                        int token_153 = base_token + 8;
                        float tok_154 = tok_stage[token_block_145 * 64 + token_153];
                        float _mul_400;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_400) : "f"(frag[4]), "f"(tok_154));
                        float _mul_401;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_401) : "f"(_mul_400), "f"(sc));
                        converted[0] = _mul_401;
                        float _mul_402;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_402) : "f"(frag[6]), "f"(tok_154));
                        float _mul_403;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_403) : "f"(_mul_402), "f"(sc));
                        converted[1] = _mul_403;
                        float _mul_404;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_404) : "f"(frag[36]), "f"(tok_154));
                        float _mul_405;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_405) : "f"(_mul_404), "f"(sc));
                        converted[2] = _mul_405;
                        float _mul_406;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_406) : "f"(frag[38]), "f"(tok_154));
                        float _mul_407;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_407) : "f"(_mul_406), "f"(sc));
                        converted[3] = _mul_407;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_155 = token_153 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_153 * 64 + (base_feature ^ swizzle_feature_155)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_153 * 64 + (base_feature - 64 ^ swizzle_feature_155)) / 4] = packed_word;
                        }
                        int token_156 = base_token + 8 + 1;
                        float tok_157 = tok_stage[token_block_145 * 64 + token_156];
                        float _mul_408;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_408) : "f"(frag[5]), "f"(tok_157));
                        float _mul_409;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_409) : "f"(_mul_408), "f"(sc));
                        converted[0] = _mul_409;
                        float _mul_410;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_410) : "f"(frag[7]), "f"(tok_157));
                        float _mul_411;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_411) : "f"(_mul_410), "f"(sc));
                        converted[1] = _mul_411;
                        float _mul_412;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_412) : "f"(frag[37]), "f"(tok_157));
                        float _mul_413;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_413) : "f"(_mul_412), "f"(sc));
                        converted[2] = _mul_413;
                        float _mul_414;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_414) : "f"(frag[39]), "f"(tok_157));
                        float _mul_415;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_415) : "f"(_mul_414), "f"(sc));
                        converted[3] = _mul_415;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_158 = token_156 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_156 * 64 + (base_feature ^ swizzle_feature_158)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_156 * 64 + (base_feature - 64 ^ swizzle_feature_158)) / 4] = packed_word;
                        }
                        int token_159 = base_token + 16;
                        float tok_160 = tok_stage[token_block_145 * 64 + token_159];
                        float _mul_416;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_416) : "f"(frag[8]), "f"(tok_160));
                        float _mul_417;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_417) : "f"(_mul_416), "f"(sc));
                        converted[0] = _mul_417;
                        float _mul_418;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_418) : "f"(frag[10]), "f"(tok_160));
                        float _mul_419;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_419) : "f"(_mul_418), "f"(sc));
                        converted[1] = _mul_419;
                        float _mul_420;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_420) : "f"(frag[40]), "f"(tok_160));
                        float _mul_421;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_421) : "f"(_mul_420), "f"(sc));
                        converted[2] = _mul_421;
                        float _mul_422;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_422) : "f"(frag[42]), "f"(tok_160));
                        float _mul_423;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_423) : "f"(_mul_422), "f"(sc));
                        converted[3] = _mul_423;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_161 = token_159 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_159 * 64 + (base_feature ^ swizzle_feature_161)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_159 * 64 + (base_feature - 64 ^ swizzle_feature_161)) / 4] = packed_word;
                        }
                        int token_162 = base_token + 16 + 1;
                        float tok_163 = tok_stage[token_block_145 * 64 + token_162];
                        float _mul_424;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_424) : "f"(frag[9]), "f"(tok_163));
                        float _mul_425;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_425) : "f"(_mul_424), "f"(sc));
                        converted[0] = _mul_425;
                        float _mul_426;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_426) : "f"(frag[11]), "f"(tok_163));
                        float _mul_427;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_427) : "f"(_mul_426), "f"(sc));
                        converted[1] = _mul_427;
                        float _mul_428;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_428) : "f"(frag[41]), "f"(tok_163));
                        float _mul_429;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_429) : "f"(_mul_428), "f"(sc));
                        converted[2] = _mul_429;
                        float _mul_430;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_430) : "f"(frag[43]), "f"(tok_163));
                        float _mul_431;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_431) : "f"(_mul_430), "f"(sc));
                        converted[3] = _mul_431;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_164 = token_162 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_162 * 64 + (base_feature ^ swizzle_feature_164)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_162 * 64 + (base_feature - 64 ^ swizzle_feature_164)) / 4] = packed_word;
                        }
                        int token_165 = base_token + 24;
                        float tok_166 = tok_stage[token_block_145 * 64 + token_165];
                        float _mul_432;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_432) : "f"(frag[12]), "f"(tok_166));
                        float _mul_433;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_433) : "f"(_mul_432), "f"(sc));
                        converted[0] = _mul_433;
                        float _mul_434;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_434) : "f"(frag[14]), "f"(tok_166));
                        float _mul_435;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_435) : "f"(_mul_434), "f"(sc));
                        converted[1] = _mul_435;
                        float _mul_436;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_436) : "f"(frag[44]), "f"(tok_166));
                        float _mul_437;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_437) : "f"(_mul_436), "f"(sc));
                        converted[2] = _mul_437;
                        float _mul_438;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_438) : "f"(frag[46]), "f"(tok_166));
                        float _mul_439;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_439) : "f"(_mul_438), "f"(sc));
                        converted[3] = _mul_439;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_167 = token_165 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_165 * 64 + (base_feature ^ swizzle_feature_167)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_165 * 64 + (base_feature - 64 ^ swizzle_feature_167)) / 4] = packed_word;
                        }
                        int token_168 = base_token + 24 + 1;
                        float tok_169 = tok_stage[token_block_145 * 64 + token_168];
                        float _mul_440;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_440) : "f"(frag[13]), "f"(tok_169));
                        float _mul_441;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_441) : "f"(_mul_440), "f"(sc));
                        converted[0] = _mul_441;
                        float _mul_442;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_442) : "f"(frag[15]), "f"(tok_169));
                        float _mul_443;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_443) : "f"(_mul_442), "f"(sc));
                        converted[1] = _mul_443;
                        float _mul_444;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_444) : "f"(frag[45]), "f"(tok_169));
                        float _mul_445;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_445) : "f"(_mul_444), "f"(sc));
                        converted[2] = _mul_445;
                        float _mul_446;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_446) : "f"(frag[47]), "f"(tok_169));
                        float _mul_447;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_447) : "f"(_mul_446), "f"(sc));
                        converted[3] = _mul_447;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_170 = token_168 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_168 * 64 + (base_feature ^ swizzle_feature_170)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_168 * 64 + (base_feature - 64 ^ swizzle_feature_170)) / 4] = packed_word;
                        }
                        int token_171 = base_token + 32;
                        float tok_172 = tok_stage[token_block_145 * 64 + token_171];
                        float _mul_448;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_448) : "f"(frag[16]), "f"(tok_172));
                        float _mul_449;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_449) : "f"(_mul_448), "f"(sc));
                        converted[0] = _mul_449;
                        float _mul_450;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_450) : "f"(frag[18]), "f"(tok_172));
                        float _mul_451;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_451) : "f"(_mul_450), "f"(sc));
                        converted[1] = _mul_451;
                        float _mul_452;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_452) : "f"(frag[48]), "f"(tok_172));
                        float _mul_453;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_453) : "f"(_mul_452), "f"(sc));
                        converted[2] = _mul_453;
                        float _mul_454;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_454) : "f"(frag[50]), "f"(tok_172));
                        float _mul_455;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_455) : "f"(_mul_454), "f"(sc));
                        converted[3] = _mul_455;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_173 = token_171 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_171 * 64 + (base_feature ^ swizzle_feature_173)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_171 * 64 + (base_feature - 64 ^ swizzle_feature_173)) / 4] = packed_word;
                        }
                        int token_174 = base_token + 32 + 1;
                        float tok_175 = tok_stage[token_block_145 * 64 + token_174];
                        float _mul_456;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_456) : "f"(frag[17]), "f"(tok_175));
                        float _mul_457;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_457) : "f"(_mul_456), "f"(sc));
                        converted[0] = _mul_457;
                        float _mul_458;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_458) : "f"(frag[19]), "f"(tok_175));
                        float _mul_459;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_459) : "f"(_mul_458), "f"(sc));
                        converted[1] = _mul_459;
                        float _mul_460;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_460) : "f"(frag[49]), "f"(tok_175));
                        float _mul_461;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_461) : "f"(_mul_460), "f"(sc));
                        converted[2] = _mul_461;
                        float _mul_462;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_462) : "f"(frag[51]), "f"(tok_175));
                        float _mul_463;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_463) : "f"(_mul_462), "f"(sc));
                        converted[3] = _mul_463;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_176 = token_174 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_174 * 64 + (base_feature ^ swizzle_feature_176)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_174 * 64 + (base_feature - 64 ^ swizzle_feature_176)) / 4] = packed_word;
                        }
                        int token_177 = base_token + 40;
                        float tok_178 = tok_stage[token_block_145 * 64 + token_177];
                        float _mul_464;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_464) : "f"(frag[20]), "f"(tok_178));
                        float _mul_465;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_465) : "f"(_mul_464), "f"(sc));
                        converted[0] = _mul_465;
                        float _mul_466;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_466) : "f"(frag[22]), "f"(tok_178));
                        float _mul_467;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_467) : "f"(_mul_466), "f"(sc));
                        converted[1] = _mul_467;
                        float _mul_468;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_468) : "f"(frag[52]), "f"(tok_178));
                        float _mul_469;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_469) : "f"(_mul_468), "f"(sc));
                        converted[2] = _mul_469;
                        float _mul_470;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_470) : "f"(frag[54]), "f"(tok_178));
                        float _mul_471;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_471) : "f"(_mul_470), "f"(sc));
                        converted[3] = _mul_471;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_179 = token_177 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_177 * 64 + (base_feature ^ swizzle_feature_179)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_177 * 64 + (base_feature - 64 ^ swizzle_feature_179)) / 4] = packed_word;
                        }
                        int token_180 = base_token + 40 + 1;
                        float tok_181 = tok_stage[token_block_145 * 64 + token_180];
                        float _mul_472;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_472) : "f"(frag[21]), "f"(tok_181));
                        float _mul_473;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_473) : "f"(_mul_472), "f"(sc));
                        converted[0] = _mul_473;
                        float _mul_474;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_474) : "f"(frag[23]), "f"(tok_181));
                        float _mul_475;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_475) : "f"(_mul_474), "f"(sc));
                        converted[1] = _mul_475;
                        float _mul_476;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_476) : "f"(frag[53]), "f"(tok_181));
                        float _mul_477;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_477) : "f"(_mul_476), "f"(sc));
                        converted[2] = _mul_477;
                        float _mul_478;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_478) : "f"(frag[55]), "f"(tok_181));
                        float _mul_479;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_479) : "f"(_mul_478), "f"(sc));
                        converted[3] = _mul_479;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_182 = token_180 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_180 * 64 + (base_feature ^ swizzle_feature_182)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_180 * 64 + (base_feature - 64 ^ swizzle_feature_182)) / 4] = packed_word;
                        }
                        int token_183 = base_token + 48;
                        float tok_184 = tok_stage[token_block_145 * 64 + token_183];
                        float _mul_480;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_480) : "f"(frag[24]), "f"(tok_184));
                        float _mul_481;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_481) : "f"(_mul_480), "f"(sc));
                        converted[0] = _mul_481;
                        float _mul_482;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_482) : "f"(frag[26]), "f"(tok_184));
                        float _mul_483;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_483) : "f"(_mul_482), "f"(sc));
                        converted[1] = _mul_483;
                        float _mul_484;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_484) : "f"(frag[56]), "f"(tok_184));
                        float _mul_485;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_485) : "f"(_mul_484), "f"(sc));
                        converted[2] = _mul_485;
                        float _mul_486;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_486) : "f"(frag[58]), "f"(tok_184));
                        float _mul_487;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_487) : "f"(_mul_486), "f"(sc));
                        converted[3] = _mul_487;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_185 = token_183 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_183 * 64 + (base_feature ^ swizzle_feature_185)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_183 * 64 + (base_feature - 64 ^ swizzle_feature_185)) / 4] = packed_word;
                        }
                        int token_186 = base_token + 48 + 1;
                        float tok_187 = tok_stage[token_block_145 * 64 + token_186];
                        float _mul_488;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_488) : "f"(frag[25]), "f"(tok_187));
                        float _mul_489;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_489) : "f"(_mul_488), "f"(sc));
                        converted[0] = _mul_489;
                        float _mul_490;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_490) : "f"(frag[27]), "f"(tok_187));
                        float _mul_491;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_491) : "f"(_mul_490), "f"(sc));
                        converted[1] = _mul_491;
                        float _mul_492;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_492) : "f"(frag[57]), "f"(tok_187));
                        float _mul_493;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_493) : "f"(_mul_492), "f"(sc));
                        converted[2] = _mul_493;
                        float _mul_494;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_494) : "f"(frag[59]), "f"(tok_187));
                        float _mul_495;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_495) : "f"(_mul_494), "f"(sc));
                        converted[3] = _mul_495;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_188 = token_186 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_186 * 64 + (base_feature ^ swizzle_feature_188)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_186 * 64 + (base_feature - 64 ^ swizzle_feature_188)) / 4] = packed_word;
                        }
                        int token_189 = base_token + 56;
                        float tok_190 = tok_stage[token_block_145 * 64 + token_189];
                        float _mul_496;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_496) : "f"(frag[28]), "f"(tok_190));
                        float _mul_497;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_497) : "f"(_mul_496), "f"(sc));
                        converted[0] = _mul_497;
                        float _mul_498;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_498) : "f"(frag[30]), "f"(tok_190));
                        float _mul_499;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_499) : "f"(_mul_498), "f"(sc));
                        converted[1] = _mul_499;
                        float _mul_500;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_500) : "f"(frag[60]), "f"(tok_190));
                        float _mul_501;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_501) : "f"(_mul_500), "f"(sc));
                        converted[2] = _mul_501;
                        float _mul_502;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_502) : "f"(frag[62]), "f"(tok_190));
                        float _mul_503;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_503) : "f"(_mul_502), "f"(sc));
                        converted[3] = _mul_503;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_191 = token_189 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_189 * 64 + (base_feature ^ swizzle_feature_191)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_189 * 64 + (base_feature - 64 ^ swizzle_feature_191)) / 4] = packed_word;
                        }
                        int token_192 = base_token + 56 + 1;
                        float tok_193 = tok_stage[token_block_145 * 64 + token_192];
                        float _mul_504;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_504) : "f"(frag[29]), "f"(tok_193));
                        float _mul_505;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_505) : "f"(_mul_504), "f"(sc));
                        converted[0] = _mul_505;
                        float _mul_506;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_506) : "f"(frag[31]), "f"(tok_193));
                        float _mul_507;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_507) : "f"(_mul_506), "f"(sc));
                        converted[1] = _mul_507;
                        float _mul_508;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_508) : "f"(frag[61]), "f"(tok_193));
                        float _mul_509;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_509) : "f"(_mul_508), "f"(sc));
                        converted[2] = _mul_509;
                        float _mul_510;
                        asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_510) : "f"(frag[63]), "f"(tok_193));
                        float _mul_511;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_511) : "f"(_mul_510), "f"(sc));
                        converted[3] = _mul_511;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_194 = token_192 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_192 * 64 + (base_feature ^ swizzle_feature_194)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_192 * 64 + (base_feature - 64 ^ swizzle_feature_194)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows_3 = (256 - valid_rows % 256) % 256;
                                tma_store_4d((&C), m_tile * 128, padding_rows_3 + token_block_145 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows_3 + token_block_145 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr + 8192);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        epilogue_local_idx = epilogue_local_idx ^ 1;
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                unsigned int valid = 0;
                unsigned int next_x = 0;
                unsigned int next_y = 0;
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
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                valid = _clc_valid_5;
                uint32_t _clc_ctaid_10 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_10)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                next_x = _clc_ctaid_10;
                uint32_t _clc_ctaid_11 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_11)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                next_y = _clc_ctaid_11;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage) * 8);
                work_stage += 1;
                if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                unsigned int valid_0 = valid;
                m_tile = next_x;
                n_tile = next_y;
                if (valid_0 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 4) {
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_1 = num_non_exiting_ctas[0];
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            unsigned int _phase_b_empty = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m * grid_n; _tile_iter_1++) {
                int valid_rows_1 = (unsigned int)tile_mn_limit[n_tile_1] - n_tile_1 * 256;
                if (tile_count_1 > (int)n_tile_1) {
                    if (valid_rows_1 > 0) {
                        int padding_rows_4 = (256 - valid_rows_1 % 256) % 256;
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            mbarrier_wait(b_empty_addr + (stage) * 8, _phase_b_empty);
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_b_addr + stage * 32768, (&B), iter_k * 128, padding_rows_4, 1073741824, n_tile_1 * 256 - (unsigned int)padding_rows_4 + 1073741824, b_full_addr + (stage) * 8);
                                mbarrier_arrive_expect_tx(b_full_addr + (stage) * 8, 32768);
                            }
                            stage += 1;
                            if (stage == 3) { stage = 0; _phase_b_empty ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                unsigned int valid_1 = 0;
                unsigned int next_x_1 = 0;
                unsigned int next_y_1 = 0;
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
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                valid_1 = _clc_valid_1;
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
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                next_x_1 = _clc_ctaid_2;
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
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                next_y_1 = _clc_ctaid_3;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
                work_stage_1 += 1;
                if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                unsigned int valid_0_1 = valid_1;
                m_tile_1 = next_x_1;
                n_tile_1 = next_y_1;
                if (valid_0_1 == 0) {
                    break;
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 5) {
        { // load_sfb_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_2 = num_non_exiting_ctas[0];
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            unsigned int _phase_sfb_empty = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m * grid_n; _tile_iter_2++) {
                if (tile_count_2 > (int)n_tile_2) {
                    int valid_rows_sfb = (unsigned int)tile_mn_limit[n_tile_2] - n_tile_2 * 256;
                    if (valid_rows_sfb > 0) {
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                            mbarrier_wait(sfb_empty_addr + (stage_1) * 8, _phase_sfb_empty);
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_sfb_addr + stage_1 * 4096, (&SFB), 0, 0, iter_k_1 * 4, n_tile_2 * 2, sfb_full_addr + (stage_1) * 8);
                                mbarrier_arrive_expect_tx(sfb_full_addr + (stage_1) * 8, 4096);
                            }
                            stage_1 += 1;
                            if (stage_1 == 3) { stage_1 = 0; _phase_sfb_empty ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                unsigned int valid_2 = 0;
                unsigned int next_x_2 = 0;
                unsigned int next_y_2 = 0;
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                valid_2 = _clc_valid_2;
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                next_x_2 = _clc_ctaid_4;
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                next_y_2 = _clc_ctaid_5;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
                work_stage_2 += 1;
                if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                unsigned int valid_0_2 = valid_2;
                m_tile_2 = next_x_2;
                n_tile_2 = next_y_2;
                if (valid_0_2 == 0) {
                    break;
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_a ----
    if (warp == 6) {
        { // load_a_main
            int tile_count_3 = num_non_exiting_ctas[0];
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_a_empty = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m * grid_n; _tile_iter_3++) {
                if (tile_count_3 > (int)n_tile_3) {
                    int expert = tile_expert[n_tile_3];
                    mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                    mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                    throttle_stage += 1;
                    if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                    int valid_rows_a = (unsigned int)tile_mn_limit[n_tile_3] - n_tile_3 * 256;
                    if (valid_rows_a > 0) {
                        #pragma unroll 1
                        for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                            mbarrier_wait(a_empty_addr + (stage_2) * 8, _phase_a_empty);
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_a_addr + stage_2 * 16384, (&A), 0, m_tile_3 * 128, iter_k_2, expert, a_full_addr + (stage_2) * 8);
                                mbarrier_arrive_expect_tx(a_full_addr + (stage_2) * 8, 16384);
                            }
                            stage_2 += 1;
                            if (stage_2 == 3) { stage_2 = 0; _phase_a_empty ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                unsigned int valid_3 = 0;
                unsigned int next_x_3 = 0;
                unsigned int next_y_3 = 0;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                valid_3 = _clc_valid_0;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_x_3 = _clc_ctaid_0;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_y_3 = _clc_ctaid_1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
                work_stage_3 += 1;
                if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                unsigned int valid_0_3 = valid_3;
                m_tile_3 = next_x_3;
                n_tile_3 = next_y_3;
                if (valid_0_3 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 7) {
        { // load_sfa_main
            int tile_count_4 = num_non_exiting_ctas[0];
            unsigned int stage_3 = 0;
            unsigned int work_stage_4 = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int _phase_sfa_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < grid_m * grid_n; _tile_iter_4++) {
                int expert_sfa = tile_expert[n_tile_4];
                if (tile_count_4 > (int)n_tile_4) {
                    int valid_rows_sfa = (unsigned int)tile_mn_limit[n_tile_4] - n_tile_4 * 256;
                    if (valid_rows_sfa > 0) {
                        #pragma unroll 1
                        for (int iter_k_3 = 0; iter_k_3 < K_tiles; iter_k_3++) {
                            mbarrier_wait(sfa_empty_addr + (stage_3) * 8, _phase_sfa_empty);
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_sfa_addr + stage_3 * 2048, (&SFA), 0, 0, iter_k_3 * 4, (unsigned int)(expert_sfa * grid_m) + m_tile_4, sfa_full_addr + (stage_3) * 8);
                                mbarrier_arrive_expect_tx(sfa_full_addr + (stage_3) * 8, 2048);
                            }
                            stage_3 += 1;
                            if (stage_3 == 3) { stage_3 = 0; _phase_sfa_empty ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
                unsigned int valid_4 = 0;
                unsigned int next_x_4 = 0;
                unsigned int next_y_4 = 0;
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
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                valid_4 = _clc_valid_3;
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
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                next_x_4 = _clc_ctaid_6;
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
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                next_y_4 = _clc_ctaid_7;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
                work_stage_4 += 1;
                if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
                unsigned int valid_0_4 = valid_4;
                m_tile_4 = next_x_4;
                n_tile_4 = next_y_4;
                if (valid_0_4 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: copy_sfab_mma ----
    if (warp == 8) {
        { // copy_sfab_mma_main
            int tile_count_5 = num_non_exiting_ctas[0];
            unsigned int stage_4 = 0;
            unsigned int work_stage_5 = 0;
            int mma_local_idx = 0;
            unsigned int m_tile_5 = blockIdx.x;
            unsigned int n_tile_5 = blockIdx.y;
            unsigned int _phase_mma_free_0 = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < grid_m * grid_n; _tile_iter_5++) {
                if (tile_count_5 > (int)n_tile_5) {
                    int valid_rows_m = (unsigned int)tile_mn_limit[n_tile_5] - n_tile_5 * 256;
                    if (valid_rows_m > 0) {
                        mbarrier_wait(mma_free_addr, _phase_mma_free_0);
                        _phase_mma_free_0 ^= 1;
                        #pragma unroll 1
                        for (int iter_k_4 = 0; iter_k_4 < K_tiles; iter_k_4++) {
                            mbarrier_wait(a_full_addr + (stage_4) * 8, _phase_a_full);
                            mbarrier_wait(b_full_addr + (stage_4) * 8, _phase_b_full);
                            mbarrier_wait(sfa_full_addr + (stage_4) * 8, _phase_sfa_full);
                            mbarrier_wait(sfb_full_addr + (stage_4) * 8, _phase_sfb_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfa)), "l"(_tcgen05_cp_desc_0)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfa + 4)), "l"(_tcgen05_cp_desc_1)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfa + 8)), "l"(_tcgen05_cp_desc_2)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfa + 12)), "l"(_tcgen05_cp_desc_3)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb)), "l"(_tcgen05_cp_desc_4)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 8)), "l"(_tcgen05_cp_desc_5)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 16)), "l"(_tcgen05_cp_desc_6)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 24)), "l"(_tcgen05_cp_desc_7)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 4)), "l"(_tcgen05_cp_desc_8)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 12)), "l"(_tcgen05_cp_desc_9)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 20)), "l"(_tcgen05_cp_desc_10)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 28)), "l"(_tcgen05_cp_desc_11)
                                        : "memory");
                                }
                            }
                            int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (stage_4) * 1024);
                            int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (stage_4) * 2048);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                        0x8400480U, tmem_sfa + 0, tmem_sfb + 0, ((((1) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_1 = make_warp_uniform((((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_4) * 1024);
                            int _mma_b_lo_1 = make_warp_uniform((((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_4) * 2048);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                        0x8400480U, tmem_sfa + 4 + 0, tmem_sfb + 8 + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_2 = make_warp_uniform((((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_4) * 1024);
                            int _mma_b_lo_2 = make_warp_uniform((((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_4) * 2048);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                        0x8400480U, tmem_sfa + 8 + 0, tmem_sfb + 16 + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_3 = make_warp_uniform((((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_4) * 1024);
                            int _mma_b_lo_3 = make_warp_uniform((((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_4) * 2048);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                        0x8400480U, tmem_sfa + 12 + 0, tmem_sfb + 24 + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            elect_commit(a_empty_addr + (stage_4) * 8);
                            elect_commit(b_empty_addr + (stage_4) * 8);
                            elect_commit(sfa_empty_addr + (stage_4) * 8);
                            elect_commit(sfb_empty_addr + (stage_4) * 8);
                            if (iter_k_4 + 1 == K_tiles) {
                                elect_commit(mma_full_addr);
                            }
                            stage_4 += 1;
                            if (stage_4 == 3) { stage_4 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; _phase_sfa_full ^= 1; _phase_sfb_full ^= 1; }
                        }
                        mma_local_idx = mma_local_idx ^ 1;
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
                unsigned int valid_5 = 0;
                unsigned int next_x_5 = 0;
                unsigned int next_y_5 = 0;
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                valid_5 = _clc_valid_4;
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                next_x_5 = _clc_ctaid_8;
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                next_y_5 = _clc_ctaid_9;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
                work_stage_5 += 1;
                if (work_stage_5 == 3) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                unsigned int valid_0_5 = valid_5;
                m_tile_5 = next_x_5;
                n_tile_5 = next_y_5;
                if (valid_0_5 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 9) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_6 = num_non_exiting_ctas[0];
            unsigned int work_stage_6 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int m_tile_6 = blockIdx.x;
            unsigned int n_tile_6 = blockIdx.y;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_6 = 0; _tile_iter_6 < grid_m * grid_n; _tile_iter_6++) {
                if (tile_count_6 > (int)n_tile_6) {
                    mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                    mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                    throttle_stage_1 += 1;
                    if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                } else {
                    if (elect_sync()) {
                        mbarrier_init(fast_ready_addr, 1);
                        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
                    }
                    __syncwarp();
                    unsigned int fast_stage = 0;
                    unsigned int fast_phase = 0;
                    #pragma unroll 1
                    for (unsigned int _drain_iter = 0; _drain_iter < grid_m * grid_n; _drain_iter++) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(fast_ready_addr + (fast_stage) * 8, 64);
                            for (int slot_idx = 0; slot_idx < 4; slot_idx++) {
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_response_addr + fast_stage * 64 + slot_idx * 16), "r"(fast_ready_addr + fast_stage * 8)
                                    : "memory");
                            }
                        }
                        mbarrier_wait(fast_ready_addr + (fast_stage) * 8, fast_phase);
                        unsigned int canceled = 0;
                        for (int slot_idx_1 = 0; slot_idx_1 < 4; slot_idx_1++) {
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
                                : "r"(fast_response_addr + fast_stage * 64 + slot_idx_1 * 16)
                                : "memory");
                            canceled += _clc_valid_6;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        fast_phase ^= 1;
                        if (canceled == 0) {
                            break;
                        }
                    }
                }
                if (elect_sync()) {
                    mbarrier_wait(work_empty_addr + (work_stage_6) * 8, _phase_work_empty);
                    mbarrier_arrive_expect_tx(work_full_addr + (work_stage_6) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + work_stage_6 * 16 + 0 * 16), "r"(work_full_addr + work_stage_6 * 8)
                        : "memory");
                }
                __syncwarp();
                mbarrier_wait(work_full_addr + (work_stage_6) * 8, _phase_work_full_6);
                unsigned int valid_6 = 0;
                unsigned int next_x_6 = 0;
                unsigned int next_y_6 = 0;
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
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                valid_6 = _clc_valid_7;
                uint32_t _clc_ctaid_12 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_12)
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                next_x_6 = _clc_ctaid_12;
                uint32_t _clc_ctaid_13 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_13)
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                next_y_6 = _clc_ctaid_13;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                work_stage_6 += 1;
                if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_empty ^= 1; _phase_work_full_6 ^= 1; }
                unsigned int valid_0_6 = valid_6;
                m_tile_6 = next_x_6;
                n_tile_6 = next_y_6;
                if (valid_0_6 == 0) {
                    break;
                }
            }
            #pragma unroll 1
            for (unsigned int _tail_iter = 0; _tail_iter < 3; _tail_iter++) {
                mbarrier_wait(work_empty_addr + (work_stage_6) * 8, _phase_work_empty);
                work_stage_6 += 1;
                if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_empty ^= 1; _phase_work_full_6 ^= 1; }
            }
        }
    }
    // ---- Role: padding ----
    if (warp >= 10 && warp <= 11) {
        { // padding_main
            unsigned int work_stage_7 = 0;
            unsigned int m_tile_7 = blockIdx.x;
            unsigned int n_tile_7 = blockIdx.y;
            unsigned int _phase_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_7 = 0; _tile_iter_7 < grid_m * grid_n; _tile_iter_7++) {
                mbarrier_wait(work_full_addr + (work_stage_7) * 8, _phase_work_full_7);
                unsigned int valid_7 = 0;
                unsigned int next_x_7 = 0;
                unsigned int next_y_7 = 0;
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
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                valid_7 = _clc_valid_8;
                uint32_t _clc_ctaid_14 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_14)
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                next_x_7 = _clc_ctaid_14;
                uint32_t _clc_ctaid_15 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_15)
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                next_y_7 = _clc_ctaid_15;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                work_stage_7 += 1;
                if (work_stage_7 == 3) { work_stage_7 = 0; _phase_work_full_7 ^= 1; }
                unsigned int valid_0_7 = valid_7;
                m_tile_7 = next_x_7;
                n_tile_7 = next_y_7;
                if (valid_0_7 == 0) {
                    break;
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
