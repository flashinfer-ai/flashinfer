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
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_K_PIPE_STAGES 6
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_DRAIN_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 99328
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_EPI_STAGING_OFF 197632
#define SMEM_EPI_STAGING_STAGE_BYTES 4096
#define SMEM_EPI_STAGING_STRIDE 4096
#define SMEM_WORK_RESPONSE_OFF 201728
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_DRAIN_RESPONSE_OFF 201776
#define SMEM_FAST_DRAIN_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_DRAIN_RESPONSE_STRIDE 64
#define SMEM_TOTAL 201856
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_stepfun_moe_91584b1845d09c318acd(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    #define b_full_addr (mbar_base + 48)
    #define k_done_addr (mbar_base + 96)
    #define mma_full_addr (mbar_base + 144)
    #define mma_free_addr (mbar_base + 160)
    #define work_full_addr (mbar_base + 176)
    #define work_empty_addr (mbar_base + 200)
    #define throttle_full_addr (mbar_base + 224)
    #define throttle_empty_addr (mbar_base + 248)
    #define drain_full_addr (mbar_base + 272)

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
    uint16_t* epi_staging = reinterpret_cast<uint16_t*>(smem_raw + SMEM_EPI_STAGING_OFF);
    const int epi_staging_addr = smem + SMEM_EPI_STAGING_OFF;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WORK_RESPONSE_OFF);
    const int work_response_addr = smem + SMEM_WORK_RESPONSE_OFF;
    unsigned int* fast_drain_response = reinterpret_cast<unsigned int*>(smem_raw + SMEM_FAST_DRAIN_RESPONSE_OFF);
    const int fast_drain_response_addr = smem + SMEM_FAST_DRAIN_RESPONSE_OFF;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((int)blockIdx.y >= num_non_exiting_ctas[0]) return;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 35 barriers)
    // Mbarriers at smem_raw[0..280)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 6 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            // b_full: 6 barriers, init_count=2
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            // k_done: 6 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // mma_free: 2 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            mbarrier_init(smem + 168, 256);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // work_empty: 3 barriers, init_count=960
            mbarrier_init(smem + 200, 960);
            mbarrier_init(smem + 208, 960);
            mbarrier_init(smem + 216, 960);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 224, 32);
            mbarrier_init(smem + 232, 32);
            mbarrier_init(smem + 240, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 248, 32);
            mbarrier_init(smem + 256, 32);
            mbarrier_init(smem + 264, 32);
            // --- pipeline 'drain_pipe' ---
            // drain_full: 1 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 280);
    if (warp == 0) {
        int _tmem_hold = smem + 280;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            const int warp_0 = warp;
            const int lane_1 = lane;
            unsigned int acc_stage = 0;
            unsigned int work_stage = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            int bound = num_non_exiting_ctas[0];
            float quad[4] = {0};
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m / 2 * grid_n; _tile_iter++) {
                if (m_tile >= (unsigned int)grid_m || n_tile >= (unsigned int)bound) {
                    break;
                }
                int expert = tile_expert[n_tile];
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 256;
                float sc = scale_c[expert];
                float sg = scale_gate[expert];
                float cl = clamp_limit[expert];
                float _sub_0;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(0.0f), "f"(cl));
                float neg_cl = _sub_0;
                float _mul_0;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(1.4426950216293335f), "f"(sg));
                float fused = _mul_0;
                mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_offset = acc_stage * 256;
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                    : "r"(taddr + (unsigned int)acc_offset));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row = warp_0 * 16 + lane_1 / 4 * 2;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int local_token0 = lane_1 % 4 * 2;
                int local_token1 = local_token0 + 1;
                float x0_00 = _tmem_load_0[0];
                float x0_01 = _tmem_load_0[1];
                float x1_00 = _tmem_load_0[2];
                float x1_01 = _tmem_load_0[3];
                float x0_10 = _tmem_load_1[0];
                float x0_11 = _tmem_load_1[1];
                float x1_10 = _tmem_load_1[2];
                float x1_11 = _tmem_load_1[3];
                float _max_0;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_0) : "f"(x0_00), "f"(neg_cl));
                float _min_0;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_0) : "f"(_max_0), "f"(cl));
                float x0c_00 = _min_0;
                float _max_1;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_1) : "f"(x0_01), "f"(neg_cl));
                float _min_1;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_1) : "f"(_max_1), "f"(cl));
                float x0c_01 = _min_1;
                float _max_2;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_2) : "f"(x0_10), "f"(neg_cl));
                float _min_2;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_2) : "f"(_max_2), "f"(cl));
                float x0c_10 = _min_2;
                float _max_3;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_3) : "f"(x0_11), "f"(neg_cl));
                float _min_3;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_3) : "f"(_max_3), "f"(cl));
                float x0c_11 = _min_3;
                float _fma_0;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(x0c_00), "f"(sg), "f"(0.0f));
                float lin_00 = _fma_0;
                float _fma_1;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(x0c_01), "f"(sg), "f"(0.0f));
                float lin_01 = _fma_1;
                float _fma_2;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(x0c_10), "f"(sg), "f"(0.0f));
                float lin_10 = _fma_2;
                float _fma_3;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(x0c_11), "f"(sg), "f"(0.0f));
                float lin_11 = _fma_3;
                float _mul_1;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(x1_00), "f"(fused));
                float _sub_1;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_1) : "f"(0.0f), "f"(_mul_1));
                float _exp2_0 = approx_exp2(_sub_1);
                float _add_0;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(1.0f), "f"(_exp2_0));
                float _rcp_0 = approx_rcp(_add_0);
                float sig_00 = _rcp_0;
                float _mul_2;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(x1_01), "f"(fused));
                float _sub_2;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_2) : "f"(0.0f), "f"(_mul_2));
                float _exp2_1 = approx_exp2(_sub_2);
                float _add_1;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(1.0f), "f"(_exp2_1));
                float _rcp_1 = approx_rcp(_add_1);
                float sig_01 = _rcp_1;
                float _mul_3;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(x1_10), "f"(fused));
                float _sub_3;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_3) : "f"(0.0f), "f"(_mul_3));
                float _exp2_2 = approx_exp2(_sub_3);
                float _add_2;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(1.0f), "f"(_exp2_2));
                float _rcp_2 = approx_rcp(_add_2);
                float sig_10 = _rcp_2;
                float _mul_4;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(x1_11), "f"(fused));
                float _sub_4;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_4) : "f"(0.0f), "f"(_mul_4));
                float _exp2_3 = approx_exp2(_sub_4);
                float _add_3;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(1.0f), "f"(_exp2_3));
                float _rcp_3 = approx_rcp(_add_3);
                float sig_11 = _rcp_3;
                float _mul_5;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(x1_00), "f"(sig_00));
                float act_00 = _mul_5;
                float _mul_6;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(x1_01), "f"(sig_01));
                float act_01 = _mul_6;
                float _mul_7;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(x1_10), "f"(sig_10));
                float act_10 = _mul_7;
                float _mul_8;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(x1_11), "f"(sig_11));
                float act_11 = _mul_8;
                float _min_4;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_4) : "f"(act_00), "f"(cl));
                act_00 = _min_4;
                float _min_5;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_5) : "f"(act_01), "f"(cl));
                act_01 = _min_5;
                float _min_6;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_6) : "f"(act_10), "f"(cl));
                act_10 = _min_6;
                float _min_7;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_7) : "f"(act_11), "f"(cl));
                act_11 = _min_7;
                float _mul_9;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(lin_00), "f"(act_00));
                float y_00 = _mul_9;
                float _mul_10;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(lin_10), "f"(act_10));
                float y_10 = _mul_10;
                float _mul_11;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(lin_01), "f"(act_01));
                float y_01 = _mul_11;
                float _mul_12;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(lin_11), "f"(act_11));
                float y_11 = _mul_12;
                float _mul_13;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(y_00), "f"(sc));
                quad[0] = _mul_13;
                float _mul_14;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(y_10), "f"(sc));
                quad[1] = _mul_14;
                float _mul_15;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(y_01), "f"(sc));
                quad[2] = _mul_15;
                float _mul_16;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(y_11), "f"(sc));
                quad[3] = _mul_16;
                uint32_t _fp8_0[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_0[0] = _packed;
                }
                int off0 = local_token0 * 64 + base_row;
                int off1 = local_token1 * 64 + base_row;
                int swz0 = off0 ^ (off0 >> 7 & 3) << 4;
                int swz1 = off1 ^ (off1 >> 7 & 3) << 4;
                epi_staging[swz0 >> 1] = _fp8_0[0] & 65535;
                epi_staging[swz1 >> 1] = _fp8_0[0] >> 16;
                int local_token0_0 = lane_1 % 4 * 2 + 8;
                int local_token1_1 = local_token0_0 + 1;
                float x0_00_2 = _tmem_load_0[4];
                float x0_01_3 = _tmem_load_0[5];
                float x1_00_4 = _tmem_load_0[6];
                float x1_01_5 = _tmem_load_0[7];
                float x0_10_6 = _tmem_load_1[4];
                float x0_11_7 = _tmem_load_1[5];
                float x1_10_8 = _tmem_load_1[6];
                float x1_11_9 = _tmem_load_1[7];
                float _max_4;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_4) : "f"(x0_00_2), "f"(neg_cl));
                float _min_8;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_8) : "f"(_max_4), "f"(cl));
                float x0c_00_10 = _min_8;
                float _max_5;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_5) : "f"(x0_01_3), "f"(neg_cl));
                float _min_9;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_9) : "f"(_max_5), "f"(cl));
                float x0c_01_11 = _min_9;
                float _max_6;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_6) : "f"(x0_10_6), "f"(neg_cl));
                float _min_10;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_10) : "f"(_max_6), "f"(cl));
                float x0c_10_12 = _min_10;
                float _max_7;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_7) : "f"(x0_11_7), "f"(neg_cl));
                float _min_11;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_11) : "f"(_max_7), "f"(cl));
                float x0c_11_13 = _min_11;
                float _fma_4;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(x0c_00_10), "f"(sg), "f"(0.0f));
                float lin_00_14 = _fma_4;
                float _fma_5;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(x0c_01_11), "f"(sg), "f"(0.0f));
                float lin_01_15 = _fma_5;
                float _fma_6;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(x0c_10_12), "f"(sg), "f"(0.0f));
                float lin_10_16 = _fma_6;
                float _fma_7;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(x0c_11_13), "f"(sg), "f"(0.0f));
                float lin_11_17 = _fma_7;
                float _mul_17;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(x1_00_4), "f"(fused));
                float _sub_5;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(0.0f), "f"(_mul_17));
                float _exp2_4 = approx_exp2(_sub_5);
                float _add_4;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(1.0f), "f"(_exp2_4));
                float _rcp_4 = approx_rcp(_add_4);
                float sig_00_18 = _rcp_4;
                float _mul_18;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(x1_01_5), "f"(fused));
                float _sub_6;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_6) : "f"(0.0f), "f"(_mul_18));
                float _exp2_5 = approx_exp2(_sub_6);
                float _add_5;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(1.0f), "f"(_exp2_5));
                float _rcp_5 = approx_rcp(_add_5);
                float sig_01_19 = _rcp_5;
                float _mul_19;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(x1_10_8), "f"(fused));
                float _sub_7;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(0.0f), "f"(_mul_19));
                float _exp2_6 = approx_exp2(_sub_7);
                float _add_6;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(1.0f), "f"(_exp2_6));
                float _rcp_6 = approx_rcp(_add_6);
                float sig_10_20 = _rcp_6;
                float _mul_20;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(x1_11_9), "f"(fused));
                float _sub_8;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_8) : "f"(0.0f), "f"(_mul_20));
                float _exp2_7 = approx_exp2(_sub_8);
                float _add_7;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(1.0f), "f"(_exp2_7));
                float _rcp_7 = approx_rcp(_add_7);
                float sig_11_21 = _rcp_7;
                float _mul_21;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(x1_00_4), "f"(sig_00_18));
                float act_00_22 = _mul_21;
                float _mul_22;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(x1_01_5), "f"(sig_01_19));
                float act_01_23 = _mul_22;
                float _mul_23;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(x1_10_8), "f"(sig_10_20));
                float act_10_24 = _mul_23;
                float _mul_24;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(x1_11_9), "f"(sig_11_21));
                float act_11_25 = _mul_24;
                float _min_12;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_12) : "f"(act_00_22), "f"(cl));
                act_00_22 = _min_12;
                float _min_13;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_13) : "f"(act_01_23), "f"(cl));
                act_01_23 = _min_13;
                float _min_14;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_14) : "f"(act_10_24), "f"(cl));
                act_10_24 = _min_14;
                float _min_15;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_15) : "f"(act_11_25), "f"(cl));
                act_11_25 = _min_15;
                float _mul_25;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(lin_00_14), "f"(act_00_22));
                float y_00_26 = _mul_25;
                float _mul_26;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(lin_10_16), "f"(act_10_24));
                float y_10_27 = _mul_26;
                float _mul_27;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(lin_01_15), "f"(act_01_23));
                float y_01_28 = _mul_27;
                float _mul_28;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(lin_11_17), "f"(act_11_25));
                float y_11_29 = _mul_28;
                float _mul_29;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(y_00_26), "f"(sc));
                quad[0] = _mul_29;
                float _mul_30;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(y_10_27), "f"(sc));
                quad[1] = _mul_30;
                float _mul_31;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(y_01_28), "f"(sc));
                quad[2] = _mul_31;
                float _mul_32;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(y_11_29), "f"(sc));
                quad[3] = _mul_32;
                uint32_t _fp8_1[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_1[0] = _packed;
                }
                int off0_30 = local_token0_0 * 64 + base_row;
                int off1_31 = local_token1_1 * 64 + base_row;
                int swz0_32 = off0_30 ^ (off0_30 >> 7 & 3) << 4;
                int swz1_33 = off1_31 ^ (off1_31 >> 7 & 3) << 4;
                epi_staging[swz0_32 >> 1] = _fp8_1[0] & 65535;
                epi_staging[swz1_33 >> 1] = _fp8_1[0] >> 16;
                int local_token0_34 = lane_1 % 4 * 2 + 16;
                int local_token1_35 = local_token0_34 + 1;
                float x0_00_36 = _tmem_load_0[8];
                float x0_01_37 = _tmem_load_0[9];
                float x1_00_38 = _tmem_load_0[10];
                float x1_01_39 = _tmem_load_0[11];
                float x0_10_40 = _tmem_load_1[8];
                float x0_11_41 = _tmem_load_1[9];
                float x1_10_42 = _tmem_load_1[10];
                float x1_11_43 = _tmem_load_1[11];
                float _max_8;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_8) : "f"(x0_00_36), "f"(neg_cl));
                float _min_16;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_16) : "f"(_max_8), "f"(cl));
                float x0c_00_44 = _min_16;
                float _max_9;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_9) : "f"(x0_01_37), "f"(neg_cl));
                float _min_17;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_17) : "f"(_max_9), "f"(cl));
                float x0c_01_45 = _min_17;
                float _max_10;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_10) : "f"(x0_10_40), "f"(neg_cl));
                float _min_18;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_18) : "f"(_max_10), "f"(cl));
                float x0c_10_46 = _min_18;
                float _max_11;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_11) : "f"(x0_11_41), "f"(neg_cl));
                float _min_19;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_19) : "f"(_max_11), "f"(cl));
                float x0c_11_47 = _min_19;
                float _fma_8;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(x0c_00_44), "f"(sg), "f"(0.0f));
                float lin_00_48 = _fma_8;
                float _fma_9;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(x0c_01_45), "f"(sg), "f"(0.0f));
                float lin_01_49 = _fma_9;
                float _fma_10;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(x0c_10_46), "f"(sg), "f"(0.0f));
                float lin_10_50 = _fma_10;
                float _fma_11;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(x0c_11_47), "f"(sg), "f"(0.0f));
                float lin_11_51 = _fma_11;
                float _mul_33;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(x1_00_38), "f"(fused));
                float _sub_9;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_9) : "f"(0.0f), "f"(_mul_33));
                float _exp2_8 = approx_exp2(_sub_9);
                float _add_8;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(1.0f), "f"(_exp2_8));
                float _rcp_8 = approx_rcp(_add_8);
                float sig_00_52 = _rcp_8;
                float _mul_34;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(x1_01_39), "f"(fused));
                float _sub_10;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_10) : "f"(0.0f), "f"(_mul_34));
                float _exp2_9 = approx_exp2(_sub_10);
                float _add_9;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(1.0f), "f"(_exp2_9));
                float _rcp_9 = approx_rcp(_add_9);
                float sig_01_53 = _rcp_9;
                float _mul_35;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(x1_10_42), "f"(fused));
                float _sub_11;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_11) : "f"(0.0f), "f"(_mul_35));
                float _exp2_10 = approx_exp2(_sub_11);
                float _add_10;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(1.0f), "f"(_exp2_10));
                float _rcp_10 = approx_rcp(_add_10);
                float sig_10_54 = _rcp_10;
                float _mul_36;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(x1_11_43), "f"(fused));
                float _sub_12;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(0.0f), "f"(_mul_36));
                float _exp2_11 = approx_exp2(_sub_12);
                float _add_11;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(1.0f), "f"(_exp2_11));
                float _rcp_11 = approx_rcp(_add_11);
                float sig_11_55 = _rcp_11;
                float _mul_37;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(x1_00_38), "f"(sig_00_52));
                float act_00_56 = _mul_37;
                float _mul_38;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(x1_01_39), "f"(sig_01_53));
                float act_01_57 = _mul_38;
                float _mul_39;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(x1_10_42), "f"(sig_10_54));
                float act_10_58 = _mul_39;
                float _mul_40;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(x1_11_43), "f"(sig_11_55));
                float act_11_59 = _mul_40;
                float _min_20;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_20) : "f"(act_00_56), "f"(cl));
                act_00_56 = _min_20;
                float _min_21;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_21) : "f"(act_01_57), "f"(cl));
                act_01_57 = _min_21;
                float _min_22;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_22) : "f"(act_10_58), "f"(cl));
                act_10_58 = _min_22;
                float _min_23;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_23) : "f"(act_11_59), "f"(cl));
                act_11_59 = _min_23;
                float _mul_41;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(lin_00_48), "f"(act_00_56));
                float y_00_60 = _mul_41;
                float _mul_42;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_42) : "f"(lin_10_50), "f"(act_10_58));
                float y_10_61 = _mul_42;
                float _mul_43;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_43) : "f"(lin_01_49), "f"(act_01_57));
                float y_01_62 = _mul_43;
                float _mul_44;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_44) : "f"(lin_11_51), "f"(act_11_59));
                float y_11_63 = _mul_44;
                float _mul_45;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_45) : "f"(y_00_60), "f"(sc));
                quad[0] = _mul_45;
                float _mul_46;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_46) : "f"(y_10_61), "f"(sc));
                quad[1] = _mul_46;
                float _mul_47;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_47) : "f"(y_01_62), "f"(sc));
                quad[2] = _mul_47;
                float _mul_48;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_48) : "f"(y_11_63), "f"(sc));
                quad[3] = _mul_48;
                uint32_t _fp8_2[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_2[0] = _packed;
                }
                int off0_64 = local_token0_34 * 64 + base_row;
                int off1_65 = local_token1_35 * 64 + base_row;
                int swz0_66 = off0_64 ^ (off0_64 >> 7 & 3) << 4;
                int swz1_67 = off1_65 ^ (off1_65 >> 7 & 3) << 4;
                epi_staging[swz0_66 >> 1] = _fp8_2[0] & 65535;
                epi_staging[swz1_67 >> 1] = _fp8_2[0] >> 16;
                int local_token0_68 = lane_1 % 4 * 2 + 24;
                int local_token1_69 = local_token0_68 + 1;
                float x0_00_70 = _tmem_load_0[12];
                float x0_01_71 = _tmem_load_0[13];
                float x1_00_72 = _tmem_load_0[14];
                float x1_01_73 = _tmem_load_0[15];
                float x0_10_74 = _tmem_load_1[12];
                float x0_11_75 = _tmem_load_1[13];
                float x1_10_76 = _tmem_load_1[14];
                float x1_11_77 = _tmem_load_1[15];
                float _max_12;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_12) : "f"(x0_00_70), "f"(neg_cl));
                float _min_24;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_24) : "f"(_max_12), "f"(cl));
                float x0c_00_78 = _min_24;
                float _max_13;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_13) : "f"(x0_01_71), "f"(neg_cl));
                float _min_25;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_25) : "f"(_max_13), "f"(cl));
                float x0c_01_79 = _min_25;
                float _max_14;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_14) : "f"(x0_10_74), "f"(neg_cl));
                float _min_26;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_26) : "f"(_max_14), "f"(cl));
                float x0c_10_80 = _min_26;
                float _max_15;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_15) : "f"(x0_11_75), "f"(neg_cl));
                float _min_27;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_27) : "f"(_max_15), "f"(cl));
                float x0c_11_81 = _min_27;
                float _fma_12;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(x0c_00_78), "f"(sg), "f"(0.0f));
                float lin_00_82 = _fma_12;
                float _fma_13;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(x0c_01_79), "f"(sg), "f"(0.0f));
                float lin_01_83 = _fma_13;
                float _fma_14;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(x0c_10_80), "f"(sg), "f"(0.0f));
                float lin_10_84 = _fma_14;
                float _fma_15;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(x0c_11_81), "f"(sg), "f"(0.0f));
                float lin_11_85 = _fma_15;
                float _mul_49;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_49) : "f"(x1_00_72), "f"(fused));
                float _sub_13;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_13) : "f"(0.0f), "f"(_mul_49));
                float _exp2_12 = approx_exp2(_sub_13);
                float _add_12;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(1.0f), "f"(_exp2_12));
                float _rcp_12 = approx_rcp(_add_12);
                float sig_00_86 = _rcp_12;
                float _mul_50;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_50) : "f"(x1_01_73), "f"(fused));
                float _sub_14;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(0.0f), "f"(_mul_50));
                float _exp2_13 = approx_exp2(_sub_14);
                float _add_13;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(1.0f), "f"(_exp2_13));
                float _rcp_13 = approx_rcp(_add_13);
                float sig_01_87 = _rcp_13;
                float _mul_51;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_51) : "f"(x1_10_76), "f"(fused));
                float _sub_15;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_15) : "f"(0.0f), "f"(_mul_51));
                float _exp2_14 = approx_exp2(_sub_15);
                float _add_14;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(1.0f), "f"(_exp2_14));
                float _rcp_14 = approx_rcp(_add_14);
                float sig_10_88 = _rcp_14;
                float _mul_52;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_52) : "f"(x1_11_77), "f"(fused));
                float _sub_16;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_16) : "f"(0.0f), "f"(_mul_52));
                float _exp2_15 = approx_exp2(_sub_16);
                float _add_15;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(1.0f), "f"(_exp2_15));
                float _rcp_15 = approx_rcp(_add_15);
                float sig_11_89 = _rcp_15;
                float _mul_53;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_53) : "f"(x1_00_72), "f"(sig_00_86));
                float act_00_90 = _mul_53;
                float _mul_54;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_54) : "f"(x1_01_73), "f"(sig_01_87));
                float act_01_91 = _mul_54;
                float _mul_55;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_55) : "f"(x1_10_76), "f"(sig_10_88));
                float act_10_92 = _mul_55;
                float _mul_56;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_56) : "f"(x1_11_77), "f"(sig_11_89));
                float act_11_93 = _mul_56;
                float _min_28;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_28) : "f"(act_00_90), "f"(cl));
                act_00_90 = _min_28;
                float _min_29;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_29) : "f"(act_01_91), "f"(cl));
                act_01_91 = _min_29;
                float _min_30;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_30) : "f"(act_10_92), "f"(cl));
                act_10_92 = _min_30;
                float _min_31;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_31) : "f"(act_11_93), "f"(cl));
                act_11_93 = _min_31;
                float _mul_57;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_57) : "f"(lin_00_82), "f"(act_00_90));
                float y_00_94 = _mul_57;
                float _mul_58;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_58) : "f"(lin_10_84), "f"(act_10_92));
                float y_10_95 = _mul_58;
                float _mul_59;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_59) : "f"(lin_01_83), "f"(act_01_91));
                float y_01_96 = _mul_59;
                float _mul_60;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_60) : "f"(lin_11_85), "f"(act_11_93));
                float y_11_97 = _mul_60;
                float _mul_61;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_61) : "f"(y_00_94), "f"(sc));
                quad[0] = _mul_61;
                float _mul_62;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_62) : "f"(y_10_95), "f"(sc));
                quad[1] = _mul_62;
                float _mul_63;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_63) : "f"(y_01_96), "f"(sc));
                quad[2] = _mul_63;
                float _mul_64;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_64) : "f"(y_11_97), "f"(sc));
                quad[3] = _mul_64;
                uint32_t _fp8_3[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_3[0] = _packed;
                }
                int off0_98 = local_token0_68 * 64 + base_row;
                int off1_99 = local_token1_69 * 64 + base_row;
                int swz0_100 = off0_98 ^ (off0_98 >> 7 & 3) << 4;
                int swz1_101 = off1_99 ^ (off1_99 >> 7 & 3) << 4;
                epi_staging[swz0_100 >> 1] = _fp8_3[0] & 65535;
                epi_staging[swz1_101 >> 1] = _fp8_3[0] >> 16;
                int local_token0_102 = lane_1 % 4 * 2 + 32;
                int local_token1_103 = local_token0_102 + 1;
                float x0_00_104 = _tmem_load_0[16];
                float x0_01_105 = _tmem_load_0[17];
                float x1_00_106 = _tmem_load_0[18];
                float x1_01_107 = _tmem_load_0[19];
                float x0_10_108 = _tmem_load_1[16];
                float x0_11_109 = _tmem_load_1[17];
                float x1_10_110 = _tmem_load_1[18];
                float x1_11_111 = _tmem_load_1[19];
                float _max_16;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_16) : "f"(x0_00_104), "f"(neg_cl));
                float _min_32;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_32) : "f"(_max_16), "f"(cl));
                float x0c_00_112 = _min_32;
                float _max_17;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_17) : "f"(x0_01_105), "f"(neg_cl));
                float _min_33;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_33) : "f"(_max_17), "f"(cl));
                float x0c_01_113 = _min_33;
                float _max_18;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_18) : "f"(x0_10_108), "f"(neg_cl));
                float _min_34;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_34) : "f"(_max_18), "f"(cl));
                float x0c_10_114 = _min_34;
                float _max_19;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_19) : "f"(x0_11_109), "f"(neg_cl));
                float _min_35;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_35) : "f"(_max_19), "f"(cl));
                float x0c_11_115 = _min_35;
                float _fma_16;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(x0c_00_112), "f"(sg), "f"(0.0f));
                float lin_00_116 = _fma_16;
                float _fma_17;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(x0c_01_113), "f"(sg), "f"(0.0f));
                float lin_01_117 = _fma_17;
                float _fma_18;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(x0c_10_114), "f"(sg), "f"(0.0f));
                float lin_10_118 = _fma_18;
                float _fma_19;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(x0c_11_115), "f"(sg), "f"(0.0f));
                float lin_11_119 = _fma_19;
                float _mul_65;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_65) : "f"(x1_00_106), "f"(fused));
                float _sub_17;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_17) : "f"(0.0f), "f"(_mul_65));
                float _exp2_16 = approx_exp2(_sub_17);
                float _add_16;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(1.0f), "f"(_exp2_16));
                float _rcp_16 = approx_rcp(_add_16);
                float sig_00_120 = _rcp_16;
                float _mul_66;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_66) : "f"(x1_01_107), "f"(fused));
                float _sub_18;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_18) : "f"(0.0f), "f"(_mul_66));
                float _exp2_17 = approx_exp2(_sub_18);
                float _add_17;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(1.0f), "f"(_exp2_17));
                float _rcp_17 = approx_rcp(_add_17);
                float sig_01_121 = _rcp_17;
                float _mul_67;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_67) : "f"(x1_10_110), "f"(fused));
                float _sub_19;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_19) : "f"(0.0f), "f"(_mul_67));
                float _exp2_18 = approx_exp2(_sub_19);
                float _add_18;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(1.0f), "f"(_exp2_18));
                float _rcp_18 = approx_rcp(_add_18);
                float sig_10_122 = _rcp_18;
                float _mul_68;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_68) : "f"(x1_11_111), "f"(fused));
                float _sub_20;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_20) : "f"(0.0f), "f"(_mul_68));
                float _exp2_19 = approx_exp2(_sub_20);
                float _add_19;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(1.0f), "f"(_exp2_19));
                float _rcp_19 = approx_rcp(_add_19);
                float sig_11_123 = _rcp_19;
                float _mul_69;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_69) : "f"(x1_00_106), "f"(sig_00_120));
                float act_00_124 = _mul_69;
                float _mul_70;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_70) : "f"(x1_01_107), "f"(sig_01_121));
                float act_01_125 = _mul_70;
                float _mul_71;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_71) : "f"(x1_10_110), "f"(sig_10_122));
                float act_10_126 = _mul_71;
                float _mul_72;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_72) : "f"(x1_11_111), "f"(sig_11_123));
                float act_11_127 = _mul_72;
                float _min_36;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_36) : "f"(act_00_124), "f"(cl));
                act_00_124 = _min_36;
                float _min_37;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_37) : "f"(act_01_125), "f"(cl));
                act_01_125 = _min_37;
                float _min_38;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_38) : "f"(act_10_126), "f"(cl));
                act_10_126 = _min_38;
                float _min_39;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_39) : "f"(act_11_127), "f"(cl));
                act_11_127 = _min_39;
                float _mul_73;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_73) : "f"(lin_00_116), "f"(act_00_124));
                float y_00_128 = _mul_73;
                float _mul_74;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_74) : "f"(lin_10_118), "f"(act_10_126));
                float y_10_129 = _mul_74;
                float _mul_75;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_75) : "f"(lin_01_117), "f"(act_01_125));
                float y_01_130 = _mul_75;
                float _mul_76;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_76) : "f"(lin_11_119), "f"(act_11_127));
                float y_11_131 = _mul_76;
                float _mul_77;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_77) : "f"(y_00_128), "f"(sc));
                quad[0] = _mul_77;
                float _mul_78;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_78) : "f"(y_10_129), "f"(sc));
                quad[1] = _mul_78;
                float _mul_79;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_79) : "f"(y_01_130), "f"(sc));
                quad[2] = _mul_79;
                float _mul_80;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_80) : "f"(y_11_131), "f"(sc));
                quad[3] = _mul_80;
                uint32_t _fp8_4[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_4[0] = _packed;
                }
                int off0_132 = local_token0_102 * 64 + base_row;
                int off1_133 = local_token1_103 * 64 + base_row;
                int swz0_134 = off0_132 ^ (off0_132 >> 7 & 3) << 4;
                int swz1_135 = off1_133 ^ (off1_133 >> 7 & 3) << 4;
                epi_staging[swz0_134 >> 1] = _fp8_4[0] & 65535;
                epi_staging[swz1_135 >> 1] = _fp8_4[0] >> 16;
                int local_token0_136 = lane_1 % 4 * 2 + 40;
                int local_token1_137 = local_token0_136 + 1;
                float x0_00_138 = _tmem_load_0[20];
                float x0_01_139 = _tmem_load_0[21];
                float x1_00_140 = _tmem_load_0[22];
                float x1_01_141 = _tmem_load_0[23];
                float x0_10_142 = _tmem_load_1[20];
                float x0_11_143 = _tmem_load_1[21];
                float x1_10_144 = _tmem_load_1[22];
                float x1_11_145 = _tmem_load_1[23];
                float _max_20;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_20) : "f"(x0_00_138), "f"(neg_cl));
                float _min_40;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_40) : "f"(_max_20), "f"(cl));
                float x0c_00_146 = _min_40;
                float _max_21;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_21) : "f"(x0_01_139), "f"(neg_cl));
                float _min_41;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_41) : "f"(_max_21), "f"(cl));
                float x0c_01_147 = _min_41;
                float _max_22;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_22) : "f"(x0_10_142), "f"(neg_cl));
                float _min_42;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_42) : "f"(_max_22), "f"(cl));
                float x0c_10_148 = _min_42;
                float _max_23;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_23) : "f"(x0_11_143), "f"(neg_cl));
                float _min_43;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_43) : "f"(_max_23), "f"(cl));
                float x0c_11_149 = _min_43;
                float _fma_20;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(x0c_00_146), "f"(sg), "f"(0.0f));
                float lin_00_150 = _fma_20;
                float _fma_21;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(x0c_01_147), "f"(sg), "f"(0.0f));
                float lin_01_151 = _fma_21;
                float _fma_22;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(x0c_10_148), "f"(sg), "f"(0.0f));
                float lin_10_152 = _fma_22;
                float _fma_23;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(x0c_11_149), "f"(sg), "f"(0.0f));
                float lin_11_153 = _fma_23;
                float _mul_81;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_81) : "f"(x1_00_140), "f"(fused));
                float _sub_21;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_21) : "f"(0.0f), "f"(_mul_81));
                float _exp2_20 = approx_exp2(_sub_21);
                float _add_20;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(1.0f), "f"(_exp2_20));
                float _rcp_20 = approx_rcp(_add_20);
                float sig_00_154 = _rcp_20;
                float _mul_82;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_82) : "f"(x1_01_141), "f"(fused));
                float _sub_22;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_22) : "f"(0.0f), "f"(_mul_82));
                float _exp2_21 = approx_exp2(_sub_22);
                float _add_21;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(1.0f), "f"(_exp2_21));
                float _rcp_21 = approx_rcp(_add_21);
                float sig_01_155 = _rcp_21;
                float _mul_83;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_83) : "f"(x1_10_144), "f"(fused));
                float _sub_23;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_23) : "f"(0.0f), "f"(_mul_83));
                float _exp2_22 = approx_exp2(_sub_23);
                float _add_22;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(1.0f), "f"(_exp2_22));
                float _rcp_22 = approx_rcp(_add_22);
                float sig_10_156 = _rcp_22;
                float _mul_84;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_84) : "f"(x1_11_145), "f"(fused));
                float _sub_24;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_24) : "f"(0.0f), "f"(_mul_84));
                float _exp2_23 = approx_exp2(_sub_24);
                float _add_23;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(1.0f), "f"(_exp2_23));
                float _rcp_23 = approx_rcp(_add_23);
                float sig_11_157 = _rcp_23;
                float _mul_85;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_85) : "f"(x1_00_140), "f"(sig_00_154));
                float act_00_158 = _mul_85;
                float _mul_86;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_86) : "f"(x1_01_141), "f"(sig_01_155));
                float act_01_159 = _mul_86;
                float _mul_87;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_87) : "f"(x1_10_144), "f"(sig_10_156));
                float act_10_160 = _mul_87;
                float _mul_88;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_88) : "f"(x1_11_145), "f"(sig_11_157));
                float act_11_161 = _mul_88;
                float _min_44;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_44) : "f"(act_00_158), "f"(cl));
                act_00_158 = _min_44;
                float _min_45;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_45) : "f"(act_01_159), "f"(cl));
                act_01_159 = _min_45;
                float _min_46;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_46) : "f"(act_10_160), "f"(cl));
                act_10_160 = _min_46;
                float _min_47;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_47) : "f"(act_11_161), "f"(cl));
                act_11_161 = _min_47;
                float _mul_89;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_89) : "f"(lin_00_150), "f"(act_00_158));
                float y_00_162 = _mul_89;
                float _mul_90;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_90) : "f"(lin_10_152), "f"(act_10_160));
                float y_10_163 = _mul_90;
                float _mul_91;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_91) : "f"(lin_01_151), "f"(act_01_159));
                float y_01_164 = _mul_91;
                float _mul_92;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_92) : "f"(lin_11_153), "f"(act_11_161));
                float y_11_165 = _mul_92;
                float _mul_93;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_93) : "f"(y_00_162), "f"(sc));
                quad[0] = _mul_93;
                float _mul_94;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_94) : "f"(y_10_163), "f"(sc));
                quad[1] = _mul_94;
                float _mul_95;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_95) : "f"(y_01_164), "f"(sc));
                quad[2] = _mul_95;
                float _mul_96;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_96) : "f"(y_11_165), "f"(sc));
                quad[3] = _mul_96;
                uint32_t _fp8_5[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_5[0] = _packed;
                }
                int off0_166 = local_token0_136 * 64 + base_row;
                int off1_167 = local_token1_137 * 64 + base_row;
                int swz0_168 = off0_166 ^ (off0_166 >> 7 & 3) << 4;
                int swz1_169 = off1_167 ^ (off1_167 >> 7 & 3) << 4;
                epi_staging[swz0_168 >> 1] = _fp8_5[0] & 65535;
                epi_staging[swz1_169 >> 1] = _fp8_5[0] >> 16;
                int local_token0_170 = lane_1 % 4 * 2 + 48;
                int local_token1_171 = local_token0_170 + 1;
                float x0_00_172 = _tmem_load_0[24];
                float x0_01_173 = _tmem_load_0[25];
                float x1_00_174 = _tmem_load_0[26];
                float x1_01_175 = _tmem_load_0[27];
                float x0_10_176 = _tmem_load_1[24];
                float x0_11_177 = _tmem_load_1[25];
                float x1_10_178 = _tmem_load_1[26];
                float x1_11_179 = _tmem_load_1[27];
                float _max_24;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_24) : "f"(x0_00_172), "f"(neg_cl));
                float _min_48;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_48) : "f"(_max_24), "f"(cl));
                float x0c_00_180 = _min_48;
                float _max_25;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_25) : "f"(x0_01_173), "f"(neg_cl));
                float _min_49;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_49) : "f"(_max_25), "f"(cl));
                float x0c_01_181 = _min_49;
                float _max_26;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_26) : "f"(x0_10_176), "f"(neg_cl));
                float _min_50;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_50) : "f"(_max_26), "f"(cl));
                float x0c_10_182 = _min_50;
                float _max_27;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_27) : "f"(x0_11_177), "f"(neg_cl));
                float _min_51;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_51) : "f"(_max_27), "f"(cl));
                float x0c_11_183 = _min_51;
                float _fma_24;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(x0c_00_180), "f"(sg), "f"(0.0f));
                float lin_00_184 = _fma_24;
                float _fma_25;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(x0c_01_181), "f"(sg), "f"(0.0f));
                float lin_01_185 = _fma_25;
                float _fma_26;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(x0c_10_182), "f"(sg), "f"(0.0f));
                float lin_10_186 = _fma_26;
                float _fma_27;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(x0c_11_183), "f"(sg), "f"(0.0f));
                float lin_11_187 = _fma_27;
                float _mul_97;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_97) : "f"(x1_00_174), "f"(fused));
                float _sub_25;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_25) : "f"(0.0f), "f"(_mul_97));
                float _exp2_24 = approx_exp2(_sub_25);
                float _add_24;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(1.0f), "f"(_exp2_24));
                float _rcp_24 = approx_rcp(_add_24);
                float sig_00_188 = _rcp_24;
                float _mul_98;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_98) : "f"(x1_01_175), "f"(fused));
                float _sub_26;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_26) : "f"(0.0f), "f"(_mul_98));
                float _exp2_25 = approx_exp2(_sub_26);
                float _add_25;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(1.0f), "f"(_exp2_25));
                float _rcp_25 = approx_rcp(_add_25);
                float sig_01_189 = _rcp_25;
                float _mul_99;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_99) : "f"(x1_10_178), "f"(fused));
                float _sub_27;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_27) : "f"(0.0f), "f"(_mul_99));
                float _exp2_26 = approx_exp2(_sub_27);
                float _add_26;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(1.0f), "f"(_exp2_26));
                float _rcp_26 = approx_rcp(_add_26);
                float sig_10_190 = _rcp_26;
                float _mul_100;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_100) : "f"(x1_11_179), "f"(fused));
                float _sub_28;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_28) : "f"(0.0f), "f"(_mul_100));
                float _exp2_27 = approx_exp2(_sub_28);
                float _add_27;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(1.0f), "f"(_exp2_27));
                float _rcp_27 = approx_rcp(_add_27);
                float sig_11_191 = _rcp_27;
                float _mul_101;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_101) : "f"(x1_00_174), "f"(sig_00_188));
                float act_00_192 = _mul_101;
                float _mul_102;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_102) : "f"(x1_01_175), "f"(sig_01_189));
                float act_01_193 = _mul_102;
                float _mul_103;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_103) : "f"(x1_10_178), "f"(sig_10_190));
                float act_10_194 = _mul_103;
                float _mul_104;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_104) : "f"(x1_11_179), "f"(sig_11_191));
                float act_11_195 = _mul_104;
                float _min_52;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_52) : "f"(act_00_192), "f"(cl));
                act_00_192 = _min_52;
                float _min_53;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_53) : "f"(act_01_193), "f"(cl));
                act_01_193 = _min_53;
                float _min_54;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_54) : "f"(act_10_194), "f"(cl));
                act_10_194 = _min_54;
                float _min_55;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_55) : "f"(act_11_195), "f"(cl));
                act_11_195 = _min_55;
                float _mul_105;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_105) : "f"(lin_00_184), "f"(act_00_192));
                float y_00_196 = _mul_105;
                float _mul_106;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_106) : "f"(lin_10_186), "f"(act_10_194));
                float y_10_197 = _mul_106;
                float _mul_107;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_107) : "f"(lin_01_185), "f"(act_01_193));
                float y_01_198 = _mul_107;
                float _mul_108;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_108) : "f"(lin_11_187), "f"(act_11_195));
                float y_11_199 = _mul_108;
                float _mul_109;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_109) : "f"(y_00_196), "f"(sc));
                quad[0] = _mul_109;
                float _mul_110;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_110) : "f"(y_10_197), "f"(sc));
                quad[1] = _mul_110;
                float _mul_111;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_111) : "f"(y_01_198), "f"(sc));
                quad[2] = _mul_111;
                float _mul_112;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_112) : "f"(y_11_199), "f"(sc));
                quad[3] = _mul_112;
                uint32_t _fp8_6[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_6[0] = _packed;
                }
                int off0_200 = local_token0_170 * 64 + base_row;
                int off1_201 = local_token1_171 * 64 + base_row;
                int swz0_202 = off0_200 ^ (off0_200 >> 7 & 3) << 4;
                int swz1_203 = off1_201 ^ (off1_201 >> 7 & 3) << 4;
                epi_staging[swz0_202 >> 1] = _fp8_6[0] & 65535;
                epi_staging[swz1_203 >> 1] = _fp8_6[0] >> 16;
                int local_token0_204 = lane_1 % 4 * 2 + 56;
                int local_token1_205 = local_token0_204 + 1;
                float x0_00_206 = _tmem_load_0[28];
                float x0_01_207 = _tmem_load_0[29];
                float x1_00_208 = _tmem_load_0[30];
                float x1_01_209 = _tmem_load_0[31];
                float x0_10_210 = _tmem_load_1[28];
                float x0_11_211 = _tmem_load_1[29];
                float x1_10_212 = _tmem_load_1[30];
                float x1_11_213 = _tmem_load_1[31];
                float _max_28;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_28) : "f"(x0_00_206), "f"(neg_cl));
                float _min_56;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_56) : "f"(_max_28), "f"(cl));
                float x0c_00_214 = _min_56;
                float _max_29;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_29) : "f"(x0_01_207), "f"(neg_cl));
                float _min_57;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_57) : "f"(_max_29), "f"(cl));
                float x0c_01_215 = _min_57;
                float _max_30;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_30) : "f"(x0_10_210), "f"(neg_cl));
                float _min_58;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_58) : "f"(_max_30), "f"(cl));
                float x0c_10_216 = _min_58;
                float _max_31;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_31) : "f"(x0_11_211), "f"(neg_cl));
                float _min_59;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_59) : "f"(_max_31), "f"(cl));
                float x0c_11_217 = _min_59;
                float _fma_28;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(x0c_00_214), "f"(sg), "f"(0.0f));
                float lin_00_218 = _fma_28;
                float _fma_29;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(x0c_01_215), "f"(sg), "f"(0.0f));
                float lin_01_219 = _fma_29;
                float _fma_30;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(x0c_10_216), "f"(sg), "f"(0.0f));
                float lin_10_220 = _fma_30;
                float _fma_31;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(x0c_11_217), "f"(sg), "f"(0.0f));
                float lin_11_221 = _fma_31;
                float _mul_113;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_113) : "f"(x1_00_208), "f"(fused));
                float _sub_29;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_29) : "f"(0.0f), "f"(_mul_113));
                float _exp2_28 = approx_exp2(_sub_29);
                float _add_28;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(1.0f), "f"(_exp2_28));
                float _rcp_28 = approx_rcp(_add_28);
                float sig_00_222 = _rcp_28;
                float _mul_114;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_114) : "f"(x1_01_209), "f"(fused));
                float _sub_30;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_30) : "f"(0.0f), "f"(_mul_114));
                float _exp2_29 = approx_exp2(_sub_30);
                float _add_29;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(1.0f), "f"(_exp2_29));
                float _rcp_29 = approx_rcp(_add_29);
                float sig_01_223 = _rcp_29;
                float _mul_115;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_115) : "f"(x1_10_212), "f"(fused));
                float _sub_31;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_31) : "f"(0.0f), "f"(_mul_115));
                float _exp2_30 = approx_exp2(_sub_31);
                float _add_30;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(1.0f), "f"(_exp2_30));
                float _rcp_30 = approx_rcp(_add_30);
                float sig_10_224 = _rcp_30;
                float _mul_116;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_116) : "f"(x1_11_213), "f"(fused));
                float _sub_32;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_32) : "f"(0.0f), "f"(_mul_116));
                float _exp2_31 = approx_exp2(_sub_32);
                float _add_31;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(1.0f), "f"(_exp2_31));
                float _rcp_31 = approx_rcp(_add_31);
                float sig_11_225 = _rcp_31;
                float _mul_117;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_117) : "f"(x1_00_208), "f"(sig_00_222));
                float act_00_226 = _mul_117;
                float _mul_118;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_118) : "f"(x1_01_209), "f"(sig_01_223));
                float act_01_227 = _mul_118;
                float _mul_119;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_119) : "f"(x1_10_212), "f"(sig_10_224));
                float act_10_228 = _mul_119;
                float _mul_120;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_120) : "f"(x1_11_213), "f"(sig_11_225));
                float act_11_229 = _mul_120;
                float _min_60;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_60) : "f"(act_00_226), "f"(cl));
                act_00_226 = _min_60;
                float _min_61;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_61) : "f"(act_01_227), "f"(cl));
                act_01_227 = _min_61;
                float _min_62;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_62) : "f"(act_10_228), "f"(cl));
                act_10_228 = _min_62;
                float _min_63;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_63) : "f"(act_11_229), "f"(cl));
                act_11_229 = _min_63;
                float _mul_121;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_121) : "f"(lin_00_218), "f"(act_00_226));
                float y_00_230 = _mul_121;
                float _mul_122;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_122) : "f"(lin_10_220), "f"(act_10_228));
                float y_10_231 = _mul_122;
                float _mul_123;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_123) : "f"(lin_01_219), "f"(act_01_227));
                float y_01_232 = _mul_123;
                float _mul_124;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_124) : "f"(lin_11_221), "f"(act_11_229));
                float y_11_233 = _mul_124;
                float _mul_125;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_125) : "f"(y_00_230), "f"(sc));
                quad[0] = _mul_125;
                float _mul_126;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_126) : "f"(y_10_231), "f"(sc));
                quad[1] = _mul_126;
                float _mul_127;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_127) : "f"(y_01_232), "f"(sc));
                quad[2] = _mul_127;
                float _mul_128;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_128) : "f"(y_11_233), "f"(sc));
                quad[3] = _mul_128;
                uint32_t _fp8_7[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_7[0] = _packed;
                }
                int off0_234 = local_token0_204 * 64 + base_row;
                int off1_235 = local_token1_205 * 64 + base_row;
                int swz0_236 = off0_234 ^ (off0_234 >> 7 & 3) << 4;
                int swz1_237 = off1_235 ^ (off1_235 >> 7 & 3) << 4;
                epi_staging[swz0_236 >> 1] = _fp8_7[0] & 65535;
                epi_staging[swz1_237 >> 1] = _fp8_7[0] >> 16;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows, 1073741824, n_tile * 256 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int acc_offset_238 = acc_stage * 256 + 64;
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                    : "r"(taddr + (unsigned int)acc_offset_238));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_238));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row_239 = warp_0 * 16 + lane_1 / 4 * 2;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int local_token0_240 = lane_1 % 4 * 2;
                int local_token1_241 = local_token0_240 + 1;
                float x0_00_242 = _tmem_load_2[0];
                float x0_01_243 = _tmem_load_2[1];
                float x1_00_244 = _tmem_load_2[2];
                float x1_01_245 = _tmem_load_2[3];
                float x0_10_246 = _tmem_load_3[0];
                float x0_11_247 = _tmem_load_3[1];
                float x1_10_248 = _tmem_load_3[2];
                float x1_11_249 = _tmem_load_3[3];
                float _max_32;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_32) : "f"(x0_00_242), "f"(neg_cl));
                float _min_64;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_64) : "f"(_max_32), "f"(cl));
                float x0c_00_250 = _min_64;
                float _max_33;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_33) : "f"(x0_01_243), "f"(neg_cl));
                float _min_65;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_65) : "f"(_max_33), "f"(cl));
                float x0c_01_251 = _min_65;
                float _max_34;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_34) : "f"(x0_10_246), "f"(neg_cl));
                float _min_66;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_66) : "f"(_max_34), "f"(cl));
                float x0c_10_252 = _min_66;
                float _max_35;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_35) : "f"(x0_11_247), "f"(neg_cl));
                float _min_67;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_67) : "f"(_max_35), "f"(cl));
                float x0c_11_253 = _min_67;
                float _fma_32;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(x0c_00_250), "f"(sg), "f"(0.0f));
                float lin_00_254 = _fma_32;
                float _fma_33;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(x0c_01_251), "f"(sg), "f"(0.0f));
                float lin_01_255 = _fma_33;
                float _fma_34;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(x0c_10_252), "f"(sg), "f"(0.0f));
                float lin_10_256 = _fma_34;
                float _fma_35;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(x0c_11_253), "f"(sg), "f"(0.0f));
                float lin_11_257 = _fma_35;
                float _mul_129;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_129) : "f"(x1_00_244), "f"(fused));
                float _sub_33;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_33) : "f"(0.0f), "f"(_mul_129));
                float _exp2_32 = approx_exp2(_sub_33);
                float _add_32;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(1.0f), "f"(_exp2_32));
                float _rcp_32 = approx_rcp(_add_32);
                float sig_00_258 = _rcp_32;
                float _mul_130;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_130) : "f"(x1_01_245), "f"(fused));
                float _sub_34;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_34) : "f"(0.0f), "f"(_mul_130));
                float _exp2_33 = approx_exp2(_sub_34);
                float _add_33;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(1.0f), "f"(_exp2_33));
                float _rcp_33 = approx_rcp(_add_33);
                float sig_01_259 = _rcp_33;
                float _mul_131;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_131) : "f"(x1_10_248), "f"(fused));
                float _sub_35;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_35) : "f"(0.0f), "f"(_mul_131));
                float _exp2_34 = approx_exp2(_sub_35);
                float _add_34;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(1.0f), "f"(_exp2_34));
                float _rcp_34 = approx_rcp(_add_34);
                float sig_10_260 = _rcp_34;
                float _mul_132;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_132) : "f"(x1_11_249), "f"(fused));
                float _sub_36;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_36) : "f"(0.0f), "f"(_mul_132));
                float _exp2_35 = approx_exp2(_sub_36);
                float _add_35;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(1.0f), "f"(_exp2_35));
                float _rcp_35 = approx_rcp(_add_35);
                float sig_11_261 = _rcp_35;
                float _mul_133;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_133) : "f"(x1_00_244), "f"(sig_00_258));
                float act_00_262 = _mul_133;
                float _mul_134;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_134) : "f"(x1_01_245), "f"(sig_01_259));
                float act_01_263 = _mul_134;
                float _mul_135;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_135) : "f"(x1_10_248), "f"(sig_10_260));
                float act_10_264 = _mul_135;
                float _mul_136;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_136) : "f"(x1_11_249), "f"(sig_11_261));
                float act_11_265 = _mul_136;
                float _min_68;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_68) : "f"(act_00_262), "f"(cl));
                act_00_262 = _min_68;
                float _min_69;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_69) : "f"(act_01_263), "f"(cl));
                act_01_263 = _min_69;
                float _min_70;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_70) : "f"(act_10_264), "f"(cl));
                act_10_264 = _min_70;
                float _min_71;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_71) : "f"(act_11_265), "f"(cl));
                act_11_265 = _min_71;
                float _mul_137;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_137) : "f"(lin_00_254), "f"(act_00_262));
                float y_00_266 = _mul_137;
                float _mul_138;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_138) : "f"(lin_10_256), "f"(act_10_264));
                float y_10_267 = _mul_138;
                float _mul_139;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_139) : "f"(lin_01_255), "f"(act_01_263));
                float y_01_268 = _mul_139;
                float _mul_140;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_140) : "f"(lin_11_257), "f"(act_11_265));
                float y_11_269 = _mul_140;
                float _mul_141;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_141) : "f"(y_00_266), "f"(sc));
                quad[0] = _mul_141;
                float _mul_142;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_142) : "f"(y_10_267), "f"(sc));
                quad[1] = _mul_142;
                float _mul_143;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_143) : "f"(y_01_268), "f"(sc));
                quad[2] = _mul_143;
                float _mul_144;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_144) : "f"(y_11_269), "f"(sc));
                quad[3] = _mul_144;
                uint32_t _fp8_8[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_8[0] = _packed;
                }
                int off0_270 = local_token0_240 * 64 + base_row_239;
                int off1_271 = local_token1_241 * 64 + base_row_239;
                int swz0_272 = off0_270 ^ (off0_270 >> 7 & 3) << 4;
                int swz1_273 = off1_271 ^ (off1_271 >> 7 & 3) << 4;
                epi_staging[swz0_272 >> 1] = _fp8_8[0] & 65535;
                epi_staging[swz1_273 >> 1] = _fp8_8[0] >> 16;
                int local_token0_274 = lane_1 % 4 * 2 + 8;
                int local_token1_275 = local_token0_274 + 1;
                float x0_00_276 = _tmem_load_2[4];
                float x0_01_277 = _tmem_load_2[5];
                float x1_00_278 = _tmem_load_2[6];
                float x1_01_279 = _tmem_load_2[7];
                float x0_10_280 = _tmem_load_3[4];
                float x0_11_281 = _tmem_load_3[5];
                float x1_10_282 = _tmem_load_3[6];
                float x1_11_283 = _tmem_load_3[7];
                float _max_36;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_36) : "f"(x0_00_276), "f"(neg_cl));
                float _min_72;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_72) : "f"(_max_36), "f"(cl));
                float x0c_00_284 = _min_72;
                float _max_37;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_37) : "f"(x0_01_277), "f"(neg_cl));
                float _min_73;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_73) : "f"(_max_37), "f"(cl));
                float x0c_01_285 = _min_73;
                float _max_38;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_38) : "f"(x0_10_280), "f"(neg_cl));
                float _min_74;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_74) : "f"(_max_38), "f"(cl));
                float x0c_10_286 = _min_74;
                float _max_39;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_39) : "f"(x0_11_281), "f"(neg_cl));
                float _min_75;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_75) : "f"(_max_39), "f"(cl));
                float x0c_11_287 = _min_75;
                float _fma_36;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(x0c_00_284), "f"(sg), "f"(0.0f));
                float lin_00_288 = _fma_36;
                float _fma_37;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(x0c_01_285), "f"(sg), "f"(0.0f));
                float lin_01_289 = _fma_37;
                float _fma_38;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(x0c_10_286), "f"(sg), "f"(0.0f));
                float lin_10_290 = _fma_38;
                float _fma_39;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(x0c_11_287), "f"(sg), "f"(0.0f));
                float lin_11_291 = _fma_39;
                float _mul_145;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_145) : "f"(x1_00_278), "f"(fused));
                float _sub_37;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_37) : "f"(0.0f), "f"(_mul_145));
                float _exp2_36 = approx_exp2(_sub_37);
                float _add_36;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(1.0f), "f"(_exp2_36));
                float _rcp_36 = approx_rcp(_add_36);
                float sig_00_292 = _rcp_36;
                float _mul_146;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_146) : "f"(x1_01_279), "f"(fused));
                float _sub_38;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_38) : "f"(0.0f), "f"(_mul_146));
                float _exp2_37 = approx_exp2(_sub_38);
                float _add_37;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(1.0f), "f"(_exp2_37));
                float _rcp_37 = approx_rcp(_add_37);
                float sig_01_293 = _rcp_37;
                float _mul_147;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_147) : "f"(x1_10_282), "f"(fused));
                float _sub_39;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_39) : "f"(0.0f), "f"(_mul_147));
                float _exp2_38 = approx_exp2(_sub_39);
                float _add_38;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(1.0f), "f"(_exp2_38));
                float _rcp_38 = approx_rcp(_add_38);
                float sig_10_294 = _rcp_38;
                float _mul_148;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_148) : "f"(x1_11_283), "f"(fused));
                float _sub_40;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_40) : "f"(0.0f), "f"(_mul_148));
                float _exp2_39 = approx_exp2(_sub_40);
                float _add_39;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(1.0f), "f"(_exp2_39));
                float _rcp_39 = approx_rcp(_add_39);
                float sig_11_295 = _rcp_39;
                float _mul_149;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_149) : "f"(x1_00_278), "f"(sig_00_292));
                float act_00_296 = _mul_149;
                float _mul_150;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_150) : "f"(x1_01_279), "f"(sig_01_293));
                float act_01_297 = _mul_150;
                float _mul_151;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_151) : "f"(x1_10_282), "f"(sig_10_294));
                float act_10_298 = _mul_151;
                float _mul_152;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_152) : "f"(x1_11_283), "f"(sig_11_295));
                float act_11_299 = _mul_152;
                float _min_76;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_76) : "f"(act_00_296), "f"(cl));
                act_00_296 = _min_76;
                float _min_77;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_77) : "f"(act_01_297), "f"(cl));
                act_01_297 = _min_77;
                float _min_78;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_78) : "f"(act_10_298), "f"(cl));
                act_10_298 = _min_78;
                float _min_79;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_79) : "f"(act_11_299), "f"(cl));
                act_11_299 = _min_79;
                float _mul_153;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_153) : "f"(lin_00_288), "f"(act_00_296));
                float y_00_300 = _mul_153;
                float _mul_154;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_154) : "f"(lin_10_290), "f"(act_10_298));
                float y_10_301 = _mul_154;
                float _mul_155;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_155) : "f"(lin_01_289), "f"(act_01_297));
                float y_01_302 = _mul_155;
                float _mul_156;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_156) : "f"(lin_11_291), "f"(act_11_299));
                float y_11_303 = _mul_156;
                float _mul_157;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_157) : "f"(y_00_300), "f"(sc));
                quad[0] = _mul_157;
                float _mul_158;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_158) : "f"(y_10_301), "f"(sc));
                quad[1] = _mul_158;
                float _mul_159;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_159) : "f"(y_01_302), "f"(sc));
                quad[2] = _mul_159;
                float _mul_160;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_160) : "f"(y_11_303), "f"(sc));
                quad[3] = _mul_160;
                uint32_t _fp8_9[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_9[0] = _packed;
                }
                int off0_304 = local_token0_274 * 64 + base_row_239;
                int off1_305 = local_token1_275 * 64 + base_row_239;
                int swz0_306 = off0_304 ^ (off0_304 >> 7 & 3) << 4;
                int swz1_307 = off1_305 ^ (off1_305 >> 7 & 3) << 4;
                epi_staging[swz0_306 >> 1] = _fp8_9[0] & 65535;
                epi_staging[swz1_307 >> 1] = _fp8_9[0] >> 16;
                int local_token0_308 = lane_1 % 4 * 2 + 16;
                int local_token1_309 = local_token0_308 + 1;
                float x0_00_310 = _tmem_load_2[8];
                float x0_01_311 = _tmem_load_2[9];
                float x1_00_312 = _tmem_load_2[10];
                float x1_01_313 = _tmem_load_2[11];
                float x0_10_314 = _tmem_load_3[8];
                float x0_11_315 = _tmem_load_3[9];
                float x1_10_316 = _tmem_load_3[10];
                float x1_11_317 = _tmem_load_3[11];
                float _max_40;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_40) : "f"(x0_00_310), "f"(neg_cl));
                float _min_80;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_80) : "f"(_max_40), "f"(cl));
                float x0c_00_318 = _min_80;
                float _max_41;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_41) : "f"(x0_01_311), "f"(neg_cl));
                float _min_81;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_81) : "f"(_max_41), "f"(cl));
                float x0c_01_319 = _min_81;
                float _max_42;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_42) : "f"(x0_10_314), "f"(neg_cl));
                float _min_82;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_82) : "f"(_max_42), "f"(cl));
                float x0c_10_320 = _min_82;
                float _max_43;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_43) : "f"(x0_11_315), "f"(neg_cl));
                float _min_83;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_83) : "f"(_max_43), "f"(cl));
                float x0c_11_321 = _min_83;
                float _fma_40;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(x0c_00_318), "f"(sg), "f"(0.0f));
                float lin_00_322 = _fma_40;
                float _fma_41;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(x0c_01_319), "f"(sg), "f"(0.0f));
                float lin_01_323 = _fma_41;
                float _fma_42;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(x0c_10_320), "f"(sg), "f"(0.0f));
                float lin_10_324 = _fma_42;
                float _fma_43;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(x0c_11_321), "f"(sg), "f"(0.0f));
                float lin_11_325 = _fma_43;
                float _mul_161;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_161) : "f"(x1_00_312), "f"(fused));
                float _sub_41;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_41) : "f"(0.0f), "f"(_mul_161));
                float _exp2_40 = approx_exp2(_sub_41);
                float _add_40;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(1.0f), "f"(_exp2_40));
                float _rcp_40 = approx_rcp(_add_40);
                float sig_00_326 = _rcp_40;
                float _mul_162;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_162) : "f"(x1_01_313), "f"(fused));
                float _sub_42;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_42) : "f"(0.0f), "f"(_mul_162));
                float _exp2_41 = approx_exp2(_sub_42);
                float _add_41;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(1.0f), "f"(_exp2_41));
                float _rcp_41 = approx_rcp(_add_41);
                float sig_01_327 = _rcp_41;
                float _mul_163;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_163) : "f"(x1_10_316), "f"(fused));
                float _sub_43;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_43) : "f"(0.0f), "f"(_mul_163));
                float _exp2_42 = approx_exp2(_sub_43);
                float _add_42;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(1.0f), "f"(_exp2_42));
                float _rcp_42 = approx_rcp(_add_42);
                float sig_10_328 = _rcp_42;
                float _mul_164;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_164) : "f"(x1_11_317), "f"(fused));
                float _sub_44;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_44) : "f"(0.0f), "f"(_mul_164));
                float _exp2_43 = approx_exp2(_sub_44);
                float _add_43;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(1.0f), "f"(_exp2_43));
                float _rcp_43 = approx_rcp(_add_43);
                float sig_11_329 = _rcp_43;
                float _mul_165;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_165) : "f"(x1_00_312), "f"(sig_00_326));
                float act_00_330 = _mul_165;
                float _mul_166;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_166) : "f"(x1_01_313), "f"(sig_01_327));
                float act_01_331 = _mul_166;
                float _mul_167;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_167) : "f"(x1_10_316), "f"(sig_10_328));
                float act_10_332 = _mul_167;
                float _mul_168;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_168) : "f"(x1_11_317), "f"(sig_11_329));
                float act_11_333 = _mul_168;
                float _min_84;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_84) : "f"(act_00_330), "f"(cl));
                act_00_330 = _min_84;
                float _min_85;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_85) : "f"(act_01_331), "f"(cl));
                act_01_331 = _min_85;
                float _min_86;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_86) : "f"(act_10_332), "f"(cl));
                act_10_332 = _min_86;
                float _min_87;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_87) : "f"(act_11_333), "f"(cl));
                act_11_333 = _min_87;
                float _mul_169;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_169) : "f"(lin_00_322), "f"(act_00_330));
                float y_00_334 = _mul_169;
                float _mul_170;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_170) : "f"(lin_10_324), "f"(act_10_332));
                float y_10_335 = _mul_170;
                float _mul_171;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_171) : "f"(lin_01_323), "f"(act_01_331));
                float y_01_336 = _mul_171;
                float _mul_172;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_172) : "f"(lin_11_325), "f"(act_11_333));
                float y_11_337 = _mul_172;
                float _mul_173;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_173) : "f"(y_00_334), "f"(sc));
                quad[0] = _mul_173;
                float _mul_174;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_174) : "f"(y_10_335), "f"(sc));
                quad[1] = _mul_174;
                float _mul_175;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_175) : "f"(y_01_336), "f"(sc));
                quad[2] = _mul_175;
                float _mul_176;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_176) : "f"(y_11_337), "f"(sc));
                quad[3] = _mul_176;
                uint32_t _fp8_10[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_10[0] = _packed;
                }
                int off0_338 = local_token0_308 * 64 + base_row_239;
                int off1_339 = local_token1_309 * 64 + base_row_239;
                int swz0_340 = off0_338 ^ (off0_338 >> 7 & 3) << 4;
                int swz1_341 = off1_339 ^ (off1_339 >> 7 & 3) << 4;
                epi_staging[swz0_340 >> 1] = _fp8_10[0] & 65535;
                epi_staging[swz1_341 >> 1] = _fp8_10[0] >> 16;
                int local_token0_342 = lane_1 % 4 * 2 + 24;
                int local_token1_343 = local_token0_342 + 1;
                float x0_00_344 = _tmem_load_2[12];
                float x0_01_345 = _tmem_load_2[13];
                float x1_00_346 = _tmem_load_2[14];
                float x1_01_347 = _tmem_load_2[15];
                float x0_10_348 = _tmem_load_3[12];
                float x0_11_349 = _tmem_load_3[13];
                float x1_10_350 = _tmem_load_3[14];
                float x1_11_351 = _tmem_load_3[15];
                float _max_44;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_44) : "f"(x0_00_344), "f"(neg_cl));
                float _min_88;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_88) : "f"(_max_44), "f"(cl));
                float x0c_00_352 = _min_88;
                float _max_45;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_45) : "f"(x0_01_345), "f"(neg_cl));
                float _min_89;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_89) : "f"(_max_45), "f"(cl));
                float x0c_01_353 = _min_89;
                float _max_46;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_46) : "f"(x0_10_348), "f"(neg_cl));
                float _min_90;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_90) : "f"(_max_46), "f"(cl));
                float x0c_10_354 = _min_90;
                float _max_47;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_47) : "f"(x0_11_349), "f"(neg_cl));
                float _min_91;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_91) : "f"(_max_47), "f"(cl));
                float x0c_11_355 = _min_91;
                float _fma_44;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(x0c_00_352), "f"(sg), "f"(0.0f));
                float lin_00_356 = _fma_44;
                float _fma_45;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(x0c_01_353), "f"(sg), "f"(0.0f));
                float lin_01_357 = _fma_45;
                float _fma_46;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(x0c_10_354), "f"(sg), "f"(0.0f));
                float lin_10_358 = _fma_46;
                float _fma_47;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(x0c_11_355), "f"(sg), "f"(0.0f));
                float lin_11_359 = _fma_47;
                float _mul_177;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_177) : "f"(x1_00_346), "f"(fused));
                float _sub_45;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_45) : "f"(0.0f), "f"(_mul_177));
                float _exp2_44 = approx_exp2(_sub_45);
                float _add_44;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(1.0f), "f"(_exp2_44));
                float _rcp_44 = approx_rcp(_add_44);
                float sig_00_360 = _rcp_44;
                float _mul_178;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_178) : "f"(x1_01_347), "f"(fused));
                float _sub_46;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_46) : "f"(0.0f), "f"(_mul_178));
                float _exp2_45 = approx_exp2(_sub_46);
                float _add_45;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(1.0f), "f"(_exp2_45));
                float _rcp_45 = approx_rcp(_add_45);
                float sig_01_361 = _rcp_45;
                float _mul_179;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_179) : "f"(x1_10_350), "f"(fused));
                float _sub_47;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_47) : "f"(0.0f), "f"(_mul_179));
                float _exp2_46 = approx_exp2(_sub_47);
                float _add_46;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(1.0f), "f"(_exp2_46));
                float _rcp_46 = approx_rcp(_add_46);
                float sig_10_362 = _rcp_46;
                float _mul_180;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_180) : "f"(x1_11_351), "f"(fused));
                float _sub_48;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_48) : "f"(0.0f), "f"(_mul_180));
                float _exp2_47 = approx_exp2(_sub_48);
                float _add_47;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(1.0f), "f"(_exp2_47));
                float _rcp_47 = approx_rcp(_add_47);
                float sig_11_363 = _rcp_47;
                float _mul_181;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_181) : "f"(x1_00_346), "f"(sig_00_360));
                float act_00_364 = _mul_181;
                float _mul_182;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_182) : "f"(x1_01_347), "f"(sig_01_361));
                float act_01_365 = _mul_182;
                float _mul_183;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_183) : "f"(x1_10_350), "f"(sig_10_362));
                float act_10_366 = _mul_183;
                float _mul_184;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_184) : "f"(x1_11_351), "f"(sig_11_363));
                float act_11_367 = _mul_184;
                float _min_92;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_92) : "f"(act_00_364), "f"(cl));
                act_00_364 = _min_92;
                float _min_93;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_93) : "f"(act_01_365), "f"(cl));
                act_01_365 = _min_93;
                float _min_94;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_94) : "f"(act_10_366), "f"(cl));
                act_10_366 = _min_94;
                float _min_95;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_95) : "f"(act_11_367), "f"(cl));
                act_11_367 = _min_95;
                float _mul_185;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_185) : "f"(lin_00_356), "f"(act_00_364));
                float y_00_368 = _mul_185;
                float _mul_186;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_186) : "f"(lin_10_358), "f"(act_10_366));
                float y_10_369 = _mul_186;
                float _mul_187;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_187) : "f"(lin_01_357), "f"(act_01_365));
                float y_01_370 = _mul_187;
                float _mul_188;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_188) : "f"(lin_11_359), "f"(act_11_367));
                float y_11_371 = _mul_188;
                float _mul_189;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_189) : "f"(y_00_368), "f"(sc));
                quad[0] = _mul_189;
                float _mul_190;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_190) : "f"(y_10_369), "f"(sc));
                quad[1] = _mul_190;
                float _mul_191;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_191) : "f"(y_01_370), "f"(sc));
                quad[2] = _mul_191;
                float _mul_192;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_192) : "f"(y_11_371), "f"(sc));
                quad[3] = _mul_192;
                uint32_t _fp8_11[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_11[0] = _packed;
                }
                int off0_372 = local_token0_342 * 64 + base_row_239;
                int off1_373 = local_token1_343 * 64 + base_row_239;
                int swz0_374 = off0_372 ^ (off0_372 >> 7 & 3) << 4;
                int swz1_375 = off1_373 ^ (off1_373 >> 7 & 3) << 4;
                epi_staging[swz0_374 >> 1] = _fp8_11[0] & 65535;
                epi_staging[swz1_375 >> 1] = _fp8_11[0] >> 16;
                int local_token0_376 = lane_1 % 4 * 2 + 32;
                int local_token1_377 = local_token0_376 + 1;
                float x0_00_378 = _tmem_load_2[16];
                float x0_01_379 = _tmem_load_2[17];
                float x1_00_380 = _tmem_load_2[18];
                float x1_01_381 = _tmem_load_2[19];
                float x0_10_382 = _tmem_load_3[16];
                float x0_11_383 = _tmem_load_3[17];
                float x1_10_384 = _tmem_load_3[18];
                float x1_11_385 = _tmem_load_3[19];
                float _max_48;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_48) : "f"(x0_00_378), "f"(neg_cl));
                float _min_96;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_96) : "f"(_max_48), "f"(cl));
                float x0c_00_386 = _min_96;
                float _max_49;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_49) : "f"(x0_01_379), "f"(neg_cl));
                float _min_97;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_97) : "f"(_max_49), "f"(cl));
                float x0c_01_387 = _min_97;
                float _max_50;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_50) : "f"(x0_10_382), "f"(neg_cl));
                float _min_98;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_98) : "f"(_max_50), "f"(cl));
                float x0c_10_388 = _min_98;
                float _max_51;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_51) : "f"(x0_11_383), "f"(neg_cl));
                float _min_99;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_99) : "f"(_max_51), "f"(cl));
                float x0c_11_389 = _min_99;
                float _fma_48;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(x0c_00_386), "f"(sg), "f"(0.0f));
                float lin_00_390 = _fma_48;
                float _fma_49;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(x0c_01_387), "f"(sg), "f"(0.0f));
                float lin_01_391 = _fma_49;
                float _fma_50;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(x0c_10_388), "f"(sg), "f"(0.0f));
                float lin_10_392 = _fma_50;
                float _fma_51;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(x0c_11_389), "f"(sg), "f"(0.0f));
                float lin_11_393 = _fma_51;
                float _mul_193;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_193) : "f"(x1_00_380), "f"(fused));
                float _sub_49;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_49) : "f"(0.0f), "f"(_mul_193));
                float _exp2_48 = approx_exp2(_sub_49);
                float _add_48;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(1.0f), "f"(_exp2_48));
                float _rcp_48 = approx_rcp(_add_48);
                float sig_00_394 = _rcp_48;
                float _mul_194;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_194) : "f"(x1_01_381), "f"(fused));
                float _sub_50;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_50) : "f"(0.0f), "f"(_mul_194));
                float _exp2_49 = approx_exp2(_sub_50);
                float _add_49;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(1.0f), "f"(_exp2_49));
                float _rcp_49 = approx_rcp(_add_49);
                float sig_01_395 = _rcp_49;
                float _mul_195;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_195) : "f"(x1_10_384), "f"(fused));
                float _sub_51;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_51) : "f"(0.0f), "f"(_mul_195));
                float _exp2_50 = approx_exp2(_sub_51);
                float _add_50;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(1.0f), "f"(_exp2_50));
                float _rcp_50 = approx_rcp(_add_50);
                float sig_10_396 = _rcp_50;
                float _mul_196;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_196) : "f"(x1_11_385), "f"(fused));
                float _sub_52;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_52) : "f"(0.0f), "f"(_mul_196));
                float _exp2_51 = approx_exp2(_sub_52);
                float _add_51;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(1.0f), "f"(_exp2_51));
                float _rcp_51 = approx_rcp(_add_51);
                float sig_11_397 = _rcp_51;
                float _mul_197;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_197) : "f"(x1_00_380), "f"(sig_00_394));
                float act_00_398 = _mul_197;
                float _mul_198;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_198) : "f"(x1_01_381), "f"(sig_01_395));
                float act_01_399 = _mul_198;
                float _mul_199;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_199) : "f"(x1_10_384), "f"(sig_10_396));
                float act_10_400 = _mul_199;
                float _mul_200;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_200) : "f"(x1_11_385), "f"(sig_11_397));
                float act_11_401 = _mul_200;
                float _min_100;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_100) : "f"(act_00_398), "f"(cl));
                act_00_398 = _min_100;
                float _min_101;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_101) : "f"(act_01_399), "f"(cl));
                act_01_399 = _min_101;
                float _min_102;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_102) : "f"(act_10_400), "f"(cl));
                act_10_400 = _min_102;
                float _min_103;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_103) : "f"(act_11_401), "f"(cl));
                act_11_401 = _min_103;
                float _mul_201;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_201) : "f"(lin_00_390), "f"(act_00_398));
                float y_00_402 = _mul_201;
                float _mul_202;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_202) : "f"(lin_10_392), "f"(act_10_400));
                float y_10_403 = _mul_202;
                float _mul_203;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_203) : "f"(lin_01_391), "f"(act_01_399));
                float y_01_404 = _mul_203;
                float _mul_204;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_204) : "f"(lin_11_393), "f"(act_11_401));
                float y_11_405 = _mul_204;
                float _mul_205;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_205) : "f"(y_00_402), "f"(sc));
                quad[0] = _mul_205;
                float _mul_206;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_206) : "f"(y_10_403), "f"(sc));
                quad[1] = _mul_206;
                float _mul_207;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_207) : "f"(y_01_404), "f"(sc));
                quad[2] = _mul_207;
                float _mul_208;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_208) : "f"(y_11_405), "f"(sc));
                quad[3] = _mul_208;
                uint32_t _fp8_12[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_12[0] = _packed;
                }
                int off0_406 = local_token0_376 * 64 + base_row_239;
                int off1_407 = local_token1_377 * 64 + base_row_239;
                int swz0_408 = off0_406 ^ (off0_406 >> 7 & 3) << 4;
                int swz1_409 = off1_407 ^ (off1_407 >> 7 & 3) << 4;
                epi_staging[swz0_408 >> 1] = _fp8_12[0] & 65535;
                epi_staging[swz1_409 >> 1] = _fp8_12[0] >> 16;
                int local_token0_410 = lane_1 % 4 * 2 + 40;
                int local_token1_411 = local_token0_410 + 1;
                float x0_00_412 = _tmem_load_2[20];
                float x0_01_413 = _tmem_load_2[21];
                float x1_00_414 = _tmem_load_2[22];
                float x1_01_415 = _tmem_load_2[23];
                float x0_10_416 = _tmem_load_3[20];
                float x0_11_417 = _tmem_load_3[21];
                float x1_10_418 = _tmem_load_3[22];
                float x1_11_419 = _tmem_load_3[23];
                float _max_52;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_52) : "f"(x0_00_412), "f"(neg_cl));
                float _min_104;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_104) : "f"(_max_52), "f"(cl));
                float x0c_00_420 = _min_104;
                float _max_53;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_53) : "f"(x0_01_413), "f"(neg_cl));
                float _min_105;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_105) : "f"(_max_53), "f"(cl));
                float x0c_01_421 = _min_105;
                float _max_54;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_54) : "f"(x0_10_416), "f"(neg_cl));
                float _min_106;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_106) : "f"(_max_54), "f"(cl));
                float x0c_10_422 = _min_106;
                float _max_55;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_55) : "f"(x0_11_417), "f"(neg_cl));
                float _min_107;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_107) : "f"(_max_55), "f"(cl));
                float x0c_11_423 = _min_107;
                float _fma_52;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(x0c_00_420), "f"(sg), "f"(0.0f));
                float lin_00_424 = _fma_52;
                float _fma_53;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(x0c_01_421), "f"(sg), "f"(0.0f));
                float lin_01_425 = _fma_53;
                float _fma_54;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(x0c_10_422), "f"(sg), "f"(0.0f));
                float lin_10_426 = _fma_54;
                float _fma_55;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(x0c_11_423), "f"(sg), "f"(0.0f));
                float lin_11_427 = _fma_55;
                float _mul_209;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_209) : "f"(x1_00_414), "f"(fused));
                float _sub_53;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_53) : "f"(0.0f), "f"(_mul_209));
                float _exp2_52 = approx_exp2(_sub_53);
                float _add_52;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(1.0f), "f"(_exp2_52));
                float _rcp_52 = approx_rcp(_add_52);
                float sig_00_428 = _rcp_52;
                float _mul_210;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_210) : "f"(x1_01_415), "f"(fused));
                float _sub_54;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_54) : "f"(0.0f), "f"(_mul_210));
                float _exp2_53 = approx_exp2(_sub_54);
                float _add_53;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(1.0f), "f"(_exp2_53));
                float _rcp_53 = approx_rcp(_add_53);
                float sig_01_429 = _rcp_53;
                float _mul_211;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_211) : "f"(x1_10_418), "f"(fused));
                float _sub_55;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_55) : "f"(0.0f), "f"(_mul_211));
                float _exp2_54 = approx_exp2(_sub_55);
                float _add_54;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(1.0f), "f"(_exp2_54));
                float _rcp_54 = approx_rcp(_add_54);
                float sig_10_430 = _rcp_54;
                float _mul_212;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_212) : "f"(x1_11_419), "f"(fused));
                float _sub_56;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_56) : "f"(0.0f), "f"(_mul_212));
                float _exp2_55 = approx_exp2(_sub_56);
                float _add_55;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(1.0f), "f"(_exp2_55));
                float _rcp_55 = approx_rcp(_add_55);
                float sig_11_431 = _rcp_55;
                float _mul_213;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_213) : "f"(x1_00_414), "f"(sig_00_428));
                float act_00_432 = _mul_213;
                float _mul_214;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_214) : "f"(x1_01_415), "f"(sig_01_429));
                float act_01_433 = _mul_214;
                float _mul_215;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_215) : "f"(x1_10_418), "f"(sig_10_430));
                float act_10_434 = _mul_215;
                float _mul_216;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_216) : "f"(x1_11_419), "f"(sig_11_431));
                float act_11_435 = _mul_216;
                float _min_108;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_108) : "f"(act_00_432), "f"(cl));
                act_00_432 = _min_108;
                float _min_109;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_109) : "f"(act_01_433), "f"(cl));
                act_01_433 = _min_109;
                float _min_110;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_110) : "f"(act_10_434), "f"(cl));
                act_10_434 = _min_110;
                float _min_111;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_111) : "f"(act_11_435), "f"(cl));
                act_11_435 = _min_111;
                float _mul_217;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_217) : "f"(lin_00_424), "f"(act_00_432));
                float y_00_436 = _mul_217;
                float _mul_218;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_218) : "f"(lin_10_426), "f"(act_10_434));
                float y_10_437 = _mul_218;
                float _mul_219;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_219) : "f"(lin_01_425), "f"(act_01_433));
                float y_01_438 = _mul_219;
                float _mul_220;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_220) : "f"(lin_11_427), "f"(act_11_435));
                float y_11_439 = _mul_220;
                float _mul_221;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_221) : "f"(y_00_436), "f"(sc));
                quad[0] = _mul_221;
                float _mul_222;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_222) : "f"(y_10_437), "f"(sc));
                quad[1] = _mul_222;
                float _mul_223;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_223) : "f"(y_01_438), "f"(sc));
                quad[2] = _mul_223;
                float _mul_224;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_224) : "f"(y_11_439), "f"(sc));
                quad[3] = _mul_224;
                uint32_t _fp8_13[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_13[0] = _packed;
                }
                int off0_440 = local_token0_410 * 64 + base_row_239;
                int off1_441 = local_token1_411 * 64 + base_row_239;
                int swz0_442 = off0_440 ^ (off0_440 >> 7 & 3) << 4;
                int swz1_443 = off1_441 ^ (off1_441 >> 7 & 3) << 4;
                epi_staging[swz0_442 >> 1] = _fp8_13[0] & 65535;
                epi_staging[swz1_443 >> 1] = _fp8_13[0] >> 16;
                int local_token0_444 = lane_1 % 4 * 2 + 48;
                int local_token1_445 = local_token0_444 + 1;
                float x0_00_446 = _tmem_load_2[24];
                float x0_01_447 = _tmem_load_2[25];
                float x1_00_448 = _tmem_load_2[26];
                float x1_01_449 = _tmem_load_2[27];
                float x0_10_450 = _tmem_load_3[24];
                float x0_11_451 = _tmem_load_3[25];
                float x1_10_452 = _tmem_load_3[26];
                float x1_11_453 = _tmem_load_3[27];
                float _max_56;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_56) : "f"(x0_00_446), "f"(neg_cl));
                float _min_112;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_112) : "f"(_max_56), "f"(cl));
                float x0c_00_454 = _min_112;
                float _max_57;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_57) : "f"(x0_01_447), "f"(neg_cl));
                float _min_113;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_113) : "f"(_max_57), "f"(cl));
                float x0c_01_455 = _min_113;
                float _max_58;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_58) : "f"(x0_10_450), "f"(neg_cl));
                float _min_114;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_114) : "f"(_max_58), "f"(cl));
                float x0c_10_456 = _min_114;
                float _max_59;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_59) : "f"(x0_11_451), "f"(neg_cl));
                float _min_115;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_115) : "f"(_max_59), "f"(cl));
                float x0c_11_457 = _min_115;
                float _fma_56;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(x0c_00_454), "f"(sg), "f"(0.0f));
                float lin_00_458 = _fma_56;
                float _fma_57;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(x0c_01_455), "f"(sg), "f"(0.0f));
                float lin_01_459 = _fma_57;
                float _fma_58;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(x0c_10_456), "f"(sg), "f"(0.0f));
                float lin_10_460 = _fma_58;
                float _fma_59;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(x0c_11_457), "f"(sg), "f"(0.0f));
                float lin_11_461 = _fma_59;
                float _mul_225;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_225) : "f"(x1_00_448), "f"(fused));
                float _sub_57;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_57) : "f"(0.0f), "f"(_mul_225));
                float _exp2_56 = approx_exp2(_sub_57);
                float _add_56;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(1.0f), "f"(_exp2_56));
                float _rcp_56 = approx_rcp(_add_56);
                float sig_00_462 = _rcp_56;
                float _mul_226;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_226) : "f"(x1_01_449), "f"(fused));
                float _sub_58;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_58) : "f"(0.0f), "f"(_mul_226));
                float _exp2_57 = approx_exp2(_sub_58);
                float _add_57;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(1.0f), "f"(_exp2_57));
                float _rcp_57 = approx_rcp(_add_57);
                float sig_01_463 = _rcp_57;
                float _mul_227;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_227) : "f"(x1_10_452), "f"(fused));
                float _sub_59;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_59) : "f"(0.0f), "f"(_mul_227));
                float _exp2_58 = approx_exp2(_sub_59);
                float _add_58;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(1.0f), "f"(_exp2_58));
                float _rcp_58 = approx_rcp(_add_58);
                float sig_10_464 = _rcp_58;
                float _mul_228;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_228) : "f"(x1_11_453), "f"(fused));
                float _sub_60;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_60) : "f"(0.0f), "f"(_mul_228));
                float _exp2_59 = approx_exp2(_sub_60);
                float _add_59;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(1.0f), "f"(_exp2_59));
                float _rcp_59 = approx_rcp(_add_59);
                float sig_11_465 = _rcp_59;
                float _mul_229;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_229) : "f"(x1_00_448), "f"(sig_00_462));
                float act_00_466 = _mul_229;
                float _mul_230;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_230) : "f"(x1_01_449), "f"(sig_01_463));
                float act_01_467 = _mul_230;
                float _mul_231;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_231) : "f"(x1_10_452), "f"(sig_10_464));
                float act_10_468 = _mul_231;
                float _mul_232;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_232) : "f"(x1_11_453), "f"(sig_11_465));
                float act_11_469 = _mul_232;
                float _min_116;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_116) : "f"(act_00_466), "f"(cl));
                act_00_466 = _min_116;
                float _min_117;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_117) : "f"(act_01_467), "f"(cl));
                act_01_467 = _min_117;
                float _min_118;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_118) : "f"(act_10_468), "f"(cl));
                act_10_468 = _min_118;
                float _min_119;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_119) : "f"(act_11_469), "f"(cl));
                act_11_469 = _min_119;
                float _mul_233;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_233) : "f"(lin_00_458), "f"(act_00_466));
                float y_00_470 = _mul_233;
                float _mul_234;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_234) : "f"(lin_10_460), "f"(act_10_468));
                float y_10_471 = _mul_234;
                float _mul_235;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_235) : "f"(lin_01_459), "f"(act_01_467));
                float y_01_472 = _mul_235;
                float _mul_236;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_236) : "f"(lin_11_461), "f"(act_11_469));
                float y_11_473 = _mul_236;
                float _mul_237;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_237) : "f"(y_00_470), "f"(sc));
                quad[0] = _mul_237;
                float _mul_238;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_238) : "f"(y_10_471), "f"(sc));
                quad[1] = _mul_238;
                float _mul_239;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_239) : "f"(y_01_472), "f"(sc));
                quad[2] = _mul_239;
                float _mul_240;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_240) : "f"(y_11_473), "f"(sc));
                quad[3] = _mul_240;
                uint32_t _fp8_14[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_14[0] = _packed;
                }
                int off0_474 = local_token0_444 * 64 + base_row_239;
                int off1_475 = local_token1_445 * 64 + base_row_239;
                int swz0_476 = off0_474 ^ (off0_474 >> 7 & 3) << 4;
                int swz1_477 = off1_475 ^ (off1_475 >> 7 & 3) << 4;
                epi_staging[swz0_476 >> 1] = _fp8_14[0] & 65535;
                epi_staging[swz1_477 >> 1] = _fp8_14[0] >> 16;
                int local_token0_478 = lane_1 % 4 * 2 + 56;
                int local_token1_479 = local_token0_478 + 1;
                float x0_00_480 = _tmem_load_2[28];
                float x0_01_481 = _tmem_load_2[29];
                float x1_00_482 = _tmem_load_2[30];
                float x1_01_483 = _tmem_load_2[31];
                float x0_10_484 = _tmem_load_3[28];
                float x0_11_485 = _tmem_load_3[29];
                float x1_10_486 = _tmem_load_3[30];
                float x1_11_487 = _tmem_load_3[31];
                float _max_60;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_60) : "f"(x0_00_480), "f"(neg_cl));
                float _min_120;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_120) : "f"(_max_60), "f"(cl));
                float x0c_00_488 = _min_120;
                float _max_61;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_61) : "f"(x0_01_481), "f"(neg_cl));
                float _min_121;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_121) : "f"(_max_61), "f"(cl));
                float x0c_01_489 = _min_121;
                float _max_62;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_62) : "f"(x0_10_484), "f"(neg_cl));
                float _min_122;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_122) : "f"(_max_62), "f"(cl));
                float x0c_10_490 = _min_122;
                float _max_63;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_63) : "f"(x0_11_485), "f"(neg_cl));
                float _min_123;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_123) : "f"(_max_63), "f"(cl));
                float x0c_11_491 = _min_123;
                float _fma_60;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(x0c_00_488), "f"(sg), "f"(0.0f));
                float lin_00_492 = _fma_60;
                float _fma_61;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(x0c_01_489), "f"(sg), "f"(0.0f));
                float lin_01_493 = _fma_61;
                float _fma_62;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(x0c_10_490), "f"(sg), "f"(0.0f));
                float lin_10_494 = _fma_62;
                float _fma_63;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(x0c_11_491), "f"(sg), "f"(0.0f));
                float lin_11_495 = _fma_63;
                float _mul_241;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_241) : "f"(x1_00_482), "f"(fused));
                float _sub_61;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_61) : "f"(0.0f), "f"(_mul_241));
                float _exp2_60 = approx_exp2(_sub_61);
                float _add_60;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(1.0f), "f"(_exp2_60));
                float _rcp_60 = approx_rcp(_add_60);
                float sig_00_496 = _rcp_60;
                float _mul_242;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_242) : "f"(x1_01_483), "f"(fused));
                float _sub_62;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_62) : "f"(0.0f), "f"(_mul_242));
                float _exp2_61 = approx_exp2(_sub_62);
                float _add_61;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(1.0f), "f"(_exp2_61));
                float _rcp_61 = approx_rcp(_add_61);
                float sig_01_497 = _rcp_61;
                float _mul_243;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_243) : "f"(x1_10_486), "f"(fused));
                float _sub_63;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_63) : "f"(0.0f), "f"(_mul_243));
                float _exp2_62 = approx_exp2(_sub_63);
                float _add_62;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(1.0f), "f"(_exp2_62));
                float _rcp_62 = approx_rcp(_add_62);
                float sig_10_498 = _rcp_62;
                float _mul_244;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_244) : "f"(x1_11_487), "f"(fused));
                float _sub_64;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_64) : "f"(0.0f), "f"(_mul_244));
                float _exp2_63 = approx_exp2(_sub_64);
                float _add_63;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(1.0f), "f"(_exp2_63));
                float _rcp_63 = approx_rcp(_add_63);
                float sig_11_499 = _rcp_63;
                float _mul_245;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_245) : "f"(x1_00_482), "f"(sig_00_496));
                float act_00_500 = _mul_245;
                float _mul_246;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_246) : "f"(x1_01_483), "f"(sig_01_497));
                float act_01_501 = _mul_246;
                float _mul_247;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_247) : "f"(x1_10_486), "f"(sig_10_498));
                float act_10_502 = _mul_247;
                float _mul_248;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_248) : "f"(x1_11_487), "f"(sig_11_499));
                float act_11_503 = _mul_248;
                float _min_124;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_124) : "f"(act_00_500), "f"(cl));
                act_00_500 = _min_124;
                float _min_125;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_125) : "f"(act_01_501), "f"(cl));
                act_01_501 = _min_125;
                float _min_126;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_126) : "f"(act_10_502), "f"(cl));
                act_10_502 = _min_126;
                float _min_127;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_127) : "f"(act_11_503), "f"(cl));
                act_11_503 = _min_127;
                float _mul_249;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_249) : "f"(lin_00_492), "f"(act_00_500));
                float y_00_504 = _mul_249;
                float _mul_250;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_250) : "f"(lin_10_494), "f"(act_10_502));
                float y_10_505 = _mul_250;
                float _mul_251;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_251) : "f"(lin_01_493), "f"(act_01_501));
                float y_01_506 = _mul_251;
                float _mul_252;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_252) : "f"(lin_11_495), "f"(act_11_503));
                float y_11_507 = _mul_252;
                float _mul_253;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_253) : "f"(y_00_504), "f"(sc));
                quad[0] = _mul_253;
                float _mul_254;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_254) : "f"(y_10_505), "f"(sc));
                quad[1] = _mul_254;
                float _mul_255;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_255) : "f"(y_01_506), "f"(sc));
                quad[2] = _mul_255;
                float _mul_256;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_256) : "f"(y_11_507), "f"(sc));
                quad[3] = _mul_256;
                uint32_t _fp8_15[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_15[0] = _packed;
                }
                int off0_508 = local_token0_478 * 64 + base_row_239;
                int off1_509 = local_token1_479 * 64 + base_row_239;
                int swz0_510 = off0_508 ^ (off0_508 >> 7 & 3) << 4;
                int swz1_511 = off1_509 ^ (off1_509 >> 7 & 3) << 4;
                epi_staging[swz0_510 >> 1] = _fp8_15[0] & 65535;
                epi_staging[swz1_511 >> 1] = _fp8_15[0] >> 16;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_1 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_1 + 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int acc_offset_512 = acc_stage * 256 + 128;
                float _tmem_load_4[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                    : "r"(taddr + (unsigned int)acc_offset_512));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_5[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[31]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_512));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row_513 = warp_0 * 16 + lane_1 / 4 * 2;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int local_token0_514 = lane_1 % 4 * 2;
                int local_token1_515 = local_token0_514 + 1;
                float x0_00_516 = _tmem_load_4[0];
                float x0_01_517 = _tmem_load_4[1];
                float x1_00_518 = _tmem_load_4[2];
                float x1_01_519 = _tmem_load_4[3];
                float x0_10_520 = _tmem_load_5[0];
                float x0_11_521 = _tmem_load_5[1];
                float x1_10_522 = _tmem_load_5[2];
                float x1_11_523 = _tmem_load_5[3];
                float _max_64;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_64) : "f"(x0_00_516), "f"(neg_cl));
                float _min_128;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_128) : "f"(_max_64), "f"(cl));
                float x0c_00_524 = _min_128;
                float _max_65;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_65) : "f"(x0_01_517), "f"(neg_cl));
                float _min_129;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_129) : "f"(_max_65), "f"(cl));
                float x0c_01_525 = _min_129;
                float _max_66;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_66) : "f"(x0_10_520), "f"(neg_cl));
                float _min_130;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_130) : "f"(_max_66), "f"(cl));
                float x0c_10_526 = _min_130;
                float _max_67;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_67) : "f"(x0_11_521), "f"(neg_cl));
                float _min_131;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_131) : "f"(_max_67), "f"(cl));
                float x0c_11_527 = _min_131;
                float _fma_64;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(x0c_00_524), "f"(sg), "f"(0.0f));
                float lin_00_528 = _fma_64;
                float _fma_65;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(x0c_01_525), "f"(sg), "f"(0.0f));
                float lin_01_529 = _fma_65;
                float _fma_66;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(x0c_10_526), "f"(sg), "f"(0.0f));
                float lin_10_530 = _fma_66;
                float _fma_67;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_67) : "f"(x0c_11_527), "f"(sg), "f"(0.0f));
                float lin_11_531 = _fma_67;
                float _mul_257;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_257) : "f"(x1_00_518), "f"(fused));
                float _sub_65;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_65) : "f"(0.0f), "f"(_mul_257));
                float _exp2_64 = approx_exp2(_sub_65);
                float _add_64;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(1.0f), "f"(_exp2_64));
                float _rcp_64 = approx_rcp(_add_64);
                float sig_00_532 = _rcp_64;
                float _mul_258;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_258) : "f"(x1_01_519), "f"(fused));
                float _sub_66;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_66) : "f"(0.0f), "f"(_mul_258));
                float _exp2_65 = approx_exp2(_sub_66);
                float _add_65;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(1.0f), "f"(_exp2_65));
                float _rcp_65 = approx_rcp(_add_65);
                float sig_01_533 = _rcp_65;
                float _mul_259;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_259) : "f"(x1_10_522), "f"(fused));
                float _sub_67;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_67) : "f"(0.0f), "f"(_mul_259));
                float _exp2_66 = approx_exp2(_sub_67);
                float _add_66;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(1.0f), "f"(_exp2_66));
                float _rcp_66 = approx_rcp(_add_66);
                float sig_10_534 = _rcp_66;
                float _mul_260;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_260) : "f"(x1_11_523), "f"(fused));
                float _sub_68;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_68) : "f"(0.0f), "f"(_mul_260));
                float _exp2_67 = approx_exp2(_sub_68);
                float _add_67;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(1.0f), "f"(_exp2_67));
                float _rcp_67 = approx_rcp(_add_67);
                float sig_11_535 = _rcp_67;
                float _mul_261;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_261) : "f"(x1_00_518), "f"(sig_00_532));
                float act_00_536 = _mul_261;
                float _mul_262;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_262) : "f"(x1_01_519), "f"(sig_01_533));
                float act_01_537 = _mul_262;
                float _mul_263;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_263) : "f"(x1_10_522), "f"(sig_10_534));
                float act_10_538 = _mul_263;
                float _mul_264;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_264) : "f"(x1_11_523), "f"(sig_11_535));
                float act_11_539 = _mul_264;
                float _min_132;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_132) : "f"(act_00_536), "f"(cl));
                act_00_536 = _min_132;
                float _min_133;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_133) : "f"(act_01_537), "f"(cl));
                act_01_537 = _min_133;
                float _min_134;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_134) : "f"(act_10_538), "f"(cl));
                act_10_538 = _min_134;
                float _min_135;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_135) : "f"(act_11_539), "f"(cl));
                act_11_539 = _min_135;
                float _mul_265;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_265) : "f"(lin_00_528), "f"(act_00_536));
                float y_00_540 = _mul_265;
                float _mul_266;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_266) : "f"(lin_10_530), "f"(act_10_538));
                float y_10_541 = _mul_266;
                float _mul_267;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_267) : "f"(lin_01_529), "f"(act_01_537));
                float y_01_542 = _mul_267;
                float _mul_268;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_268) : "f"(lin_11_531), "f"(act_11_539));
                float y_11_543 = _mul_268;
                float _mul_269;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_269) : "f"(y_00_540), "f"(sc));
                quad[0] = _mul_269;
                float _mul_270;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_270) : "f"(y_10_541), "f"(sc));
                quad[1] = _mul_270;
                float _mul_271;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_271) : "f"(y_01_542), "f"(sc));
                quad[2] = _mul_271;
                float _mul_272;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_272) : "f"(y_11_543), "f"(sc));
                quad[3] = _mul_272;
                uint32_t _fp8_16[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_16[0] = _packed;
                }
                int off0_544 = local_token0_514 * 64 + base_row_513;
                int off1_545 = local_token1_515 * 64 + base_row_513;
                int swz0_546 = off0_544 ^ (off0_544 >> 7 & 3) << 4;
                int swz1_547 = off1_545 ^ (off1_545 >> 7 & 3) << 4;
                epi_staging[swz0_546 >> 1] = _fp8_16[0] & 65535;
                epi_staging[swz1_547 >> 1] = _fp8_16[0] >> 16;
                int local_token0_548 = lane_1 % 4 * 2 + 8;
                int local_token1_549 = local_token0_548 + 1;
                float x0_00_550 = _tmem_load_4[4];
                float x0_01_551 = _tmem_load_4[5];
                float x1_00_552 = _tmem_load_4[6];
                float x1_01_553 = _tmem_load_4[7];
                float x0_10_554 = _tmem_load_5[4];
                float x0_11_555 = _tmem_load_5[5];
                float x1_10_556 = _tmem_load_5[6];
                float x1_11_557 = _tmem_load_5[7];
                float _max_68;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_68) : "f"(x0_00_550), "f"(neg_cl));
                float _min_136;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_136) : "f"(_max_68), "f"(cl));
                float x0c_00_558 = _min_136;
                float _max_69;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_69) : "f"(x0_01_551), "f"(neg_cl));
                float _min_137;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_137) : "f"(_max_69), "f"(cl));
                float x0c_01_559 = _min_137;
                float _max_70;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_70) : "f"(x0_10_554), "f"(neg_cl));
                float _min_138;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_138) : "f"(_max_70), "f"(cl));
                float x0c_10_560 = _min_138;
                float _max_71;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_71) : "f"(x0_11_555), "f"(neg_cl));
                float _min_139;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_139) : "f"(_max_71), "f"(cl));
                float x0c_11_561 = _min_139;
                float _fma_68;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_68) : "f"(x0c_00_558), "f"(sg), "f"(0.0f));
                float lin_00_562 = _fma_68;
                float _fma_69;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_69) : "f"(x0c_01_559), "f"(sg), "f"(0.0f));
                float lin_01_563 = _fma_69;
                float _fma_70;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_70) : "f"(x0c_10_560), "f"(sg), "f"(0.0f));
                float lin_10_564 = _fma_70;
                float _fma_71;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_71) : "f"(x0c_11_561), "f"(sg), "f"(0.0f));
                float lin_11_565 = _fma_71;
                float _mul_273;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_273) : "f"(x1_00_552), "f"(fused));
                float _sub_69;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_69) : "f"(0.0f), "f"(_mul_273));
                float _exp2_68 = approx_exp2(_sub_69);
                float _add_68;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(1.0f), "f"(_exp2_68));
                float _rcp_68 = approx_rcp(_add_68);
                float sig_00_566 = _rcp_68;
                float _mul_274;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_274) : "f"(x1_01_553), "f"(fused));
                float _sub_70;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_70) : "f"(0.0f), "f"(_mul_274));
                float _exp2_69 = approx_exp2(_sub_70);
                float _add_69;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(1.0f), "f"(_exp2_69));
                float _rcp_69 = approx_rcp(_add_69);
                float sig_01_567 = _rcp_69;
                float _mul_275;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_275) : "f"(x1_10_556), "f"(fused));
                float _sub_71;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_71) : "f"(0.0f), "f"(_mul_275));
                float _exp2_70 = approx_exp2(_sub_71);
                float _add_70;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(1.0f), "f"(_exp2_70));
                float _rcp_70 = approx_rcp(_add_70);
                float sig_10_568 = _rcp_70;
                float _mul_276;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_276) : "f"(x1_11_557), "f"(fused));
                float _sub_72;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_72) : "f"(0.0f), "f"(_mul_276));
                float _exp2_71 = approx_exp2(_sub_72);
                float _add_71;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(1.0f), "f"(_exp2_71));
                float _rcp_71 = approx_rcp(_add_71);
                float sig_11_569 = _rcp_71;
                float _mul_277;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_277) : "f"(x1_00_552), "f"(sig_00_566));
                float act_00_570 = _mul_277;
                float _mul_278;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_278) : "f"(x1_01_553), "f"(sig_01_567));
                float act_01_571 = _mul_278;
                float _mul_279;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_279) : "f"(x1_10_556), "f"(sig_10_568));
                float act_10_572 = _mul_279;
                float _mul_280;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_280) : "f"(x1_11_557), "f"(sig_11_569));
                float act_11_573 = _mul_280;
                float _min_140;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_140) : "f"(act_00_570), "f"(cl));
                act_00_570 = _min_140;
                float _min_141;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_141) : "f"(act_01_571), "f"(cl));
                act_01_571 = _min_141;
                float _min_142;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_142) : "f"(act_10_572), "f"(cl));
                act_10_572 = _min_142;
                float _min_143;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_143) : "f"(act_11_573), "f"(cl));
                act_11_573 = _min_143;
                float _mul_281;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_281) : "f"(lin_00_562), "f"(act_00_570));
                float y_00_574 = _mul_281;
                float _mul_282;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_282) : "f"(lin_10_564), "f"(act_10_572));
                float y_10_575 = _mul_282;
                float _mul_283;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_283) : "f"(lin_01_563), "f"(act_01_571));
                float y_01_576 = _mul_283;
                float _mul_284;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_284) : "f"(lin_11_565), "f"(act_11_573));
                float y_11_577 = _mul_284;
                float _mul_285;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_285) : "f"(y_00_574), "f"(sc));
                quad[0] = _mul_285;
                float _mul_286;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_286) : "f"(y_10_575), "f"(sc));
                quad[1] = _mul_286;
                float _mul_287;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_287) : "f"(y_01_576), "f"(sc));
                quad[2] = _mul_287;
                float _mul_288;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_288) : "f"(y_11_577), "f"(sc));
                quad[3] = _mul_288;
                uint32_t _fp8_17[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_17[0] = _packed;
                }
                int off0_578 = local_token0_548 * 64 + base_row_513;
                int off1_579 = local_token1_549 * 64 + base_row_513;
                int swz0_580 = off0_578 ^ (off0_578 >> 7 & 3) << 4;
                int swz1_581 = off1_579 ^ (off1_579 >> 7 & 3) << 4;
                epi_staging[swz0_580 >> 1] = _fp8_17[0] & 65535;
                epi_staging[swz1_581 >> 1] = _fp8_17[0] >> 16;
                int local_token0_582 = lane_1 % 4 * 2 + 16;
                int local_token1_583 = local_token0_582 + 1;
                float x0_00_584 = _tmem_load_4[8];
                float x0_01_585 = _tmem_load_4[9];
                float x1_00_586 = _tmem_load_4[10];
                float x1_01_587 = _tmem_load_4[11];
                float x0_10_588 = _tmem_load_5[8];
                float x0_11_589 = _tmem_load_5[9];
                float x1_10_590 = _tmem_load_5[10];
                float x1_11_591 = _tmem_load_5[11];
                float _max_72;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_72) : "f"(x0_00_584), "f"(neg_cl));
                float _min_144;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_144) : "f"(_max_72), "f"(cl));
                float x0c_00_592 = _min_144;
                float _max_73;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_73) : "f"(x0_01_585), "f"(neg_cl));
                float _min_145;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_145) : "f"(_max_73), "f"(cl));
                float x0c_01_593 = _min_145;
                float _max_74;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_74) : "f"(x0_10_588), "f"(neg_cl));
                float _min_146;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_146) : "f"(_max_74), "f"(cl));
                float x0c_10_594 = _min_146;
                float _max_75;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_75) : "f"(x0_11_589), "f"(neg_cl));
                float _min_147;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_147) : "f"(_max_75), "f"(cl));
                float x0c_11_595 = _min_147;
                float _fma_72;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(x0c_00_592), "f"(sg), "f"(0.0f));
                float lin_00_596 = _fma_72;
                float _fma_73;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(x0c_01_593), "f"(sg), "f"(0.0f));
                float lin_01_597 = _fma_73;
                float _fma_74;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(x0c_10_594), "f"(sg), "f"(0.0f));
                float lin_10_598 = _fma_74;
                float _fma_75;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(x0c_11_595), "f"(sg), "f"(0.0f));
                float lin_11_599 = _fma_75;
                float _mul_289;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_289) : "f"(x1_00_586), "f"(fused));
                float _sub_73;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_73) : "f"(0.0f), "f"(_mul_289));
                float _exp2_72 = approx_exp2(_sub_73);
                float _add_72;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(1.0f), "f"(_exp2_72));
                float _rcp_72 = approx_rcp(_add_72);
                float sig_00_600 = _rcp_72;
                float _mul_290;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_290) : "f"(x1_01_587), "f"(fused));
                float _sub_74;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_74) : "f"(0.0f), "f"(_mul_290));
                float _exp2_73 = approx_exp2(_sub_74);
                float _add_73;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(1.0f), "f"(_exp2_73));
                float _rcp_73 = approx_rcp(_add_73);
                float sig_01_601 = _rcp_73;
                float _mul_291;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_291) : "f"(x1_10_590), "f"(fused));
                float _sub_75;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_75) : "f"(0.0f), "f"(_mul_291));
                float _exp2_74 = approx_exp2(_sub_75);
                float _add_74;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(1.0f), "f"(_exp2_74));
                float _rcp_74 = approx_rcp(_add_74);
                float sig_10_602 = _rcp_74;
                float _mul_292;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_292) : "f"(x1_11_591), "f"(fused));
                float _sub_76;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_76) : "f"(0.0f), "f"(_mul_292));
                float _exp2_75 = approx_exp2(_sub_76);
                float _add_75;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(1.0f), "f"(_exp2_75));
                float _rcp_75 = approx_rcp(_add_75);
                float sig_11_603 = _rcp_75;
                float _mul_293;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_293) : "f"(x1_00_586), "f"(sig_00_600));
                float act_00_604 = _mul_293;
                float _mul_294;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_294) : "f"(x1_01_587), "f"(sig_01_601));
                float act_01_605 = _mul_294;
                float _mul_295;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_295) : "f"(x1_10_590), "f"(sig_10_602));
                float act_10_606 = _mul_295;
                float _mul_296;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_296) : "f"(x1_11_591), "f"(sig_11_603));
                float act_11_607 = _mul_296;
                float _min_148;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_148) : "f"(act_00_604), "f"(cl));
                act_00_604 = _min_148;
                float _min_149;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_149) : "f"(act_01_605), "f"(cl));
                act_01_605 = _min_149;
                float _min_150;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_150) : "f"(act_10_606), "f"(cl));
                act_10_606 = _min_150;
                float _min_151;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_151) : "f"(act_11_607), "f"(cl));
                act_11_607 = _min_151;
                float _mul_297;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_297) : "f"(lin_00_596), "f"(act_00_604));
                float y_00_608 = _mul_297;
                float _mul_298;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_298) : "f"(lin_10_598), "f"(act_10_606));
                float y_10_609 = _mul_298;
                float _mul_299;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_299) : "f"(lin_01_597), "f"(act_01_605));
                float y_01_610 = _mul_299;
                float _mul_300;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_300) : "f"(lin_11_599), "f"(act_11_607));
                float y_11_611 = _mul_300;
                float _mul_301;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_301) : "f"(y_00_608), "f"(sc));
                quad[0] = _mul_301;
                float _mul_302;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_302) : "f"(y_10_609), "f"(sc));
                quad[1] = _mul_302;
                float _mul_303;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_303) : "f"(y_01_610), "f"(sc));
                quad[2] = _mul_303;
                float _mul_304;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_304) : "f"(y_11_611), "f"(sc));
                quad[3] = _mul_304;
                uint32_t _fp8_18[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_18[0] = _packed;
                }
                int off0_612 = local_token0_582 * 64 + base_row_513;
                int off1_613 = local_token1_583 * 64 + base_row_513;
                int swz0_614 = off0_612 ^ (off0_612 >> 7 & 3) << 4;
                int swz1_615 = off1_613 ^ (off1_613 >> 7 & 3) << 4;
                epi_staging[swz0_614 >> 1] = _fp8_18[0] & 65535;
                epi_staging[swz1_615 >> 1] = _fp8_18[0] >> 16;
                int local_token0_616 = lane_1 % 4 * 2 + 24;
                int local_token1_617 = local_token0_616 + 1;
                float x0_00_618 = _tmem_load_4[12];
                float x0_01_619 = _tmem_load_4[13];
                float x1_00_620 = _tmem_load_4[14];
                float x1_01_621 = _tmem_load_4[15];
                float x0_10_622 = _tmem_load_5[12];
                float x0_11_623 = _tmem_load_5[13];
                float x1_10_624 = _tmem_load_5[14];
                float x1_11_625 = _tmem_load_5[15];
                float _max_76;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_76) : "f"(x0_00_618), "f"(neg_cl));
                float _min_152;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_152) : "f"(_max_76), "f"(cl));
                float x0c_00_626 = _min_152;
                float _max_77;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_77) : "f"(x0_01_619), "f"(neg_cl));
                float _min_153;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_153) : "f"(_max_77), "f"(cl));
                float x0c_01_627 = _min_153;
                float _max_78;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_78) : "f"(x0_10_622), "f"(neg_cl));
                float _min_154;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_154) : "f"(_max_78), "f"(cl));
                float x0c_10_628 = _min_154;
                float _max_79;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_79) : "f"(x0_11_623), "f"(neg_cl));
                float _min_155;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_155) : "f"(_max_79), "f"(cl));
                float x0c_11_629 = _min_155;
                float _fma_76;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(x0c_00_626), "f"(sg), "f"(0.0f));
                float lin_00_630 = _fma_76;
                float _fma_77;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(x0c_01_627), "f"(sg), "f"(0.0f));
                float lin_01_631 = _fma_77;
                float _fma_78;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(x0c_10_628), "f"(sg), "f"(0.0f));
                float lin_10_632 = _fma_78;
                float _fma_79;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(x0c_11_629), "f"(sg), "f"(0.0f));
                float lin_11_633 = _fma_79;
                float _mul_305;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_305) : "f"(x1_00_620), "f"(fused));
                float _sub_77;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_77) : "f"(0.0f), "f"(_mul_305));
                float _exp2_76 = approx_exp2(_sub_77);
                float _add_76;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(1.0f), "f"(_exp2_76));
                float _rcp_76 = approx_rcp(_add_76);
                float sig_00_634 = _rcp_76;
                float _mul_306;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_306) : "f"(x1_01_621), "f"(fused));
                float _sub_78;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_78) : "f"(0.0f), "f"(_mul_306));
                float _exp2_77 = approx_exp2(_sub_78);
                float _add_77;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(1.0f), "f"(_exp2_77));
                float _rcp_77 = approx_rcp(_add_77);
                float sig_01_635 = _rcp_77;
                float _mul_307;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_307) : "f"(x1_10_624), "f"(fused));
                float _sub_79;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_79) : "f"(0.0f), "f"(_mul_307));
                float _exp2_78 = approx_exp2(_sub_79);
                float _add_78;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(1.0f), "f"(_exp2_78));
                float _rcp_78 = approx_rcp(_add_78);
                float sig_10_636 = _rcp_78;
                float _mul_308;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_308) : "f"(x1_11_625), "f"(fused));
                float _sub_80;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_80) : "f"(0.0f), "f"(_mul_308));
                float _exp2_79 = approx_exp2(_sub_80);
                float _add_79;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(1.0f), "f"(_exp2_79));
                float _rcp_79 = approx_rcp(_add_79);
                float sig_11_637 = _rcp_79;
                float _mul_309;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_309) : "f"(x1_00_620), "f"(sig_00_634));
                float act_00_638 = _mul_309;
                float _mul_310;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_310) : "f"(x1_01_621), "f"(sig_01_635));
                float act_01_639 = _mul_310;
                float _mul_311;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_311) : "f"(x1_10_624), "f"(sig_10_636));
                float act_10_640 = _mul_311;
                float _mul_312;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_312) : "f"(x1_11_625), "f"(sig_11_637));
                float act_11_641 = _mul_312;
                float _min_156;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_156) : "f"(act_00_638), "f"(cl));
                act_00_638 = _min_156;
                float _min_157;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_157) : "f"(act_01_639), "f"(cl));
                act_01_639 = _min_157;
                float _min_158;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_158) : "f"(act_10_640), "f"(cl));
                act_10_640 = _min_158;
                float _min_159;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_159) : "f"(act_11_641), "f"(cl));
                act_11_641 = _min_159;
                float _mul_313;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_313) : "f"(lin_00_630), "f"(act_00_638));
                float y_00_642 = _mul_313;
                float _mul_314;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_314) : "f"(lin_10_632), "f"(act_10_640));
                float y_10_643 = _mul_314;
                float _mul_315;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_315) : "f"(lin_01_631), "f"(act_01_639));
                float y_01_644 = _mul_315;
                float _mul_316;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_316) : "f"(lin_11_633), "f"(act_11_641));
                float y_11_645 = _mul_316;
                float _mul_317;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_317) : "f"(y_00_642), "f"(sc));
                quad[0] = _mul_317;
                float _mul_318;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_318) : "f"(y_10_643), "f"(sc));
                quad[1] = _mul_318;
                float _mul_319;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_319) : "f"(y_01_644), "f"(sc));
                quad[2] = _mul_319;
                float _mul_320;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_320) : "f"(y_11_645), "f"(sc));
                quad[3] = _mul_320;
                uint32_t _fp8_19[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_19[0] = _packed;
                }
                int off0_646 = local_token0_616 * 64 + base_row_513;
                int off1_647 = local_token1_617 * 64 + base_row_513;
                int swz0_648 = off0_646 ^ (off0_646 >> 7 & 3) << 4;
                int swz1_649 = off1_647 ^ (off1_647 >> 7 & 3) << 4;
                epi_staging[swz0_648 >> 1] = _fp8_19[0] & 65535;
                epi_staging[swz1_649 >> 1] = _fp8_19[0] >> 16;
                int local_token0_650 = lane_1 % 4 * 2 + 32;
                int local_token1_651 = local_token0_650 + 1;
                float x0_00_652 = _tmem_load_4[16];
                float x0_01_653 = _tmem_load_4[17];
                float x1_00_654 = _tmem_load_4[18];
                float x1_01_655 = _tmem_load_4[19];
                float x0_10_656 = _tmem_load_5[16];
                float x0_11_657 = _tmem_load_5[17];
                float x1_10_658 = _tmem_load_5[18];
                float x1_11_659 = _tmem_load_5[19];
                float _max_80;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_80) : "f"(x0_00_652), "f"(neg_cl));
                float _min_160;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_160) : "f"(_max_80), "f"(cl));
                float x0c_00_660 = _min_160;
                float _max_81;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_81) : "f"(x0_01_653), "f"(neg_cl));
                float _min_161;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_161) : "f"(_max_81), "f"(cl));
                float x0c_01_661 = _min_161;
                float _max_82;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_82) : "f"(x0_10_656), "f"(neg_cl));
                float _min_162;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_162) : "f"(_max_82), "f"(cl));
                float x0c_10_662 = _min_162;
                float _max_83;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_83) : "f"(x0_11_657), "f"(neg_cl));
                float _min_163;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_163) : "f"(_max_83), "f"(cl));
                float x0c_11_663 = _min_163;
                float _fma_80;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(x0c_00_660), "f"(sg), "f"(0.0f));
                float lin_00_664 = _fma_80;
                float _fma_81;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(x0c_01_661), "f"(sg), "f"(0.0f));
                float lin_01_665 = _fma_81;
                float _fma_82;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(x0c_10_662), "f"(sg), "f"(0.0f));
                float lin_10_666 = _fma_82;
                float _fma_83;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(x0c_11_663), "f"(sg), "f"(0.0f));
                float lin_11_667 = _fma_83;
                float _mul_321;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_321) : "f"(x1_00_654), "f"(fused));
                float _sub_81;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_81) : "f"(0.0f), "f"(_mul_321));
                float _exp2_80 = approx_exp2(_sub_81);
                float _add_80;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(1.0f), "f"(_exp2_80));
                float _rcp_80 = approx_rcp(_add_80);
                float sig_00_668 = _rcp_80;
                float _mul_322;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_322) : "f"(x1_01_655), "f"(fused));
                float _sub_82;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_82) : "f"(0.0f), "f"(_mul_322));
                float _exp2_81 = approx_exp2(_sub_82);
                float _add_81;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(1.0f), "f"(_exp2_81));
                float _rcp_81 = approx_rcp(_add_81);
                float sig_01_669 = _rcp_81;
                float _mul_323;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_323) : "f"(x1_10_658), "f"(fused));
                float _sub_83;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_83) : "f"(0.0f), "f"(_mul_323));
                float _exp2_82 = approx_exp2(_sub_83);
                float _add_82;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(1.0f), "f"(_exp2_82));
                float _rcp_82 = approx_rcp(_add_82);
                float sig_10_670 = _rcp_82;
                float _mul_324;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_324) : "f"(x1_11_659), "f"(fused));
                float _sub_84;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_84) : "f"(0.0f), "f"(_mul_324));
                float _exp2_83 = approx_exp2(_sub_84);
                float _add_83;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(1.0f), "f"(_exp2_83));
                float _rcp_83 = approx_rcp(_add_83);
                float sig_11_671 = _rcp_83;
                float _mul_325;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_325) : "f"(x1_00_654), "f"(sig_00_668));
                float act_00_672 = _mul_325;
                float _mul_326;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_326) : "f"(x1_01_655), "f"(sig_01_669));
                float act_01_673 = _mul_326;
                float _mul_327;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_327) : "f"(x1_10_658), "f"(sig_10_670));
                float act_10_674 = _mul_327;
                float _mul_328;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_328) : "f"(x1_11_659), "f"(sig_11_671));
                float act_11_675 = _mul_328;
                float _min_164;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_164) : "f"(act_00_672), "f"(cl));
                act_00_672 = _min_164;
                float _min_165;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_165) : "f"(act_01_673), "f"(cl));
                act_01_673 = _min_165;
                float _min_166;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_166) : "f"(act_10_674), "f"(cl));
                act_10_674 = _min_166;
                float _min_167;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_167) : "f"(act_11_675), "f"(cl));
                act_11_675 = _min_167;
                float _mul_329;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_329) : "f"(lin_00_664), "f"(act_00_672));
                float y_00_676 = _mul_329;
                float _mul_330;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_330) : "f"(lin_10_666), "f"(act_10_674));
                float y_10_677 = _mul_330;
                float _mul_331;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_331) : "f"(lin_01_665), "f"(act_01_673));
                float y_01_678 = _mul_331;
                float _mul_332;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_332) : "f"(lin_11_667), "f"(act_11_675));
                float y_11_679 = _mul_332;
                float _mul_333;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_333) : "f"(y_00_676), "f"(sc));
                quad[0] = _mul_333;
                float _mul_334;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_334) : "f"(y_10_677), "f"(sc));
                quad[1] = _mul_334;
                float _mul_335;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_335) : "f"(y_01_678), "f"(sc));
                quad[2] = _mul_335;
                float _mul_336;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_336) : "f"(y_11_679), "f"(sc));
                quad[3] = _mul_336;
                uint32_t _fp8_20[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_20[0] = _packed;
                }
                int off0_680 = local_token0_650 * 64 + base_row_513;
                int off1_681 = local_token1_651 * 64 + base_row_513;
                int swz0_682 = off0_680 ^ (off0_680 >> 7 & 3) << 4;
                int swz1_683 = off1_681 ^ (off1_681 >> 7 & 3) << 4;
                epi_staging[swz0_682 >> 1] = _fp8_20[0] & 65535;
                epi_staging[swz1_683 >> 1] = _fp8_20[0] >> 16;
                int local_token0_684 = lane_1 % 4 * 2 + 40;
                int local_token1_685 = local_token0_684 + 1;
                float x0_00_686 = _tmem_load_4[20];
                float x0_01_687 = _tmem_load_4[21];
                float x1_00_688 = _tmem_load_4[22];
                float x1_01_689 = _tmem_load_4[23];
                float x0_10_690 = _tmem_load_5[20];
                float x0_11_691 = _tmem_load_5[21];
                float x1_10_692 = _tmem_load_5[22];
                float x1_11_693 = _tmem_load_5[23];
                float _max_84;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_84) : "f"(x0_00_686), "f"(neg_cl));
                float _min_168;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_168) : "f"(_max_84), "f"(cl));
                float x0c_00_694 = _min_168;
                float _max_85;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_85) : "f"(x0_01_687), "f"(neg_cl));
                float _min_169;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_169) : "f"(_max_85), "f"(cl));
                float x0c_01_695 = _min_169;
                float _max_86;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_86) : "f"(x0_10_690), "f"(neg_cl));
                float _min_170;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_170) : "f"(_max_86), "f"(cl));
                float x0c_10_696 = _min_170;
                float _max_87;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_87) : "f"(x0_11_691), "f"(neg_cl));
                float _min_171;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_171) : "f"(_max_87), "f"(cl));
                float x0c_11_697 = _min_171;
                float _fma_84;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(x0c_00_694), "f"(sg), "f"(0.0f));
                float lin_00_698 = _fma_84;
                float _fma_85;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(x0c_01_695), "f"(sg), "f"(0.0f));
                float lin_01_699 = _fma_85;
                float _fma_86;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(x0c_10_696), "f"(sg), "f"(0.0f));
                float lin_10_700 = _fma_86;
                float _fma_87;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(x0c_11_697), "f"(sg), "f"(0.0f));
                float lin_11_701 = _fma_87;
                float _mul_337;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_337) : "f"(x1_00_688), "f"(fused));
                float _sub_85;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_85) : "f"(0.0f), "f"(_mul_337));
                float _exp2_84 = approx_exp2(_sub_85);
                float _add_84;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(1.0f), "f"(_exp2_84));
                float _rcp_84 = approx_rcp(_add_84);
                float sig_00_702 = _rcp_84;
                float _mul_338;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_338) : "f"(x1_01_689), "f"(fused));
                float _sub_86;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_86) : "f"(0.0f), "f"(_mul_338));
                float _exp2_85 = approx_exp2(_sub_86);
                float _add_85;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(1.0f), "f"(_exp2_85));
                float _rcp_85 = approx_rcp(_add_85);
                float sig_01_703 = _rcp_85;
                float _mul_339;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_339) : "f"(x1_10_692), "f"(fused));
                float _sub_87;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_87) : "f"(0.0f), "f"(_mul_339));
                float _exp2_86 = approx_exp2(_sub_87);
                float _add_86;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(1.0f), "f"(_exp2_86));
                float _rcp_86 = approx_rcp(_add_86);
                float sig_10_704 = _rcp_86;
                float _mul_340;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_340) : "f"(x1_11_693), "f"(fused));
                float _sub_88;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_88) : "f"(0.0f), "f"(_mul_340));
                float _exp2_87 = approx_exp2(_sub_88);
                float _add_87;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(1.0f), "f"(_exp2_87));
                float _rcp_87 = approx_rcp(_add_87);
                float sig_11_705 = _rcp_87;
                float _mul_341;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_341) : "f"(x1_00_688), "f"(sig_00_702));
                float act_00_706 = _mul_341;
                float _mul_342;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_342) : "f"(x1_01_689), "f"(sig_01_703));
                float act_01_707 = _mul_342;
                float _mul_343;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_343) : "f"(x1_10_692), "f"(sig_10_704));
                float act_10_708 = _mul_343;
                float _mul_344;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_344) : "f"(x1_11_693), "f"(sig_11_705));
                float act_11_709 = _mul_344;
                float _min_172;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_172) : "f"(act_00_706), "f"(cl));
                act_00_706 = _min_172;
                float _min_173;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_173) : "f"(act_01_707), "f"(cl));
                act_01_707 = _min_173;
                float _min_174;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_174) : "f"(act_10_708), "f"(cl));
                act_10_708 = _min_174;
                float _min_175;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_175) : "f"(act_11_709), "f"(cl));
                act_11_709 = _min_175;
                float _mul_345;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_345) : "f"(lin_00_698), "f"(act_00_706));
                float y_00_710 = _mul_345;
                float _mul_346;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_346) : "f"(lin_10_700), "f"(act_10_708));
                float y_10_711 = _mul_346;
                float _mul_347;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_347) : "f"(lin_01_699), "f"(act_01_707));
                float y_01_712 = _mul_347;
                float _mul_348;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_348) : "f"(lin_11_701), "f"(act_11_709));
                float y_11_713 = _mul_348;
                float _mul_349;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_349) : "f"(y_00_710), "f"(sc));
                quad[0] = _mul_349;
                float _mul_350;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_350) : "f"(y_10_711), "f"(sc));
                quad[1] = _mul_350;
                float _mul_351;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_351) : "f"(y_01_712), "f"(sc));
                quad[2] = _mul_351;
                float _mul_352;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_352) : "f"(y_11_713), "f"(sc));
                quad[3] = _mul_352;
                uint32_t _fp8_21[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_21[0] = _packed;
                }
                int off0_714 = local_token0_684 * 64 + base_row_513;
                int off1_715 = local_token1_685 * 64 + base_row_513;
                int swz0_716 = off0_714 ^ (off0_714 >> 7 & 3) << 4;
                int swz1_717 = off1_715 ^ (off1_715 >> 7 & 3) << 4;
                epi_staging[swz0_716 >> 1] = _fp8_21[0] & 65535;
                epi_staging[swz1_717 >> 1] = _fp8_21[0] >> 16;
                int local_token0_718 = lane_1 % 4 * 2 + 48;
                int local_token1_719 = local_token0_718 + 1;
                float x0_00_720 = _tmem_load_4[24];
                float x0_01_721 = _tmem_load_4[25];
                float x1_00_722 = _tmem_load_4[26];
                float x1_01_723 = _tmem_load_4[27];
                float x0_10_724 = _tmem_load_5[24];
                float x0_11_725 = _tmem_load_5[25];
                float x1_10_726 = _tmem_load_5[26];
                float x1_11_727 = _tmem_load_5[27];
                float _max_88;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_88) : "f"(x0_00_720), "f"(neg_cl));
                float _min_176;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_176) : "f"(_max_88), "f"(cl));
                float x0c_00_728 = _min_176;
                float _max_89;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_89) : "f"(x0_01_721), "f"(neg_cl));
                float _min_177;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_177) : "f"(_max_89), "f"(cl));
                float x0c_01_729 = _min_177;
                float _max_90;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_90) : "f"(x0_10_724), "f"(neg_cl));
                float _min_178;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_178) : "f"(_max_90), "f"(cl));
                float x0c_10_730 = _min_178;
                float _max_91;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_91) : "f"(x0_11_725), "f"(neg_cl));
                float _min_179;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_179) : "f"(_max_91), "f"(cl));
                float x0c_11_731 = _min_179;
                float _fma_88;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(x0c_00_728), "f"(sg), "f"(0.0f));
                float lin_00_732 = _fma_88;
                float _fma_89;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(x0c_01_729), "f"(sg), "f"(0.0f));
                float lin_01_733 = _fma_89;
                float _fma_90;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(x0c_10_730), "f"(sg), "f"(0.0f));
                float lin_10_734 = _fma_90;
                float _fma_91;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(x0c_11_731), "f"(sg), "f"(0.0f));
                float lin_11_735 = _fma_91;
                float _mul_353;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_353) : "f"(x1_00_722), "f"(fused));
                float _sub_89;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_89) : "f"(0.0f), "f"(_mul_353));
                float _exp2_88 = approx_exp2(_sub_89);
                float _add_88;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(1.0f), "f"(_exp2_88));
                float _rcp_88 = approx_rcp(_add_88);
                float sig_00_736 = _rcp_88;
                float _mul_354;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_354) : "f"(x1_01_723), "f"(fused));
                float _sub_90;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_90) : "f"(0.0f), "f"(_mul_354));
                float _exp2_89 = approx_exp2(_sub_90);
                float _add_89;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(1.0f), "f"(_exp2_89));
                float _rcp_89 = approx_rcp(_add_89);
                float sig_01_737 = _rcp_89;
                float _mul_355;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_355) : "f"(x1_10_726), "f"(fused));
                float _sub_91;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_91) : "f"(0.0f), "f"(_mul_355));
                float _exp2_90 = approx_exp2(_sub_91);
                float _add_90;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_90) : "f"(1.0f), "f"(_exp2_90));
                float _rcp_90 = approx_rcp(_add_90);
                float sig_10_738 = _rcp_90;
                float _mul_356;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_356) : "f"(x1_11_727), "f"(fused));
                float _sub_92;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_92) : "f"(0.0f), "f"(_mul_356));
                float _exp2_91 = approx_exp2(_sub_92);
                float _add_91;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(1.0f), "f"(_exp2_91));
                float _rcp_91 = approx_rcp(_add_91);
                float sig_11_739 = _rcp_91;
                float _mul_357;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_357) : "f"(x1_00_722), "f"(sig_00_736));
                float act_00_740 = _mul_357;
                float _mul_358;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_358) : "f"(x1_01_723), "f"(sig_01_737));
                float act_01_741 = _mul_358;
                float _mul_359;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_359) : "f"(x1_10_726), "f"(sig_10_738));
                float act_10_742 = _mul_359;
                float _mul_360;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_360) : "f"(x1_11_727), "f"(sig_11_739));
                float act_11_743 = _mul_360;
                float _min_180;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_180) : "f"(act_00_740), "f"(cl));
                act_00_740 = _min_180;
                float _min_181;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_181) : "f"(act_01_741), "f"(cl));
                act_01_741 = _min_181;
                float _min_182;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_182) : "f"(act_10_742), "f"(cl));
                act_10_742 = _min_182;
                float _min_183;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_183) : "f"(act_11_743), "f"(cl));
                act_11_743 = _min_183;
                float _mul_361;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_361) : "f"(lin_00_732), "f"(act_00_740));
                float y_00_744 = _mul_361;
                float _mul_362;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_362) : "f"(lin_10_734), "f"(act_10_742));
                float y_10_745 = _mul_362;
                float _mul_363;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_363) : "f"(lin_01_733), "f"(act_01_741));
                float y_01_746 = _mul_363;
                float _mul_364;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_364) : "f"(lin_11_735), "f"(act_11_743));
                float y_11_747 = _mul_364;
                float _mul_365;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_365) : "f"(y_00_744), "f"(sc));
                quad[0] = _mul_365;
                float _mul_366;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_366) : "f"(y_10_745), "f"(sc));
                quad[1] = _mul_366;
                float _mul_367;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_367) : "f"(y_01_746), "f"(sc));
                quad[2] = _mul_367;
                float _mul_368;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_368) : "f"(y_11_747), "f"(sc));
                quad[3] = _mul_368;
                uint32_t _fp8_22[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_22[0] = _packed;
                }
                int off0_748 = local_token0_718 * 64 + base_row_513;
                int off1_749 = local_token1_719 * 64 + base_row_513;
                int swz0_750 = off0_748 ^ (off0_748 >> 7 & 3) << 4;
                int swz1_751 = off1_749 ^ (off1_749 >> 7 & 3) << 4;
                epi_staging[swz0_750 >> 1] = _fp8_22[0] & 65535;
                epi_staging[swz1_751 >> 1] = _fp8_22[0] >> 16;
                int local_token0_752 = lane_1 % 4 * 2 + 56;
                int local_token1_753 = local_token0_752 + 1;
                float x0_00_754 = _tmem_load_4[28];
                float x0_01_755 = _tmem_load_4[29];
                float x1_00_756 = _tmem_load_4[30];
                float x1_01_757 = _tmem_load_4[31];
                float x0_10_758 = _tmem_load_5[28];
                float x0_11_759 = _tmem_load_5[29];
                float x1_10_760 = _tmem_load_5[30];
                float x1_11_761 = _tmem_load_5[31];
                float _max_92;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_92) : "f"(x0_00_754), "f"(neg_cl));
                float _min_184;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_184) : "f"(_max_92), "f"(cl));
                float x0c_00_762 = _min_184;
                float _max_93;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_93) : "f"(x0_01_755), "f"(neg_cl));
                float _min_185;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_185) : "f"(_max_93), "f"(cl));
                float x0c_01_763 = _min_185;
                float _max_94;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_94) : "f"(x0_10_758), "f"(neg_cl));
                float _min_186;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_186) : "f"(_max_94), "f"(cl));
                float x0c_10_764 = _min_186;
                float _max_95;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_95) : "f"(x0_11_759), "f"(neg_cl));
                float _min_187;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_187) : "f"(_max_95), "f"(cl));
                float x0c_11_765 = _min_187;
                float _fma_92;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(x0c_00_762), "f"(sg), "f"(0.0f));
                float lin_00_766 = _fma_92;
                float _fma_93;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(x0c_01_763), "f"(sg), "f"(0.0f));
                float lin_01_767 = _fma_93;
                float _fma_94;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(x0c_10_764), "f"(sg), "f"(0.0f));
                float lin_10_768 = _fma_94;
                float _fma_95;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(x0c_11_765), "f"(sg), "f"(0.0f));
                float lin_11_769 = _fma_95;
                float _mul_369;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_369) : "f"(x1_00_756), "f"(fused));
                float _sub_93;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_93) : "f"(0.0f), "f"(_mul_369));
                float _exp2_92 = approx_exp2(_sub_93);
                float _add_92;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(1.0f), "f"(_exp2_92));
                float _rcp_92 = approx_rcp(_add_92);
                float sig_00_770 = _rcp_92;
                float _mul_370;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_370) : "f"(x1_01_757), "f"(fused));
                float _sub_94;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_94) : "f"(0.0f), "f"(_mul_370));
                float _exp2_93 = approx_exp2(_sub_94);
                float _add_93;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(1.0f), "f"(_exp2_93));
                float _rcp_93 = approx_rcp(_add_93);
                float sig_01_771 = _rcp_93;
                float _mul_371;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_371) : "f"(x1_10_760), "f"(fused));
                float _sub_95;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_95) : "f"(0.0f), "f"(_mul_371));
                float _exp2_94 = approx_exp2(_sub_95);
                float _add_94;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(1.0f), "f"(_exp2_94));
                float _rcp_94 = approx_rcp(_add_94);
                float sig_10_772 = _rcp_94;
                float _mul_372;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_372) : "f"(x1_11_761), "f"(fused));
                float _sub_96;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_96) : "f"(0.0f), "f"(_mul_372));
                float _exp2_95 = approx_exp2(_sub_96);
                float _add_95;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(1.0f), "f"(_exp2_95));
                float _rcp_95 = approx_rcp(_add_95);
                float sig_11_773 = _rcp_95;
                float _mul_373;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_373) : "f"(x1_00_756), "f"(sig_00_770));
                float act_00_774 = _mul_373;
                float _mul_374;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_374) : "f"(x1_01_757), "f"(sig_01_771));
                float act_01_775 = _mul_374;
                float _mul_375;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_375) : "f"(x1_10_760), "f"(sig_10_772));
                float act_10_776 = _mul_375;
                float _mul_376;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_376) : "f"(x1_11_761), "f"(sig_11_773));
                float act_11_777 = _mul_376;
                float _min_188;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_188) : "f"(act_00_774), "f"(cl));
                act_00_774 = _min_188;
                float _min_189;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_189) : "f"(act_01_775), "f"(cl));
                act_01_775 = _min_189;
                float _min_190;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_190) : "f"(act_10_776), "f"(cl));
                act_10_776 = _min_190;
                float _min_191;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_191) : "f"(act_11_777), "f"(cl));
                act_11_777 = _min_191;
                float _mul_377;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_377) : "f"(lin_00_766), "f"(act_00_774));
                float y_00_778 = _mul_377;
                float _mul_378;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_378) : "f"(lin_10_768), "f"(act_10_776));
                float y_10_779 = _mul_378;
                float _mul_379;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_379) : "f"(lin_01_767), "f"(act_01_775));
                float y_01_780 = _mul_379;
                float _mul_380;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_380) : "f"(lin_11_769), "f"(act_11_777));
                float y_11_781 = _mul_380;
                float _mul_381;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_381) : "f"(y_00_778), "f"(sc));
                quad[0] = _mul_381;
                float _mul_382;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_382) : "f"(y_10_779), "f"(sc));
                quad[1] = _mul_382;
                float _mul_383;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_383) : "f"(y_01_780), "f"(sc));
                quad[2] = _mul_383;
                float _mul_384;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_384) : "f"(y_11_781), "f"(sc));
                quad[3] = _mul_384;
                uint32_t _fp8_23[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_23[0] = _packed;
                }
                int off0_782 = local_token0_752 * 64 + base_row_513;
                int off1_783 = local_token1_753 * 64 + base_row_513;
                int swz0_784 = off0_782 ^ (off0_782 >> 7 & 3) << 4;
                int swz1_785 = off1_783 ^ (off1_783 >> 7 & 3) << 4;
                epi_staging[swz0_784 >> 1] = _fp8_23[0] & 65535;
                epi_staging[swz1_785 >> 1] = _fp8_23[0] >> 16;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_2 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_2 + 128, 1073741824, n_tile * 256 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int acc_offset_786 = acc_stage * 256 + 192;
                float _tmem_load_6[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[31]))
                    : "r"(taddr + (unsigned int)acc_offset_786));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_7[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[31]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_786));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row_787 = warp_0 * 16 + lane_1 / 4 * 2;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int local_token0_788 = lane_1 % 4 * 2;
                int local_token1_789 = local_token0_788 + 1;
                float x0_00_790 = _tmem_load_6[0];
                float x0_01_791 = _tmem_load_6[1];
                float x1_00_792 = _tmem_load_6[2];
                float x1_01_793 = _tmem_load_6[3];
                float x0_10_794 = _tmem_load_7[0];
                float x0_11_795 = _tmem_load_7[1];
                float x1_10_796 = _tmem_load_7[2];
                float x1_11_797 = _tmem_load_7[3];
                float _max_96;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_96) : "f"(x0_00_790), "f"(neg_cl));
                float _min_192;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_192) : "f"(_max_96), "f"(cl));
                float x0c_00_798 = _min_192;
                float _max_97;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_97) : "f"(x0_01_791), "f"(neg_cl));
                float _min_193;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_193) : "f"(_max_97), "f"(cl));
                float x0c_01_799 = _min_193;
                float _max_98;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_98) : "f"(x0_10_794), "f"(neg_cl));
                float _min_194;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_194) : "f"(_max_98), "f"(cl));
                float x0c_10_800 = _min_194;
                float _max_99;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_99) : "f"(x0_11_795), "f"(neg_cl));
                float _min_195;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_195) : "f"(_max_99), "f"(cl));
                float x0c_11_801 = _min_195;
                float _fma_96;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(x0c_00_798), "f"(sg), "f"(0.0f));
                float lin_00_802 = _fma_96;
                float _fma_97;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(x0c_01_799), "f"(sg), "f"(0.0f));
                float lin_01_803 = _fma_97;
                float _fma_98;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(x0c_10_800), "f"(sg), "f"(0.0f));
                float lin_10_804 = _fma_98;
                float _fma_99;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(x0c_11_801), "f"(sg), "f"(0.0f));
                float lin_11_805 = _fma_99;
                float _mul_385;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_385) : "f"(x1_00_792), "f"(fused));
                float _sub_97;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_97) : "f"(0.0f), "f"(_mul_385));
                float _exp2_96 = approx_exp2(_sub_97);
                float _add_96;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(1.0f), "f"(_exp2_96));
                float _rcp_96 = approx_rcp(_add_96);
                float sig_00_806 = _rcp_96;
                float _mul_386;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_386) : "f"(x1_01_793), "f"(fused));
                float _sub_98;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_98) : "f"(0.0f), "f"(_mul_386));
                float _exp2_97 = approx_exp2(_sub_98);
                float _add_97;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(1.0f), "f"(_exp2_97));
                float _rcp_97 = approx_rcp(_add_97);
                float sig_01_807 = _rcp_97;
                float _mul_387;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_387) : "f"(x1_10_796), "f"(fused));
                float _sub_99;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_99) : "f"(0.0f), "f"(_mul_387));
                float _exp2_98 = approx_exp2(_sub_99);
                float _add_98;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(1.0f), "f"(_exp2_98));
                float _rcp_98 = approx_rcp(_add_98);
                float sig_10_808 = _rcp_98;
                float _mul_388;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_388) : "f"(x1_11_797), "f"(fused));
                float _sub_100;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_100) : "f"(0.0f), "f"(_mul_388));
                float _exp2_99 = approx_exp2(_sub_100);
                float _add_99;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(1.0f), "f"(_exp2_99));
                float _rcp_99 = approx_rcp(_add_99);
                float sig_11_809 = _rcp_99;
                float _mul_389;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_389) : "f"(x1_00_792), "f"(sig_00_806));
                float act_00_810 = _mul_389;
                float _mul_390;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_390) : "f"(x1_01_793), "f"(sig_01_807));
                float act_01_811 = _mul_390;
                float _mul_391;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_391) : "f"(x1_10_796), "f"(sig_10_808));
                float act_10_812 = _mul_391;
                float _mul_392;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_392) : "f"(x1_11_797), "f"(sig_11_809));
                float act_11_813 = _mul_392;
                float _min_196;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_196) : "f"(act_00_810), "f"(cl));
                act_00_810 = _min_196;
                float _min_197;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_197) : "f"(act_01_811), "f"(cl));
                act_01_811 = _min_197;
                float _min_198;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_198) : "f"(act_10_812), "f"(cl));
                act_10_812 = _min_198;
                float _min_199;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_199) : "f"(act_11_813), "f"(cl));
                act_11_813 = _min_199;
                float _mul_393;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_393) : "f"(lin_00_802), "f"(act_00_810));
                float y_00_814 = _mul_393;
                float _mul_394;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_394) : "f"(lin_10_804), "f"(act_10_812));
                float y_10_815 = _mul_394;
                float _mul_395;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_395) : "f"(lin_01_803), "f"(act_01_811));
                float y_01_816 = _mul_395;
                float _mul_396;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_396) : "f"(lin_11_805), "f"(act_11_813));
                float y_11_817 = _mul_396;
                float _mul_397;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_397) : "f"(y_00_814), "f"(sc));
                quad[0] = _mul_397;
                float _mul_398;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_398) : "f"(y_10_815), "f"(sc));
                quad[1] = _mul_398;
                float _mul_399;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_399) : "f"(y_01_816), "f"(sc));
                quad[2] = _mul_399;
                float _mul_400;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_400) : "f"(y_11_817), "f"(sc));
                quad[3] = _mul_400;
                uint32_t _fp8_24[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_24[0] = _packed;
                }
                int off0_818 = local_token0_788 * 64 + base_row_787;
                int off1_819 = local_token1_789 * 64 + base_row_787;
                int swz0_820 = off0_818 ^ (off0_818 >> 7 & 3) << 4;
                int swz1_821 = off1_819 ^ (off1_819 >> 7 & 3) << 4;
                epi_staging[swz0_820 >> 1] = _fp8_24[0] & 65535;
                epi_staging[swz1_821 >> 1] = _fp8_24[0] >> 16;
                int local_token0_822 = lane_1 % 4 * 2 + 8;
                int local_token1_823 = local_token0_822 + 1;
                float x0_00_824 = _tmem_load_6[4];
                float x0_01_825 = _tmem_load_6[5];
                float x1_00_826 = _tmem_load_6[6];
                float x1_01_827 = _tmem_load_6[7];
                float x0_10_828 = _tmem_load_7[4];
                float x0_11_829 = _tmem_load_7[5];
                float x1_10_830 = _tmem_load_7[6];
                float x1_11_831 = _tmem_load_7[7];
                float _max_100;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_100) : "f"(x0_00_824), "f"(neg_cl));
                float _min_200;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_200) : "f"(_max_100), "f"(cl));
                float x0c_00_832 = _min_200;
                float _max_101;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_101) : "f"(x0_01_825), "f"(neg_cl));
                float _min_201;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_201) : "f"(_max_101), "f"(cl));
                float x0c_01_833 = _min_201;
                float _max_102;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_102) : "f"(x0_10_828), "f"(neg_cl));
                float _min_202;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_202) : "f"(_max_102), "f"(cl));
                float x0c_10_834 = _min_202;
                float _max_103;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_103) : "f"(x0_11_829), "f"(neg_cl));
                float _min_203;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_203) : "f"(_max_103), "f"(cl));
                float x0c_11_835 = _min_203;
                float _fma_100;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_100) : "f"(x0c_00_832), "f"(sg), "f"(0.0f));
                float lin_00_836 = _fma_100;
                float _fma_101;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_101) : "f"(x0c_01_833), "f"(sg), "f"(0.0f));
                float lin_01_837 = _fma_101;
                float _fma_102;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_102) : "f"(x0c_10_834), "f"(sg), "f"(0.0f));
                float lin_10_838 = _fma_102;
                float _fma_103;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_103) : "f"(x0c_11_835), "f"(sg), "f"(0.0f));
                float lin_11_839 = _fma_103;
                float _mul_401;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_401) : "f"(x1_00_826), "f"(fused));
                float _sub_101;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_101) : "f"(0.0f), "f"(_mul_401));
                float _exp2_100 = approx_exp2(_sub_101);
                float _add_100;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(1.0f), "f"(_exp2_100));
                float _rcp_100 = approx_rcp(_add_100);
                float sig_00_840 = _rcp_100;
                float _mul_402;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_402) : "f"(x1_01_827), "f"(fused));
                float _sub_102;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_102) : "f"(0.0f), "f"(_mul_402));
                float _exp2_101 = approx_exp2(_sub_102);
                float _add_101;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(1.0f), "f"(_exp2_101));
                float _rcp_101 = approx_rcp(_add_101);
                float sig_01_841 = _rcp_101;
                float _mul_403;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_403) : "f"(x1_10_830), "f"(fused));
                float _sub_103;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_103) : "f"(0.0f), "f"(_mul_403));
                float _exp2_102 = approx_exp2(_sub_103);
                float _add_102;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(1.0f), "f"(_exp2_102));
                float _rcp_102 = approx_rcp(_add_102);
                float sig_10_842 = _rcp_102;
                float _mul_404;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_404) : "f"(x1_11_831), "f"(fused));
                float _sub_104;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_104) : "f"(0.0f), "f"(_mul_404));
                float _exp2_103 = approx_exp2(_sub_104);
                float _add_103;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(1.0f), "f"(_exp2_103));
                float _rcp_103 = approx_rcp(_add_103);
                float sig_11_843 = _rcp_103;
                float _mul_405;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_405) : "f"(x1_00_826), "f"(sig_00_840));
                float act_00_844 = _mul_405;
                float _mul_406;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_406) : "f"(x1_01_827), "f"(sig_01_841));
                float act_01_845 = _mul_406;
                float _mul_407;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_407) : "f"(x1_10_830), "f"(sig_10_842));
                float act_10_846 = _mul_407;
                float _mul_408;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_408) : "f"(x1_11_831), "f"(sig_11_843));
                float act_11_847 = _mul_408;
                float _min_204;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_204) : "f"(act_00_844), "f"(cl));
                act_00_844 = _min_204;
                float _min_205;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_205) : "f"(act_01_845), "f"(cl));
                act_01_845 = _min_205;
                float _min_206;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_206) : "f"(act_10_846), "f"(cl));
                act_10_846 = _min_206;
                float _min_207;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_207) : "f"(act_11_847), "f"(cl));
                act_11_847 = _min_207;
                float _mul_409;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_409) : "f"(lin_00_836), "f"(act_00_844));
                float y_00_848 = _mul_409;
                float _mul_410;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_410) : "f"(lin_10_838), "f"(act_10_846));
                float y_10_849 = _mul_410;
                float _mul_411;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_411) : "f"(lin_01_837), "f"(act_01_845));
                float y_01_850 = _mul_411;
                float _mul_412;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_412) : "f"(lin_11_839), "f"(act_11_847));
                float y_11_851 = _mul_412;
                float _mul_413;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_413) : "f"(y_00_848), "f"(sc));
                quad[0] = _mul_413;
                float _mul_414;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_414) : "f"(y_10_849), "f"(sc));
                quad[1] = _mul_414;
                float _mul_415;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_415) : "f"(y_01_850), "f"(sc));
                quad[2] = _mul_415;
                float _mul_416;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_416) : "f"(y_11_851), "f"(sc));
                quad[3] = _mul_416;
                uint32_t _fp8_25[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_25[0] = _packed;
                }
                int off0_852 = local_token0_822 * 64 + base_row_787;
                int off1_853 = local_token1_823 * 64 + base_row_787;
                int swz0_854 = off0_852 ^ (off0_852 >> 7 & 3) << 4;
                int swz1_855 = off1_853 ^ (off1_853 >> 7 & 3) << 4;
                epi_staging[swz0_854 >> 1] = _fp8_25[0] & 65535;
                epi_staging[swz1_855 >> 1] = _fp8_25[0] >> 16;
                int local_token0_856 = lane_1 % 4 * 2 + 16;
                int local_token1_857 = local_token0_856 + 1;
                float x0_00_858 = _tmem_load_6[8];
                float x0_01_859 = _tmem_load_6[9];
                float x1_00_860 = _tmem_load_6[10];
                float x1_01_861 = _tmem_load_6[11];
                float x0_10_862 = _tmem_load_7[8];
                float x0_11_863 = _tmem_load_7[9];
                float x1_10_864 = _tmem_load_7[10];
                float x1_11_865 = _tmem_load_7[11];
                float _max_104;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_104) : "f"(x0_00_858), "f"(neg_cl));
                float _min_208;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_208) : "f"(_max_104), "f"(cl));
                float x0c_00_866 = _min_208;
                float _max_105;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_105) : "f"(x0_01_859), "f"(neg_cl));
                float _min_209;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_209) : "f"(_max_105), "f"(cl));
                float x0c_01_867 = _min_209;
                float _max_106;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_106) : "f"(x0_10_862), "f"(neg_cl));
                float _min_210;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_210) : "f"(_max_106), "f"(cl));
                float x0c_10_868 = _min_210;
                float _max_107;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_107) : "f"(x0_11_863), "f"(neg_cl));
                float _min_211;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_211) : "f"(_max_107), "f"(cl));
                float x0c_11_869 = _min_211;
                float _fma_104;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_104) : "f"(x0c_00_866), "f"(sg), "f"(0.0f));
                float lin_00_870 = _fma_104;
                float _fma_105;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_105) : "f"(x0c_01_867), "f"(sg), "f"(0.0f));
                float lin_01_871 = _fma_105;
                float _fma_106;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_106) : "f"(x0c_10_868), "f"(sg), "f"(0.0f));
                float lin_10_872 = _fma_106;
                float _fma_107;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_107) : "f"(x0c_11_869), "f"(sg), "f"(0.0f));
                float lin_11_873 = _fma_107;
                float _mul_417;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_417) : "f"(x1_00_860), "f"(fused));
                float _sub_105;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_105) : "f"(0.0f), "f"(_mul_417));
                float _exp2_104 = approx_exp2(_sub_105);
                float _add_104;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(1.0f), "f"(_exp2_104));
                float _rcp_104 = approx_rcp(_add_104);
                float sig_00_874 = _rcp_104;
                float _mul_418;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_418) : "f"(x1_01_861), "f"(fused));
                float _sub_106;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_106) : "f"(0.0f), "f"(_mul_418));
                float _exp2_105 = approx_exp2(_sub_106);
                float _add_105;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_105) : "f"(1.0f), "f"(_exp2_105));
                float _rcp_105 = approx_rcp(_add_105);
                float sig_01_875 = _rcp_105;
                float _mul_419;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_419) : "f"(x1_10_864), "f"(fused));
                float _sub_107;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_107) : "f"(0.0f), "f"(_mul_419));
                float _exp2_106 = approx_exp2(_sub_107);
                float _add_106;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(1.0f), "f"(_exp2_106));
                float _rcp_106 = approx_rcp(_add_106);
                float sig_10_876 = _rcp_106;
                float _mul_420;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_420) : "f"(x1_11_865), "f"(fused));
                float _sub_108;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_108) : "f"(0.0f), "f"(_mul_420));
                float _exp2_107 = approx_exp2(_sub_108);
                float _add_107;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(1.0f), "f"(_exp2_107));
                float _rcp_107 = approx_rcp(_add_107);
                float sig_11_877 = _rcp_107;
                float _mul_421;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_421) : "f"(x1_00_860), "f"(sig_00_874));
                float act_00_878 = _mul_421;
                float _mul_422;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_422) : "f"(x1_01_861), "f"(sig_01_875));
                float act_01_879 = _mul_422;
                float _mul_423;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_423) : "f"(x1_10_864), "f"(sig_10_876));
                float act_10_880 = _mul_423;
                float _mul_424;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_424) : "f"(x1_11_865), "f"(sig_11_877));
                float act_11_881 = _mul_424;
                float _min_212;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_212) : "f"(act_00_878), "f"(cl));
                act_00_878 = _min_212;
                float _min_213;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_213) : "f"(act_01_879), "f"(cl));
                act_01_879 = _min_213;
                float _min_214;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_214) : "f"(act_10_880), "f"(cl));
                act_10_880 = _min_214;
                float _min_215;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_215) : "f"(act_11_881), "f"(cl));
                act_11_881 = _min_215;
                float _mul_425;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_425) : "f"(lin_00_870), "f"(act_00_878));
                float y_00_882 = _mul_425;
                float _mul_426;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_426) : "f"(lin_10_872), "f"(act_10_880));
                float y_10_883 = _mul_426;
                float _mul_427;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_427) : "f"(lin_01_871), "f"(act_01_879));
                float y_01_884 = _mul_427;
                float _mul_428;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_428) : "f"(lin_11_873), "f"(act_11_881));
                float y_11_885 = _mul_428;
                float _mul_429;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_429) : "f"(y_00_882), "f"(sc));
                quad[0] = _mul_429;
                float _mul_430;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_430) : "f"(y_10_883), "f"(sc));
                quad[1] = _mul_430;
                float _mul_431;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_431) : "f"(y_01_884), "f"(sc));
                quad[2] = _mul_431;
                float _mul_432;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_432) : "f"(y_11_885), "f"(sc));
                quad[3] = _mul_432;
                uint32_t _fp8_26[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_26[0] = _packed;
                }
                int off0_886 = local_token0_856 * 64 + base_row_787;
                int off1_887 = local_token1_857 * 64 + base_row_787;
                int swz0_888 = off0_886 ^ (off0_886 >> 7 & 3) << 4;
                int swz1_889 = off1_887 ^ (off1_887 >> 7 & 3) << 4;
                epi_staging[swz0_888 >> 1] = _fp8_26[0] & 65535;
                epi_staging[swz1_889 >> 1] = _fp8_26[0] >> 16;
                int local_token0_890 = lane_1 % 4 * 2 + 24;
                int local_token1_891 = local_token0_890 + 1;
                float x0_00_892 = _tmem_load_6[12];
                float x0_01_893 = _tmem_load_6[13];
                float x1_00_894 = _tmem_load_6[14];
                float x1_01_895 = _tmem_load_6[15];
                float x0_10_896 = _tmem_load_7[12];
                float x0_11_897 = _tmem_load_7[13];
                float x1_10_898 = _tmem_load_7[14];
                float x1_11_899 = _tmem_load_7[15];
                float _max_108;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_108) : "f"(x0_00_892), "f"(neg_cl));
                float _min_216;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_216) : "f"(_max_108), "f"(cl));
                float x0c_00_900 = _min_216;
                float _max_109;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_109) : "f"(x0_01_893), "f"(neg_cl));
                float _min_217;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_217) : "f"(_max_109), "f"(cl));
                float x0c_01_901 = _min_217;
                float _max_110;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_110) : "f"(x0_10_896), "f"(neg_cl));
                float _min_218;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_218) : "f"(_max_110), "f"(cl));
                float x0c_10_902 = _min_218;
                float _max_111;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_111) : "f"(x0_11_897), "f"(neg_cl));
                float _min_219;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_219) : "f"(_max_111), "f"(cl));
                float x0c_11_903 = _min_219;
                float _fma_108;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_108) : "f"(x0c_00_900), "f"(sg), "f"(0.0f));
                float lin_00_904 = _fma_108;
                float _fma_109;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_109) : "f"(x0c_01_901), "f"(sg), "f"(0.0f));
                float lin_01_905 = _fma_109;
                float _fma_110;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_110) : "f"(x0c_10_902), "f"(sg), "f"(0.0f));
                float lin_10_906 = _fma_110;
                float _fma_111;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_111) : "f"(x0c_11_903), "f"(sg), "f"(0.0f));
                float lin_11_907 = _fma_111;
                float _mul_433;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_433) : "f"(x1_00_894), "f"(fused));
                float _sub_109;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_109) : "f"(0.0f), "f"(_mul_433));
                float _exp2_108 = approx_exp2(_sub_109);
                float _add_108;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(1.0f), "f"(_exp2_108));
                float _rcp_108 = approx_rcp(_add_108);
                float sig_00_908 = _rcp_108;
                float _mul_434;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_434) : "f"(x1_01_895), "f"(fused));
                float _sub_110;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_110) : "f"(0.0f), "f"(_mul_434));
                float _exp2_109 = approx_exp2(_sub_110);
                float _add_109;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(1.0f), "f"(_exp2_109));
                float _rcp_109 = approx_rcp(_add_109);
                float sig_01_909 = _rcp_109;
                float _mul_435;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_435) : "f"(x1_10_898), "f"(fused));
                float _sub_111;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_111) : "f"(0.0f), "f"(_mul_435));
                float _exp2_110 = approx_exp2(_sub_111);
                float _add_110;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(1.0f), "f"(_exp2_110));
                float _rcp_110 = approx_rcp(_add_110);
                float sig_10_910 = _rcp_110;
                float _mul_436;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_436) : "f"(x1_11_899), "f"(fused));
                float _sub_112;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_112) : "f"(0.0f), "f"(_mul_436));
                float _exp2_111 = approx_exp2(_sub_112);
                float _add_111;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(1.0f), "f"(_exp2_111));
                float _rcp_111 = approx_rcp(_add_111);
                float sig_11_911 = _rcp_111;
                float _mul_437;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_437) : "f"(x1_00_894), "f"(sig_00_908));
                float act_00_912 = _mul_437;
                float _mul_438;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_438) : "f"(x1_01_895), "f"(sig_01_909));
                float act_01_913 = _mul_438;
                float _mul_439;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_439) : "f"(x1_10_898), "f"(sig_10_910));
                float act_10_914 = _mul_439;
                float _mul_440;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_440) : "f"(x1_11_899), "f"(sig_11_911));
                float act_11_915 = _mul_440;
                float _min_220;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_220) : "f"(act_00_912), "f"(cl));
                act_00_912 = _min_220;
                float _min_221;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_221) : "f"(act_01_913), "f"(cl));
                act_01_913 = _min_221;
                float _min_222;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_222) : "f"(act_10_914), "f"(cl));
                act_10_914 = _min_222;
                float _min_223;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_223) : "f"(act_11_915), "f"(cl));
                act_11_915 = _min_223;
                float _mul_441;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_441) : "f"(lin_00_904), "f"(act_00_912));
                float y_00_916 = _mul_441;
                float _mul_442;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_442) : "f"(lin_10_906), "f"(act_10_914));
                float y_10_917 = _mul_442;
                float _mul_443;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_443) : "f"(lin_01_905), "f"(act_01_913));
                float y_01_918 = _mul_443;
                float _mul_444;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_444) : "f"(lin_11_907), "f"(act_11_915));
                float y_11_919 = _mul_444;
                float _mul_445;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_445) : "f"(y_00_916), "f"(sc));
                quad[0] = _mul_445;
                float _mul_446;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_446) : "f"(y_10_917), "f"(sc));
                quad[1] = _mul_446;
                float _mul_447;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_447) : "f"(y_01_918), "f"(sc));
                quad[2] = _mul_447;
                float _mul_448;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_448) : "f"(y_11_919), "f"(sc));
                quad[3] = _mul_448;
                uint32_t _fp8_27[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_27[0] = _packed;
                }
                int off0_920 = local_token0_890 * 64 + base_row_787;
                int off1_921 = local_token1_891 * 64 + base_row_787;
                int swz0_922 = off0_920 ^ (off0_920 >> 7 & 3) << 4;
                int swz1_923 = off1_921 ^ (off1_921 >> 7 & 3) << 4;
                epi_staging[swz0_922 >> 1] = _fp8_27[0] & 65535;
                epi_staging[swz1_923 >> 1] = _fp8_27[0] >> 16;
                int local_token0_924 = lane_1 % 4 * 2 + 32;
                int local_token1_925 = local_token0_924 + 1;
                float x0_00_926 = _tmem_load_6[16];
                float x0_01_927 = _tmem_load_6[17];
                float x1_00_928 = _tmem_load_6[18];
                float x1_01_929 = _tmem_load_6[19];
                float x0_10_930 = _tmem_load_7[16];
                float x0_11_931 = _tmem_load_7[17];
                float x1_10_932 = _tmem_load_7[18];
                float x1_11_933 = _tmem_load_7[19];
                float _max_112;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_112) : "f"(x0_00_926), "f"(neg_cl));
                float _min_224;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_224) : "f"(_max_112), "f"(cl));
                float x0c_00_934 = _min_224;
                float _max_113;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_113) : "f"(x0_01_927), "f"(neg_cl));
                float _min_225;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_225) : "f"(_max_113), "f"(cl));
                float x0c_01_935 = _min_225;
                float _max_114;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_114) : "f"(x0_10_930), "f"(neg_cl));
                float _min_226;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_226) : "f"(_max_114), "f"(cl));
                float x0c_10_936 = _min_226;
                float _max_115;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_115) : "f"(x0_11_931), "f"(neg_cl));
                float _min_227;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_227) : "f"(_max_115), "f"(cl));
                float x0c_11_937 = _min_227;
                float _fma_112;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_112) : "f"(x0c_00_934), "f"(sg), "f"(0.0f));
                float lin_00_938 = _fma_112;
                float _fma_113;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_113) : "f"(x0c_01_935), "f"(sg), "f"(0.0f));
                float lin_01_939 = _fma_113;
                float _fma_114;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_114) : "f"(x0c_10_936), "f"(sg), "f"(0.0f));
                float lin_10_940 = _fma_114;
                float _fma_115;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_115) : "f"(x0c_11_937), "f"(sg), "f"(0.0f));
                float lin_11_941 = _fma_115;
                float _mul_449;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_449) : "f"(x1_00_928), "f"(fused));
                float _sub_113;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_113) : "f"(0.0f), "f"(_mul_449));
                float _exp2_112 = approx_exp2(_sub_113);
                float _add_112;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(1.0f), "f"(_exp2_112));
                float _rcp_112 = approx_rcp(_add_112);
                float sig_00_942 = _rcp_112;
                float _mul_450;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_450) : "f"(x1_01_929), "f"(fused));
                float _sub_114;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_114) : "f"(0.0f), "f"(_mul_450));
                float _exp2_113 = approx_exp2(_sub_114);
                float _add_113;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(1.0f), "f"(_exp2_113));
                float _rcp_113 = approx_rcp(_add_113);
                float sig_01_943 = _rcp_113;
                float _mul_451;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_451) : "f"(x1_10_932), "f"(fused));
                float _sub_115;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_115) : "f"(0.0f), "f"(_mul_451));
                float _exp2_114 = approx_exp2(_sub_115);
                float _add_114;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(1.0f), "f"(_exp2_114));
                float _rcp_114 = approx_rcp(_add_114);
                float sig_10_944 = _rcp_114;
                float _mul_452;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_452) : "f"(x1_11_933), "f"(fused));
                float _sub_116;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_116) : "f"(0.0f), "f"(_mul_452));
                float _exp2_115 = approx_exp2(_sub_116);
                float _add_115;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(1.0f), "f"(_exp2_115));
                float _rcp_115 = approx_rcp(_add_115);
                float sig_11_945 = _rcp_115;
                float _mul_453;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_453) : "f"(x1_00_928), "f"(sig_00_942));
                float act_00_946 = _mul_453;
                float _mul_454;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_454) : "f"(x1_01_929), "f"(sig_01_943));
                float act_01_947 = _mul_454;
                float _mul_455;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_455) : "f"(x1_10_932), "f"(sig_10_944));
                float act_10_948 = _mul_455;
                float _mul_456;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_456) : "f"(x1_11_933), "f"(sig_11_945));
                float act_11_949 = _mul_456;
                float _min_228;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_228) : "f"(act_00_946), "f"(cl));
                act_00_946 = _min_228;
                float _min_229;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_229) : "f"(act_01_947), "f"(cl));
                act_01_947 = _min_229;
                float _min_230;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_230) : "f"(act_10_948), "f"(cl));
                act_10_948 = _min_230;
                float _min_231;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_231) : "f"(act_11_949), "f"(cl));
                act_11_949 = _min_231;
                float _mul_457;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_457) : "f"(lin_00_938), "f"(act_00_946));
                float y_00_950 = _mul_457;
                float _mul_458;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_458) : "f"(lin_10_940), "f"(act_10_948));
                float y_10_951 = _mul_458;
                float _mul_459;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_459) : "f"(lin_01_939), "f"(act_01_947));
                float y_01_952 = _mul_459;
                float _mul_460;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_460) : "f"(lin_11_941), "f"(act_11_949));
                float y_11_953 = _mul_460;
                float _mul_461;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_461) : "f"(y_00_950), "f"(sc));
                quad[0] = _mul_461;
                float _mul_462;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_462) : "f"(y_10_951), "f"(sc));
                quad[1] = _mul_462;
                float _mul_463;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_463) : "f"(y_01_952), "f"(sc));
                quad[2] = _mul_463;
                float _mul_464;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_464) : "f"(y_11_953), "f"(sc));
                quad[3] = _mul_464;
                uint32_t _fp8_28[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_28[0] = _packed;
                }
                int off0_954 = local_token0_924 * 64 + base_row_787;
                int off1_955 = local_token1_925 * 64 + base_row_787;
                int swz0_956 = off0_954 ^ (off0_954 >> 7 & 3) << 4;
                int swz1_957 = off1_955 ^ (off1_955 >> 7 & 3) << 4;
                epi_staging[swz0_956 >> 1] = _fp8_28[0] & 65535;
                epi_staging[swz1_957 >> 1] = _fp8_28[0] >> 16;
                int local_token0_958 = lane_1 % 4 * 2 + 40;
                int local_token1_959 = local_token0_958 + 1;
                float x0_00_960 = _tmem_load_6[20];
                float x0_01_961 = _tmem_load_6[21];
                float x1_00_962 = _tmem_load_6[22];
                float x1_01_963 = _tmem_load_6[23];
                float x0_10_964 = _tmem_load_7[20];
                float x0_11_965 = _tmem_load_7[21];
                float x1_10_966 = _tmem_load_7[22];
                float x1_11_967 = _tmem_load_7[23];
                float _max_116;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_116) : "f"(x0_00_960), "f"(neg_cl));
                float _min_232;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_232) : "f"(_max_116), "f"(cl));
                float x0c_00_968 = _min_232;
                float _max_117;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_117) : "f"(x0_01_961), "f"(neg_cl));
                float _min_233;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_233) : "f"(_max_117), "f"(cl));
                float x0c_01_969 = _min_233;
                float _max_118;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_118) : "f"(x0_10_964), "f"(neg_cl));
                float _min_234;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_234) : "f"(_max_118), "f"(cl));
                float x0c_10_970 = _min_234;
                float _max_119;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_119) : "f"(x0_11_965), "f"(neg_cl));
                float _min_235;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_235) : "f"(_max_119), "f"(cl));
                float x0c_11_971 = _min_235;
                float _fma_116;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_116) : "f"(x0c_00_968), "f"(sg), "f"(0.0f));
                float lin_00_972 = _fma_116;
                float _fma_117;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_117) : "f"(x0c_01_969), "f"(sg), "f"(0.0f));
                float lin_01_973 = _fma_117;
                float _fma_118;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_118) : "f"(x0c_10_970), "f"(sg), "f"(0.0f));
                float lin_10_974 = _fma_118;
                float _fma_119;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_119) : "f"(x0c_11_971), "f"(sg), "f"(0.0f));
                float lin_11_975 = _fma_119;
                float _mul_465;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_465) : "f"(x1_00_962), "f"(fused));
                float _sub_117;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_117) : "f"(0.0f), "f"(_mul_465));
                float _exp2_116 = approx_exp2(_sub_117);
                float _add_116;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(1.0f), "f"(_exp2_116));
                float _rcp_116 = approx_rcp(_add_116);
                float sig_00_976 = _rcp_116;
                float _mul_466;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_466) : "f"(x1_01_963), "f"(fused));
                float _sub_118;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_118) : "f"(0.0f), "f"(_mul_466));
                float _exp2_117 = approx_exp2(_sub_118);
                float _add_117;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(1.0f), "f"(_exp2_117));
                float _rcp_117 = approx_rcp(_add_117);
                float sig_01_977 = _rcp_117;
                float _mul_467;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_467) : "f"(x1_10_966), "f"(fused));
                float _sub_119;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_119) : "f"(0.0f), "f"(_mul_467));
                float _exp2_118 = approx_exp2(_sub_119);
                float _add_118;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(1.0f), "f"(_exp2_118));
                float _rcp_118 = approx_rcp(_add_118);
                float sig_10_978 = _rcp_118;
                float _mul_468;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_468) : "f"(x1_11_967), "f"(fused));
                float _sub_120;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_120) : "f"(0.0f), "f"(_mul_468));
                float _exp2_119 = approx_exp2(_sub_120);
                float _add_119;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(1.0f), "f"(_exp2_119));
                float _rcp_119 = approx_rcp(_add_119);
                float sig_11_979 = _rcp_119;
                float _mul_469;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_469) : "f"(x1_00_962), "f"(sig_00_976));
                float act_00_980 = _mul_469;
                float _mul_470;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_470) : "f"(x1_01_963), "f"(sig_01_977));
                float act_01_981 = _mul_470;
                float _mul_471;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_471) : "f"(x1_10_966), "f"(sig_10_978));
                float act_10_982 = _mul_471;
                float _mul_472;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_472) : "f"(x1_11_967), "f"(sig_11_979));
                float act_11_983 = _mul_472;
                float _min_236;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_236) : "f"(act_00_980), "f"(cl));
                act_00_980 = _min_236;
                float _min_237;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_237) : "f"(act_01_981), "f"(cl));
                act_01_981 = _min_237;
                float _min_238;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_238) : "f"(act_10_982), "f"(cl));
                act_10_982 = _min_238;
                float _min_239;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_239) : "f"(act_11_983), "f"(cl));
                act_11_983 = _min_239;
                float _mul_473;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_473) : "f"(lin_00_972), "f"(act_00_980));
                float y_00_984 = _mul_473;
                float _mul_474;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_474) : "f"(lin_10_974), "f"(act_10_982));
                float y_10_985 = _mul_474;
                float _mul_475;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_475) : "f"(lin_01_973), "f"(act_01_981));
                float y_01_986 = _mul_475;
                float _mul_476;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_476) : "f"(lin_11_975), "f"(act_11_983));
                float y_11_987 = _mul_476;
                float _mul_477;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_477) : "f"(y_00_984), "f"(sc));
                quad[0] = _mul_477;
                float _mul_478;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_478) : "f"(y_10_985), "f"(sc));
                quad[1] = _mul_478;
                float _mul_479;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_479) : "f"(y_01_986), "f"(sc));
                quad[2] = _mul_479;
                float _mul_480;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_480) : "f"(y_11_987), "f"(sc));
                quad[3] = _mul_480;
                uint32_t _fp8_29[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_29[0] = _packed;
                }
                int off0_988 = local_token0_958 * 64 + base_row_787;
                int off1_989 = local_token1_959 * 64 + base_row_787;
                int swz0_990 = off0_988 ^ (off0_988 >> 7 & 3) << 4;
                int swz1_991 = off1_989 ^ (off1_989 >> 7 & 3) << 4;
                epi_staging[swz0_990 >> 1] = _fp8_29[0] & 65535;
                epi_staging[swz1_991 >> 1] = _fp8_29[0] >> 16;
                int local_token0_992 = lane_1 % 4 * 2 + 48;
                int local_token1_993 = local_token0_992 + 1;
                float x0_00_994 = _tmem_load_6[24];
                float x0_01_995 = _tmem_load_6[25];
                float x1_00_996 = _tmem_load_6[26];
                float x1_01_997 = _tmem_load_6[27];
                float x0_10_998 = _tmem_load_7[24];
                float x0_11_999 = _tmem_load_7[25];
                float x1_10_1000 = _tmem_load_7[26];
                float x1_11_1001 = _tmem_load_7[27];
                float _max_120;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_120) : "f"(x0_00_994), "f"(neg_cl));
                float _min_240;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_240) : "f"(_max_120), "f"(cl));
                float x0c_00_1002 = _min_240;
                float _max_121;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_121) : "f"(x0_01_995), "f"(neg_cl));
                float _min_241;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_241) : "f"(_max_121), "f"(cl));
                float x0c_01_1003 = _min_241;
                float _max_122;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_122) : "f"(x0_10_998), "f"(neg_cl));
                float _min_242;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_242) : "f"(_max_122), "f"(cl));
                float x0c_10_1004 = _min_242;
                float _max_123;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_123) : "f"(x0_11_999), "f"(neg_cl));
                float _min_243;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_243) : "f"(_max_123), "f"(cl));
                float x0c_11_1005 = _min_243;
                float _fma_120;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_120) : "f"(x0c_00_1002), "f"(sg), "f"(0.0f));
                float lin_00_1006 = _fma_120;
                float _fma_121;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_121) : "f"(x0c_01_1003), "f"(sg), "f"(0.0f));
                float lin_01_1007 = _fma_121;
                float _fma_122;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_122) : "f"(x0c_10_1004), "f"(sg), "f"(0.0f));
                float lin_10_1008 = _fma_122;
                float _fma_123;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_123) : "f"(x0c_11_1005), "f"(sg), "f"(0.0f));
                float lin_11_1009 = _fma_123;
                float _mul_481;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_481) : "f"(x1_00_996), "f"(fused));
                float _sub_121;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_121) : "f"(0.0f), "f"(_mul_481));
                float _exp2_120 = approx_exp2(_sub_121);
                float _add_120;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(1.0f), "f"(_exp2_120));
                float _rcp_120 = approx_rcp(_add_120);
                float sig_00_1010 = _rcp_120;
                float _mul_482;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_482) : "f"(x1_01_997), "f"(fused));
                float _sub_122;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_122) : "f"(0.0f), "f"(_mul_482));
                float _exp2_121 = approx_exp2(_sub_122);
                float _add_121;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(1.0f), "f"(_exp2_121));
                float _rcp_121 = approx_rcp(_add_121);
                float sig_01_1011 = _rcp_121;
                float _mul_483;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_483) : "f"(x1_10_1000), "f"(fused));
                float _sub_123;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_123) : "f"(0.0f), "f"(_mul_483));
                float _exp2_122 = approx_exp2(_sub_123);
                float _add_122;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(1.0f), "f"(_exp2_122));
                float _rcp_122 = approx_rcp(_add_122);
                float sig_10_1012 = _rcp_122;
                float _mul_484;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_484) : "f"(x1_11_1001), "f"(fused));
                float _sub_124;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_124) : "f"(0.0f), "f"(_mul_484));
                float _exp2_123 = approx_exp2(_sub_124);
                float _add_123;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(1.0f), "f"(_exp2_123));
                float _rcp_123 = approx_rcp(_add_123);
                float sig_11_1013 = _rcp_123;
                float _mul_485;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_485) : "f"(x1_00_996), "f"(sig_00_1010));
                float act_00_1014 = _mul_485;
                float _mul_486;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_486) : "f"(x1_01_997), "f"(sig_01_1011));
                float act_01_1015 = _mul_486;
                float _mul_487;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_487) : "f"(x1_10_1000), "f"(sig_10_1012));
                float act_10_1016 = _mul_487;
                float _mul_488;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_488) : "f"(x1_11_1001), "f"(sig_11_1013));
                float act_11_1017 = _mul_488;
                float _min_244;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_244) : "f"(act_00_1014), "f"(cl));
                act_00_1014 = _min_244;
                float _min_245;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_245) : "f"(act_01_1015), "f"(cl));
                act_01_1015 = _min_245;
                float _min_246;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_246) : "f"(act_10_1016), "f"(cl));
                act_10_1016 = _min_246;
                float _min_247;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_247) : "f"(act_11_1017), "f"(cl));
                act_11_1017 = _min_247;
                float _mul_489;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_489) : "f"(lin_00_1006), "f"(act_00_1014));
                float y_00_1018 = _mul_489;
                float _mul_490;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_490) : "f"(lin_10_1008), "f"(act_10_1016));
                float y_10_1019 = _mul_490;
                float _mul_491;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_491) : "f"(lin_01_1007), "f"(act_01_1015));
                float y_01_1020 = _mul_491;
                float _mul_492;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_492) : "f"(lin_11_1009), "f"(act_11_1017));
                float y_11_1021 = _mul_492;
                float _mul_493;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_493) : "f"(y_00_1018), "f"(sc));
                quad[0] = _mul_493;
                float _mul_494;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_494) : "f"(y_10_1019), "f"(sc));
                quad[1] = _mul_494;
                float _mul_495;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_495) : "f"(y_01_1020), "f"(sc));
                quad[2] = _mul_495;
                float _mul_496;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_496) : "f"(y_11_1021), "f"(sc));
                quad[3] = _mul_496;
                uint32_t _fp8_30[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_30[0] = _packed;
                }
                int off0_1022 = local_token0_992 * 64 + base_row_787;
                int off1_1023 = local_token1_993 * 64 + base_row_787;
                int swz0_1024 = off0_1022 ^ (off0_1022 >> 7 & 3) << 4;
                int swz1_1025 = off1_1023 ^ (off1_1023 >> 7 & 3) << 4;
                epi_staging[swz0_1024 >> 1] = _fp8_30[0] & 65535;
                epi_staging[swz1_1025 >> 1] = _fp8_30[0] >> 16;
                int local_token0_1026 = lane_1 % 4 * 2 + 56;
                int local_token1_1027 = local_token0_1026 + 1;
                float x0_00_1028 = _tmem_load_6[28];
                float x0_01_1029 = _tmem_load_6[29];
                float x1_00_1030 = _tmem_load_6[30];
                float x1_01_1031 = _tmem_load_6[31];
                float x0_10_1032 = _tmem_load_7[28];
                float x0_11_1033 = _tmem_load_7[29];
                float x1_10_1034 = _tmem_load_7[30];
                float x1_11_1035 = _tmem_load_7[31];
                float _max_124;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_124) : "f"(x0_00_1028), "f"(neg_cl));
                float _min_248;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_248) : "f"(_max_124), "f"(cl));
                float x0c_00_1036 = _min_248;
                float _max_125;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_125) : "f"(x0_01_1029), "f"(neg_cl));
                float _min_249;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_249) : "f"(_max_125), "f"(cl));
                float x0c_01_1037 = _min_249;
                float _max_126;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_126) : "f"(x0_10_1032), "f"(neg_cl));
                float _min_250;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_250) : "f"(_max_126), "f"(cl));
                float x0c_10_1038 = _min_250;
                float _max_127;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_127) : "f"(x0_11_1033), "f"(neg_cl));
                float _min_251;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_251) : "f"(_max_127), "f"(cl));
                float x0c_11_1039 = _min_251;
                float _fma_124;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_124) : "f"(x0c_00_1036), "f"(sg), "f"(0.0f));
                float lin_00_1040 = _fma_124;
                float _fma_125;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_125) : "f"(x0c_01_1037), "f"(sg), "f"(0.0f));
                float lin_01_1041 = _fma_125;
                float _fma_126;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_126) : "f"(x0c_10_1038), "f"(sg), "f"(0.0f));
                float lin_10_1042 = _fma_126;
                float _fma_127;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_127) : "f"(x0c_11_1039), "f"(sg), "f"(0.0f));
                float lin_11_1043 = _fma_127;
                float _mul_497;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_497) : "f"(x1_00_1030), "f"(fused));
                float _sub_125;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_125) : "f"(0.0f), "f"(_mul_497));
                float _exp2_124 = approx_exp2(_sub_125);
                float _add_124;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(1.0f), "f"(_exp2_124));
                float _rcp_124 = approx_rcp(_add_124);
                float sig_00_1044 = _rcp_124;
                float _mul_498;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_498) : "f"(x1_01_1031), "f"(fused));
                float _sub_126;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_126) : "f"(0.0f), "f"(_mul_498));
                float _exp2_125 = approx_exp2(_sub_126);
                float _add_125;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(1.0f), "f"(_exp2_125));
                float _rcp_125 = approx_rcp(_add_125);
                float sig_01_1045 = _rcp_125;
                float _mul_499;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_499) : "f"(x1_10_1034), "f"(fused));
                float _sub_127;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_127) : "f"(0.0f), "f"(_mul_499));
                float _exp2_126 = approx_exp2(_sub_127);
                float _add_126;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(1.0f), "f"(_exp2_126));
                float _rcp_126 = approx_rcp(_add_126);
                float sig_10_1046 = _rcp_126;
                float _mul_500;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_500) : "f"(x1_11_1035), "f"(fused));
                float _sub_128;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_128) : "f"(0.0f), "f"(_mul_500));
                float _exp2_127 = approx_exp2(_sub_128);
                float _add_127;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(1.0f), "f"(_exp2_127));
                float _rcp_127 = approx_rcp(_add_127);
                float sig_11_1047 = _rcp_127;
                float _mul_501;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_501) : "f"(x1_00_1030), "f"(sig_00_1044));
                float act_00_1048 = _mul_501;
                float _mul_502;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_502) : "f"(x1_01_1031), "f"(sig_01_1045));
                float act_01_1049 = _mul_502;
                float _mul_503;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_503) : "f"(x1_10_1034), "f"(sig_10_1046));
                float act_10_1050 = _mul_503;
                float _mul_504;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_504) : "f"(x1_11_1035), "f"(sig_11_1047));
                float act_11_1051 = _mul_504;
                float _min_252;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_252) : "f"(act_00_1048), "f"(cl));
                act_00_1048 = _min_252;
                float _min_253;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_253) : "f"(act_01_1049), "f"(cl));
                act_01_1049 = _min_253;
                float _min_254;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_254) : "f"(act_10_1050), "f"(cl));
                act_10_1050 = _min_254;
                float _min_255;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_255) : "f"(act_11_1051), "f"(cl));
                act_11_1051 = _min_255;
                float _mul_505;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_505) : "f"(lin_00_1040), "f"(act_00_1048));
                float y_00_1052 = _mul_505;
                float _mul_506;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_506) : "f"(lin_10_1042), "f"(act_10_1050));
                float y_10_1053 = _mul_506;
                float _mul_507;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_507) : "f"(lin_01_1041), "f"(act_01_1049));
                float y_01_1054 = _mul_507;
                float _mul_508;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_508) : "f"(lin_11_1043), "f"(act_11_1051));
                float y_11_1055 = _mul_508;
                float _mul_509;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_509) : "f"(y_00_1052), "f"(sc));
                quad[0] = _mul_509;
                float _mul_510;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_510) : "f"(y_10_1053), "f"(sc));
                quad[1] = _mul_510;
                float _mul_511;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_511) : "f"(y_01_1054), "f"(sc));
                quad[2] = _mul_511;
                float _mul_512;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_512) : "f"(y_11_1055), "f"(sc));
                quad[3] = _mul_512;
                uint32_t _fp8_31[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_31[0] = _packed;
                }
                int off0_1056 = local_token0_1026 * 64 + base_row_787;
                int off1_1057 = local_token1_1027 * 64 + base_row_787;
                int swz0_1058 = off0_1056 ^ (off0_1056 >> 7 & 3) << 4;
                int swz1_1059 = off1_1057 ^ (off1_1057 >> 7 & 3) << 4;
                epi_staging[swz0_1058 >> 1] = _fp8_31[0] & 65535;
                epi_staging[swz1_1059 >> 1] = _fp8_31[0] >> 16;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_3 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_3 + 192, 1073741824, n_tile * 256 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((mma_free_addr + (acc_stage) * 8) & 0xFEFFFFFF) : "memory");
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
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
                unsigned int ok = _clc_valid_3;
                if (_clc_ctaid_7 >= (unsigned int)bound) {
                    ok = 0;
                }
                if (ok == 0) {
                    break;
                }
                m_tile = _clc_ctaid_6 + (unsigned int)cta_rank;
                n_tile = _clc_ctaid_7;
            }
        }
    }
    // ---- Role: load_b ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            int bound_1 = num_non_exiting_ctas[0];
            int warp_local = warp - 4;
            int route_base = 0;
            int routed[16];
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m / 2 * grid_n; _tile_iter_1++) {
                if (m_tile_1 >= (unsigned int)grid_m || n_tile_1 >= (unsigned int)bound_1) {
                    break;
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)(warp_local * 4);
                for (int row = 0; row < 4; row++) {
                    routed[row] = route_map[route_base + row];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((8 + warp_local) * 4);
                for (int row_1 = 0; row_1 < 4; row_1++) {
                    routed[4 + row_1] = route_map[route_base + row_1];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((16 + warp_local) * 4);
                for (int row_2 = 0; row_2 < 4; row_2++) {
                    routed[8 + row_2] = route_map[route_base + row_2];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((24 + warp_local) * 4);
                for (int row_3 = 0; row_3 < 4; row_3++) {
                    routed[12 + row_3] = route_map[route_base + row_3];
                }
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    if (elect_sync()) {
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)(warp_local * 512), (&B), iter_k * 128, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((8 + warp_local) * 512), (&B), iter_k * 128, routed[4], routed[5], routed[6], routed[7], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((16 + warp_local) * 512), (&B), iter_k * 128, routed[8], routed[9], routed[10], routed[11], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((24 + warp_local) * 512), (&B), iter_k * 128, routed[12], routed[13], routed[14], routed[15], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                    }
                    if (warp == 4) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((b_full_addr + (stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 6) { stage = 0; _phase_k_done ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
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
                unsigned int ok_1 = _clc_valid_1;
                if (_clc_ctaid_3 >= (unsigned int)bound_1) {
                    ok_1 = 0;
                }
                if (ok_1 == 0) {
                    break;
                }
                m_tile_1 = _clc_ctaid_2 + (unsigned int)cta_rank;
                n_tile_1 = _clc_ctaid_3;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_a ----
    if (warp == 12) {
        { // load_a_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            int bound_2 = num_non_exiting_ctas[0];
            unsigned int cta_mask_1 = 1 << cta_rank;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_k_done_1 = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m / 2 * grid_n; _tile_iter_2++) {
                if (m_tile_2 >= (unsigned int)grid_m || n_tile_2 >= (unsigned int)bound_2) {
                    break;
                }
                int expert_1 = tile_expert[n_tile_2];
                int sole_reader = 1;
                if ((unsigned int)bound_2 > n_tile_2 + 1) {
                    if (tile_expert[n_tile_2 + 1] == expert_1) {
                        sole_reader = 0;
                    }
                }
                if (n_tile_2 > 0) {
                    if (tile_expert[n_tile_2 - 1] == expert_1) {
                        sole_reader = 0;
                    }
                }
                if (cta_rank == 0) {
                    mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                    mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                    throttle_stage += 1;
                    if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                }
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                    mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
                    if (elect_sync()) {
                        if (sole_reader == 1) {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7, %8;"
                                :: "r"(smem_a_addr + stage_1 * 16384), "l"((&A)), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k_1), "r"(expert_1),
                                   "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(smem_a_addr + stage_1 * 16384), "l"((&A)), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k_1), "r"(expert_1),
                                   "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)) : "memory");
                        }
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                    }
                    stage_1 += 1;
                    if (stage_1 == 6) { stage_1 = 0; _phase_k_done_1 ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
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
                unsigned int ok_2 = _clc_valid_0;
                if (_clc_ctaid_1 >= (unsigned int)bound_2) {
                    ok_2 = 0;
                }
                if (ok_2 == 0) {
                    break;
                }
                m_tile_2 = _clc_ctaid_0 + (unsigned int)cta_rank;
                n_tile_2 = _clc_ctaid_1;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 13) {
        { // mma_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_work_full_3 = 0;
            if (cta_rank == 0) {
                unsigned int stage_2 = 0;
                unsigned int acc_stage_1 = 0;
                unsigned int work_stage_3 = 0;
                unsigned int m_tile_3 = blockIdx.x;
                unsigned int n_tile_3 = blockIdx.y;
                int bound_3 = num_non_exiting_ctas[0];
                #pragma unroll 1
                for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m / 2 * grid_n; _tile_iter_3++) {
                    if (m_tile_3 >= (unsigned int)grid_m || n_tile_3 >= (unsigned int)bound_3) {
                        break;
                    }
                    mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                    #pragma unroll 1
                    for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                        mbarrier_wait(a_full_addr + (stage_2) * 8, _phase_a_full);
                        mbarrier_wait(b_full_addr + (stage_2) * 8, _phase_b_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        mma_ss_step_cg2(_mma_a_lo_0, _mma_b_lo_0, (tmem_accum + (acc_stage_1 * 256)), 272629776, ((((1) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        mma_ss_step_cg2(_mma_a_lo_1, _mma_b_lo_1, (tmem_accum + (acc_stage_1 * 256)), 272629776, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        mma_ss_step_cg2(_mma_a_lo_2, _mma_b_lo_2, (tmem_accum + (acc_stage_1 * 256)), 272629776, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        mma_ss_step_cg2(_mma_a_lo_3, _mma_b_lo_3, (tmem_accum + (acc_stage_1 * 256)), 272629776, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        elect_commit_cg2_multicast(k_done_addr + (stage_2) * 8, (uint16_t)(3));
                        if (iter_k_2 + 1 == K_tiles) {
                            elect_commit_cg2_multicast(mma_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                        }
                        stage_2 += 1;
                        if (stage_2 == 6) { stage_2 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; }
                    }
                    acc_stage_1 += 1;
                    if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                    mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
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
                        : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                        : "memory");
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
                        : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                        : "memory");
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
                        : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                        : "memory");
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
                    unsigned int ok_3 = _clc_valid_2;
                    if (_clc_ctaid_5 >= (unsigned int)bound_3) {
                        ok_3 = 0;
                    }
                    if (ok_3 == 0) {
                        break;
                    }
                    m_tile_3 = _clc_ctaid_4 + (unsigned int)cta_rank;
                    n_tile_3 = _clc_ctaid_5;
                }
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 14) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage_1 = 0;
            int bound_4 = num_non_exiting_ctas[0];
            unsigned int drain_stage = 0;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            unsigned int _phase_drain_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < grid_m / 2 * grid_n; _tile_iter_4++) {
                    mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                    mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                    throttle_stage_1 += 1;
                    if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                    mbarrier_wait_cluster_hint(work_empty_addr + (work_stage_4) * 8, _phase_work_empty, 10000000);
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
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
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
                    unsigned int ok_4 = _clc_valid_4;
                    if (_clc_ctaid_9 >= (unsigned int)bound_4) {
                        ok_4 = 0;
                    }
                    if (ok_4 == 0) {
                        if (_clc_valid_4 != 0) {
                            #pragma unroll 1
                            for (unsigned int _drain_iter = 0; _drain_iter < (grid_m / 2 * grid_n + 4 - 1) / 4 + 1; _drain_iter++) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(drain_full_addr + (drain_stage) * 8, 64);
                                    asm volatile(
                                        "fence.proxy.async.shared::cta;\n\t"
                                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                            ".mbarrier::complete_tx::bytes.b128"
                                            " [%0], [%1];"
                                        :: "r"(fast_drain_response_addr + drain_stage * 64 + 0 * 16), "r"(drain_full_addr + drain_stage * 8)
                                        : "memory");
                                    asm volatile(
                                        "fence.proxy.async.shared::cta;\n\t"
                                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                            ".mbarrier::complete_tx::bytes.b128"
                                            " [%0], [%1];"
                                        :: "r"(fast_drain_response_addr + drain_stage * 64 + 1 * 16), "r"(drain_full_addr + drain_stage * 8)
                                        : "memory");
                                    asm volatile(
                                        "fence.proxy.async.shared::cta;\n\t"
                                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                            ".mbarrier::complete_tx::bytes.b128"
                                            " [%0], [%1];"
                                        :: "r"(fast_drain_response_addr + drain_stage * 64 + 2 * 16), "r"(drain_full_addr + drain_stage * 8)
                                        : "memory");
                                    asm volatile(
                                        "fence.proxy.async.shared::cta;\n\t"
                                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                            ".mbarrier::complete_tx::bytes.b128"
                                            " [%0], [%1];"
                                        :: "r"(fast_drain_response_addr + drain_stage * 64 + 3 * 16), "r"(drain_full_addr + drain_stage * 8)
                                        : "memory");
                                }
                                mbarrier_wait(drain_full_addr + (drain_stage) * 8, _phase_drain_full);
                                unsigned int canceled = 0;
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
                                    : "r"(fast_drain_response_addr + drain_stage * 64 + 0 * 16)
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
                                    : "r"(fast_drain_response_addr + drain_stage * 64 + 1 * 16)
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
                                    : "r"(fast_drain_response_addr + drain_stage * 64 + 2 * 16)
                                    : "memory");
                                canceled += _clc_valid_7;
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
                                    : "r"(fast_drain_response_addr + drain_stage * 64 + 3 * 16)
                                    : "memory");
                                canceled += _clc_valid_8;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                _phase_drain_full ^= 1;
                                if (canceled == 0) {
                                    break;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 15) {
        { // padding_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int work_stage_5 = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            int bound_5 = num_non_exiting_ctas[0];
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < grid_m / 2 * grid_n; _tile_iter_5++) {
                if (m_tile_4 >= (unsigned int)grid_m || n_tile_4 >= (unsigned int)bound_5) {
                    break;
                }
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
                uint32_t _clc_valid_9 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_9)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_5 * 8), "r"(0) : "memory");
                work_stage_5 += 1;
                if (work_stage_5 == 3) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                unsigned int ok_5 = _clc_valid_9;
                if (_clc_ctaid_11 >= (unsigned int)bound_5) {
                    ok_5 = 0;
                }
                if (ok_5 == 0) {
                    break;
                }
                m_tile_4 = _clc_ctaid_10 + (unsigned int)cta_rank;
                n_tile_4 = _clc_ctaid_11;
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
