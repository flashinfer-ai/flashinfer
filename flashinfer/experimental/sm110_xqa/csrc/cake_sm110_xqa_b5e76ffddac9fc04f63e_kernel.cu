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

#include "cake_sm110_xqa_device_abi.cuh"

#define TMEM_NCOLS 256
#define TMEM_SCORES_OFFSET 0
#define TMEM_ACC_OFFSET 128
#define NUM_KV_PIPE_STAGES 4
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_KV_SMEM_OFF 17408
#define SMEM_KV_SMEM_STAGE_BYTES 16384
#define SMEM_KV_SMEM_STRIDE 16384
#define SMEM_V_SMEM_OFF 17408
#define SMEM_V_SMEM_STAGE_BYTES 16384
#define SMEM_V_SMEM_STRIDE 16384
#define SMEM_FLAG_SMEM_OFF 82944
#define SMEM_FLAG_SMEM_STAGE_BYTES 16
#define SMEM_FLAG_SMEM_STRIDE 16
#define SMEM_SCRATCH_OFF 1024
#define SMEM_SCRATCH_STAGE_BYTES 90640
#define SMEM_SCRATCH_STRIDE 90640
#define SMEM_TOTAL 91776
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>
#include "cake_sm110_xqa_device_helpers.cuh"
extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_sm110_xqa_b5e76ffddac9fc04f63e(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap KV, const int* __restrict__ sequence_lengths, __half* __restrict__ output, float* __restrict__ partial, float* __restrict__ statistics, unsigned int* __restrict__ counters, unsigned int heads, unsigned int ratio, unsigned int capacity, unsigned int partitions, unsigned int partition_tokens, float attention_scale)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define kv_full_addr (mbar_base + 8)
    #define kv_empty_addr (mbar_base + 40)
    #define s_full_addr (mbar_base + 72)
    #define p_full_addr (mbar_base + 80)
    #define o_full_addr (mbar_base + 88)
    #define tmem_retire_addr (mbar_base + 96)
    #define inbox_full_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __half* q_smem = reinterpret_cast<__half*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __half* kv_smem = reinterpret_cast<__half*>(smem_raw + 17408);
    const int kv_smem_addr = smem + 17408;
    __half* v_smem = reinterpret_cast<__half*>(smem_raw + 17408);
    const int v_smem_addr = smem + 17408;
    unsigned int* flag_smem = reinterpret_cast<unsigned int*>(smem_raw + 82944);
    const int flag_smem_addr = smem + 82944;
    unsigned int* scratch = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int scratch_addr = smem + 1024;
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV))) : "memory"); }

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 17 barriers)
    // Mbarriers at smem_raw[0..136)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // p_full: 1 barriers, init_count=32
            mbarrier_init(smem + 80, 32);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // tmem_retire: 1 barriers, init_count=32
            mbarrier_init(smem + 96, 32);
            // inbox_full: 4 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
            mbarrier_expect_tx(smem + 104, 2176);
            mbarrier_expect_tx(smem + 112, 2176);
            mbarrier_expect_tx(smem + 120, 2176);
            mbarrier_expect_tx(smem + 128, 2176);
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Kernel pre-TMEM ops
    unsigned int partition = (unsigned int)blockIdx.x;
    unsigned int head = (unsigned int)blockIdx.y;
    unsigned int batch = (unsigned int)blockIdx.z;
    unsigned int early_begin = partition * partition_tokens;
    unsigned int early_q_plane = batch * heads + head;
    unsigned int early_k_plane = batch * 2 * heads + head;
    unsigned int early_v_plane = (batch * 2 + 1) * heads + head;
    if (tid == 64) {
        mbarrier_arrive_expect_tx(q_full_addr, 16384);
        tma_3d_gmem2smem(q_smem_addr, (&Q), 0, 0, early_q_plane, q_full_addr);
        tma_3d_gmem2smem(q_smem_addr + 8192, (&Q), 64, 0, early_q_plane, q_full_addr);
        #pragma unroll
        for (int tile = 0; tile < 2; tile++) {
            unsigned int early_token = early_begin + (unsigned int)tile * 64;
            mbarrier_arrive_expect_tx(kv_full_addr + (2 * tile) * 8, 16384);
            tma_3d_gmem2smem(kv_smem_addr + (unsigned int)(2 * tile * 16384), (&KV), 0, early_token, early_k_plane, kv_full_addr + (2 * tile) * 8);
            tma_3d_gmem2smem(kv_smem_addr + (unsigned int)(2 * tile * 16384) + 8192, (&KV), 64, early_token, early_k_plane, kv_full_addr + (2 * tile) * 8);
            mbarrier_arrive_expect_tx(kv_full_addr + (2 * tile + 1) * 8, 16384);
            tma_3d_gmem2smem(kv_smem_addr + (unsigned int)((2 * tile + 1) * 16384), (&KV), 0, early_token, early_v_plane, kv_full_addr + (2 * tile + 1) * 8);
            tma_3d_gmem2smem(kv_smem_addr + (unsigned int)((2 * tile + 1) * 16384) + 8192, (&KV), 64, early_token, early_v_plane, kv_full_addr + (2 * tile + 1) * 8);
        }
    }

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 136);
    if (warp == 0) {
        int _tmem_hold = smem + 136;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_acc = taddr + 128;

    // ---- Role: softmax ----
    if (warp == 0) {
        { // softmax_main
            unsigned int partition_0 = (unsigned int)blockIdx.x;
            unsigned int head_1 = (unsigned int)blockIdx.y;
            unsigned int batch_2 = (unsigned int)blockIdx.z;
            unsigned int length = (unsigned int)sequence_lengths[batch_2];
            unsigned int partition_begin = partition_0 * partition_tokens;
            unsigned int _min_0 = ((length) < (partition_begin + partition_tokens) ? (length) : (partition_begin + partition_tokens));
            unsigned int partition_end = ((_min_0) > (partition_begin) ? (_min_0) : (partition_begin));
            unsigned int partition_length = partition_end - partition_begin;
            unsigned int tile_count = (partition_length + 63) / 64;
            unsigned int tail = partition_length - (tile_count - 1) * 64;
            unsigned int lane_1 = (unsigned int)lane;
            unsigned int my_row = lane_1 % 16;
            unsigned int col_half = lane_1 / 16;
            int row_valid = ((my_row < ratio) ? 1 : 0);
            float scale_log2 = attention_scale * 1.442695f;
            float m_used = -CUDART_INF_F;
            float row_sum = 0.0f;
            unsigned int _phase_s_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_1 = 0; tile_1 < tile_count; tile_1++) {
                mbarrier_wait(s_full_addr, _phase_s_full_0);
                _phase_s_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int valid_cols = (int)(partition_length - tile_1 * 64);
                if (valid_cols > 64) {
                    valid_cols = 64;
                }
                if (row_valid == 0) {
                    valid_cols = 0;
                }
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 32;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                    : "r"(taddr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int half_valid = valid_cols - (int)col_half * 32;
                if (half_valid < 0) {
                    half_valid = 0;
                }
                if (half_valid > 32) {
                    half_valid = 32;
                }
                #if __CUDA_ARCH__ >= 1000
                const float2 _scale2_1 = {scale_log2, scale_log2};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_1);
                #else
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++) {
                    _tmem_load_0[_ls] = _tmem_load_0[_ls] * scale_log2;
                }
                #endif
                if (half_valid <= 0) {
                    if (!(0 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                    if (!(0 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                    if (!(0 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                    if (!(0 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                    if (!(0 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                    if (!(0 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                    if (!(0 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                    if (!(0 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                    if (!(0 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                    if (!(0 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                    if (!(0 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                    if (!(0 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                    if (!(0 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                    if (!(0 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                    if (!(0 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                    if (!(0 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                    if (!(0 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                    if (!(0 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                    if (!(0 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                    if (!(0 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                    if (!(0 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                    if (!(0 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                    if (!(0 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                    if (!(0 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                    if (!(0 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                    if (!(0 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                    if (!(0 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                    if (!(0 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                    if (!(0 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                    if (!(0 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                    if (!(0 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                    if (!(0 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
                } else if (half_valid < 32) {
                    uint32_t _thresh_mask_2;
                    {
                        int _lim_3 = half_valid;
                        if (_lim_3 <= 0) { _thresh_mask_2 = 0u; }
                        else if (_lim_3 >= 32) { _thresh_mask_2 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_2) : "r"(_lim_3));
                        }
                    }
                    if (!(_thresh_mask_2 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                    if (!(_thresh_mask_2 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
                }
                float2 _reg_reduce_max2_4 = {-CAKE_INF, -CAKE_INF};
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[0], _tmem_load_0[1]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[2], _tmem_load_0[3]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[4], _tmem_load_0[5]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[6], _tmem_load_0[7]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[8], _tmem_load_0[9]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[10], _tmem_load_0[11]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[12], _tmem_load_0[13]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[14], _tmem_load_0[15]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[16], _tmem_load_0[17]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[18], _tmem_load_0[19]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[20], _tmem_load_0[21]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[22], _tmem_load_0[23]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[24], _tmem_load_0[25]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[26], _tmem_load_0[27]));
                _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(_tmem_load_0[28], _tmem_load_0[29]));
                _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(_tmem_load_0[30], _tmem_load_0[31]));
                float tile_max = row_max_reduce(_reg_reduce_max2_4);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, tile_max, 16);
                tile_max = max_noftz(tile_max, _shfl_xor_sync_5);
                float m_new;
                asm("max.f32 %0, %1, %2;" : "=f"(m_new) : "f"(m_used), "f"(tile_max));
                float alpha = 1.0f;
                if (m_used == -CUDART_INF_F) {
                    m_used = m_new;
                } else if (m_new > m_used) {
                    float _exp2_6 = approx_exp2(m_used - m_new);
                    alpha = _exp2_6;
                    m_used = m_new;
                }
                int _warp_vote_7 = __any_sync(0xFFFFFFFF, alpha != 1.0f);
                if (_warp_vote_7 != 0) {
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                        : "r"(taddr + 128));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                        : "r"(taddr + 128 + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_8 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_8);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++) {
                        _tmem_load_1[_ls] = _tmem_load_1[_ls] * alpha;
                    }
                    #endif
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                        " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                        :: "r"(taddr + 128), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[63])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                }
                float score_bias = ((m_used != -CUDART_INF_F) ? -m_used : 0.0f);
                const float2 _fma_b2_9 = {1.0f, 1.0f};
                const float2 _fma_c2_10 = {score_bias, score_bias};
                float2 _fma_pair_11 = fma_f32x2(make_float2(_tmem_load_0[0], _tmem_load_0[1]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[0] = _fma_pair_11.x;
                _tmem_load_0[1] = _fma_pair_11.y;
                float2 _fma_pair_12 = fma_f32x2(make_float2(_tmem_load_0[2], _tmem_load_0[3]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[2] = _fma_pair_12.x;
                _tmem_load_0[3] = _fma_pair_12.y;
                float2 _fma_pair_13 = fma_f32x2(make_float2(_tmem_load_0[4], _tmem_load_0[5]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[4] = _fma_pair_13.x;
                _tmem_load_0[5] = _fma_pair_13.y;
                float2 _fma_pair_14 = fma_f32x2(make_float2(_tmem_load_0[6], _tmem_load_0[7]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[6] = _fma_pair_14.x;
                _tmem_load_0[7] = _fma_pair_14.y;
                float2 _fma_pair_15 = fma_f32x2(make_float2(_tmem_load_0[8], _tmem_load_0[9]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[8] = _fma_pair_15.x;
                _tmem_load_0[9] = _fma_pair_15.y;
                float2 _fma_pair_16 = fma_f32x2(make_float2(_tmem_load_0[10], _tmem_load_0[11]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[10] = _fma_pair_16.x;
                _tmem_load_0[11] = _fma_pair_16.y;
                float2 _fma_pair_17 = fma_f32x2(make_float2(_tmem_load_0[12], _tmem_load_0[13]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[12] = _fma_pair_17.x;
                _tmem_load_0[13] = _fma_pair_17.y;
                float2 _fma_pair_18 = fma_f32x2(make_float2(_tmem_load_0[14], _tmem_load_0[15]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[14] = _fma_pair_18.x;
                _tmem_load_0[15] = _fma_pair_18.y;
                float2 _fma_pair_19 = fma_f32x2(make_float2(_tmem_load_0[16], _tmem_load_0[17]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[16] = _fma_pair_19.x;
                _tmem_load_0[17] = _fma_pair_19.y;
                float2 _fma_pair_20 = fma_f32x2(make_float2(_tmem_load_0[18], _tmem_load_0[19]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[18] = _fma_pair_20.x;
                _tmem_load_0[19] = _fma_pair_20.y;
                float2 _fma_pair_21 = fma_f32x2(make_float2(_tmem_load_0[20], _tmem_load_0[21]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[20] = _fma_pair_21.x;
                _tmem_load_0[21] = _fma_pair_21.y;
                float2 _fma_pair_22 = fma_f32x2(make_float2(_tmem_load_0[22], _tmem_load_0[23]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[22] = _fma_pair_22.x;
                _tmem_load_0[23] = _fma_pair_22.y;
                float2 _fma_pair_23 = fma_f32x2(make_float2(_tmem_load_0[24], _tmem_load_0[25]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[24] = _fma_pair_23.x;
                _tmem_load_0[25] = _fma_pair_23.y;
                float2 _fma_pair_24 = fma_f32x2(make_float2(_tmem_load_0[26], _tmem_load_0[27]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[26] = _fma_pair_24.x;
                _tmem_load_0[27] = _fma_pair_24.y;
                float2 _fma_pair_25 = fma_f32x2(make_float2(_tmem_load_0[28], _tmem_load_0[29]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[28] = _fma_pair_25.x;
                _tmem_load_0[29] = _fma_pair_25.y;
                float2 _fma_pair_26 = fma_f32x2(make_float2(_tmem_load_0[30], _tmem_load_0[31]), _fma_b2_9, _fma_c2_10);
                _tmem_load_0[30] = _fma_pair_26.x;
                _tmem_load_0[31] = _fma_pair_26.y;
                #pragma unroll
                for (int _le = 0; _le < 32; _le++) {
                    _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                }
                float2 _reg_reduce_sum2_27 = make_float2(0.0f, 0.0f);
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[4], _tmem_load_0[5]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[6], _tmem_load_0[7]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[8], _tmem_load_0[9]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[10], _tmem_load_0[11]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[12], _tmem_load_0[13]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[14], _tmem_load_0[15]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[16], _tmem_load_0[17]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[18], _tmem_load_0[19]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[20], _tmem_load_0[21]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[22], _tmem_load_0[23]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[24], _tmem_load_0[25]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[26], _tmem_load_0[27]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[28], _tmem_load_0[29]));
                _reg_reduce_sum2_27 = add_f32x2(_reg_reduce_sum2_27, make_float2(_tmem_load_0[30], _tmem_load_0[31]));
                float half_sum = _reg_reduce_sum2_27.x + _reg_reduce_sum2_27.y;
                float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, half_sum, 16);
                float tile_sum = half_sum + _shfl_xor_sync_28;
                row_sum = __fmaf_rn(row_sum, alpha, tile_sum);
                unsigned int packed_p[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    packed_p[_lp] = *(uint32_t*)&_h2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                    " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + 32), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[15])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_full_addr);
            }
            float output_values[64] = {0};
            unsigned int _phase_o_full_0 = 0;
            if (tile_count > 0) {
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&output_values[0])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[1])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[2])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[3])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[4])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[5])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[6])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[7])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[8])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[9])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[10])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[11])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[12])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[13])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[14])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[15])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[16])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[17])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[18])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[19])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[20])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[21])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[22])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[23])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[24])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[25])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[26])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[27])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[28])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[29])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[30])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[31]))
                    : "r"(taddr + 128));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&output_values[32])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[33])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[34])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[35])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[36])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[37])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[38])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[39])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[40])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[41])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[42])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[43])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[44])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[45])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[46])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[47])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[48])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[49])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[50])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[51])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[52])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[53])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[54])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[55])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[56])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[57])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[58])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[59])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[60])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[61])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[62])), "=r"(*reinterpret_cast<uint32_t*>(&output_values[63]))
                    : "r"(taddr + 128 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_retire_addr);
            unsigned int col_base = col_half * 64;
            unsigned int query_row = (batch_2 * heads + head_1) * ratio + my_row;
            unsigned int rank = (unsigned int)cta_rank;
            float zero_f32 = 0.0f;
            unsigned int _phase_q_full_0 = 0;
            if (rank != 0) {
                unsigned int slot = rank - 1;
                uint32_t _mapa_0;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_0) : "r"(scratch_addr + (81936 + slot * 8192 + lane_1 * 256)), "r"(0));
                uint32_t _mapa_1;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_1) : "r"(scratch_addr + (90128 + slot * 512 + lane_1 * 16)), "r"(0));
                uint32_t _mapa_2;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mapa_2) : "r"(inbox_full_addr + my_row / 4 * 8), "r"(0));
                if (row_valid != 0) {
                    #pragma unroll
                    for (int offset = 0; offset < 64; offset += 4) {
                        asm volatile(
                            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                            :: "r"(_mapa_0 + (unsigned int)offset * 4), "r"(__float_as_uint(output_values[offset])), "r"(__float_as_uint(output_values[offset + 1])), "r"(__float_as_uint(output_values[offset + 2])), "r"(__float_as_uint(output_values[offset + 3])), "r"(_mapa_2) : "memory");
                    }
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_mapa_1), "r"(__float_as_uint(m_used)), "r"(__float_as_uint(row_sum)), "r"(__float_as_uint(zero_f32)), "r"(__float_as_uint(zero_f32)), "r"(_mapa_2) : "memory");
                }
            } else {
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                if (row_valid != 0) {
                    #pragma unroll
                    for (int offset_1 = 0; offset_1 < 64; offset_1 += 4) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(scratch_addr + (lane_1 * 256 + (unsigned int)offset_1 * 4)), "r"(__float_as_uint(output_values[offset_1])), "r"(__float_as_uint(output_values[offset_1 + 1])), "r"(__float_as_uint(output_values[offset_1 + 2])), "r"(__float_as_uint(output_values[offset_1 + 3])) : "memory");
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(scratch_addr + (8192 + lane_1 * 16)), "r"(__float_as_uint(m_used)), "r"(__float_as_uint(row_sum)), "r"(__float_as_uint(zero_f32)), "r"(__float_as_uint(zero_f32)) : "memory");
                }
                asm volatile("fence.acq_rel.cta;" ::: "memory");
                __syncwarp();
                if (lane_1 == 0) {
                    #pragma unroll 1
                    for (unsigned int live_group = 0; live_group < ratio / 4; live_group++) {
                        mbarrier_arrive(inbox_full_addr + (live_group) * 8);
                    }
                }
                unsigned int rank_0 = (unsigned int)cta_rank;
                if (rank_0 == 0) {
                    #pragma unroll 1
                    for (unsigned int live_group_1 = 0; live_group_1 < ratio / 4; live_group_1++) {
                        mbarrier_wait_cluster_hint(inbox_full_addr + (live_group_1) * 8, 0, 10000000);
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    unsigned int merge_col = lane_1 % 16 * 8;
                    unsigned int group_row = (batch_2 * heads + head_1) * ratio;
                    #pragma unroll 1
                    for (unsigned int group = 0; group < (ratio + 7) / 8; group++) {
                        unsigned int local_row = group * 8 + lane_1 / 16;
                        if (local_row < ratio) {
                            unsigned int merge_row = group_row + local_row;
                            unsigned int source_lane = merge_col / 64 * 16 + local_row;
                            unsigned int partial_offset = source_lane * 256 + merge_col % 64 * 4;
                            unsigned int stats_offset = local_row * 16;
                            float maxima[2] = {0};
                            float sums[2] = {0};
                            float record[4] = {0};
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&record[0])), "=r"(*reinterpret_cast<uint32_t*>(&record[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&record[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&record[(0) + 3]))
                                : "r"(scratch_addr + (8192 + stats_offset)));
                            maxima[0] = record[0];
                            sums[0] = record[1];
                            float record_0[4] = {0};
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&record_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&record_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&record_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&record_0[(0) + 3]))
                                : "r"(scratch_addr + (90128 + stats_offset)));
                            maxima[1] = record_0[0];
                            sums[1] = record_0[1];
                            float merge_max = -CUDART_INF_F;
                            merge_max = max_noftz(merge_max, maxima[0]);
                            merge_max = max_noftz(merge_max, maxima[1]);
                            float safe_merge_max = ((merge_max == -CUDART_INF_F) ? 0.0f : merge_max);
                            float numerator[8] = {0};
                            float denominator = 0.0f;
                            float shard_values[8] = {0};
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&shard_values[0])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values[(0) + 3]))
                                : "r"(scratch_addr + partial_offset));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&shard_values[4])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values[(4) + 3]))
                                : "r"(scratch_addr + (partial_offset + 16)));
                            float weight = approx_exp2(maxima[0] - safe_merge_max);
                            #pragma unroll
                            for (int element = 0; element < 8; element++) {
                                numerator[element] = __fmaf_rn(shard_values[element], weight, numerator[element]);
                            }
                            denominator = __fmaf_rn(sums[0], weight, denominator);
                            float shard_values_1[8] = {0};
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1[(0) + 3]))
                                : "r"(scratch_addr + (81936 + partial_offset)));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1[(4) + 3]))
                                : "r"(scratch_addr + (81936 + partial_offset + 16)));
                            float weight_2 = approx_exp2(maxima[1] - safe_merge_max);
                            #pragma unroll
                            for (int element_1 = 0; element_1 < 8; element_1++) {
                                numerator[element_1] = __fmaf_rn(shard_values_1[element_1], weight_2, numerator[element_1]);
                            }
                            denominator = __fmaf_rn(sums[1], weight_2, denominator);
                            float _rcp_29 = __frcp_rn(denominator);
                            float inverse = ((denominator > 0.0f) ? _rcp_29 : 0.0f);
                            {
                                const float2 _prescale2_30 = {inverse, inverse};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 4; _ps++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(&numerator[0])[_ps], _prescale2_30);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 8; _ps++)
                                    numerator[0 + _ps] *= inverse;
                                #endif
                                __half2 _pk[4];
                                _pk[0] = __floats2half2_rn(numerator[0 + 0], numerator[0 + 1]);
                                _pk[1] = __floats2half2_rn(numerator[0 + 2], numerator[0 + 3]);
                                _pk[2] = __floats2half2_rn(numerator[0 + 4], numerator[0 + 5]);
                                _pk[3] = __floats2half2_rn(numerator[0 + 6], numerator[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__half*)(output + (merge_row * 128 + merge_col)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 1) {
        { // mma_warp_main
            unsigned int partition_0_1 = (unsigned int)blockIdx.x;
            unsigned int head_1_1 = (unsigned int)blockIdx.y;
            unsigned int batch_2_1 = (unsigned int)blockIdx.z;
            unsigned int length_1 = (unsigned int)sequence_lengths[batch_2_1];
            unsigned int partition_begin_1 = partition_0_1 * partition_tokens;
            unsigned int _min_0_1 = ((length_1) < (partition_begin_1 + partition_tokens) ? (length_1) : (partition_begin_1 + partition_tokens));
            unsigned int partition_end_1 = ((_min_0_1) > (partition_begin_1) ? (_min_0_1) : (partition_begin_1));
            unsigned int partition_length_1 = partition_end_1 - partition_begin_1;
            unsigned int tile_count_1 = (partition_length_1 + 63) / 64;
            unsigned int tail_1 = partition_length_1 - (tile_count_1 - 1) * 64;
            unsigned int lane_2 = (unsigned int)lane;
            unsigned int zero = 0;
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 0;
            int first_pv = 1;
            unsigned int _phase_q_full_0_1 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0_1);
            _phase_q_full_0_1 ^= 1;
            unsigned int _phase_p_full_0 = 0;
            #pragma unroll 1
            for (unsigned int tile_2 = 0; tile_2 < tile_count_1; tile_2++) {
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int k_stage = kv_stage;
                kv_stage += 1;
                if (kv_stage == 4) { kv_stage = 0; kv_phase ^= 1; }
                uint32_t _elected_1 = elect_sync();
                int _smem_desc_lo_2 = ((q_smem_addr) >> 4) & 0x3FFF;
                int _smem_desc_lo_3 = (((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "setp.ne.b32 leader, %4, 0;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68157456;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_scores), "r"(0), "r"(_elected_1) : "memory");
                tcgen05_commit_leader(s_full_addr, _elected_1);
                tcgen05_commit_leader(kv_empty_addr + (k_stage) * 8, _elected_1);
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                unsigned int v_stage = kv_stage;
                kv_stage += 1;
                if (kv_stage == 4) { kv_stage = 0; kv_phase ^= 1; }
                if (tile_2 + 1 == tile_count_1 && tail_1 < 64) {
                    unsigned int stage_base = v_stage * 16384;
                    for (unsigned int chunk = tail_1 * 8 + lane_2; chunk < 512; chunk += 32) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(kv_smem_addr + (stage_base + chunk * 16)), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(kv_smem_addr + (stage_base + 8192 + chunk * 16)), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                }
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    int _smem_desc_lo_4 = ((((v_smem_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 1024;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 69271568;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_acc), "r"(_smem_desc_lo_4), "r"(tmem_scores + 32), "r"(((first_pv) ? 0 : 1)) : "memory");
                    if (tile_2 + 1 == tile_count_1) {
                        tcgen05_commit(o_full_addr);
                    }
                    tcgen05_commit(kv_empty_addr + (v_stage) * 8);
                }
                first_pv = 0;
            }
            unsigned int _phase_tmem_retire_0 = 0;
            mbarrier_wait(tmem_retire_addr, _phase_tmem_retire_0);
            _phase_tmem_retire_0 ^= 1;
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            asm volatile("tcgen05.fence::before_thread_sync;");
            unsigned int rank_1 = (unsigned int)cta_rank;
            if (rank_1 == 0) {
                #pragma unroll 1
                for (unsigned int live_group_2 = 0; live_group_2 < ratio / 4; live_group_2++) {
                    mbarrier_wait_cluster_hint(inbox_full_addr + (live_group_2) * 8, 0, 10000000);
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                unsigned int merge_col_1 = lane_2 % 16 * 8;
                unsigned int group_row_1 = (batch_2_1 * heads + head_1_1) * ratio;
                #pragma unroll 1
                for (unsigned int group_1 = 0; group_1 < (ratio + 7) / 8; group_1++) {
                    unsigned int local_row_1 = group_1 * 8 + 2 + lane_2 / 16;
                    if (local_row_1 < ratio) {
                        unsigned int merge_row_1 = group_row_1 + local_row_1;
                        unsigned int source_lane_1 = merge_col_1 / 64 * 16 + local_row_1;
                        unsigned int partial_offset_1 = source_lane_1 * 256 + merge_col_1 % 64 * 4;
                        unsigned int stats_offset_1 = local_row_1 * 16;
                        float maxima_1[2] = {0};
                        float sums_1[2] = {0};
                        float record_1[4] = {0};
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&record_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&record_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&record_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&record_1[(0) + 3]))
                            : "r"(scratch_addr + (8192 + stats_offset_1)));
                        maxima_1[0] = record_1[0];
                        sums_1[0] = record_1[1];
                        float record_0_1[4] = {0};
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&record_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&record_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&record_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&record_0_1[(0) + 3]))
                            : "r"(scratch_addr + (90128 + stats_offset_1)));
                        maxima_1[1] = record_0_1[0];
                        sums_1[1] = record_0_1[1];
                        float merge_max_1 = -CUDART_INF_F;
                        merge_max_1 = max_noftz(merge_max_1, maxima_1[0]);
                        merge_max_1 = max_noftz(merge_max_1, maxima_1[1]);
                        float safe_merge_max_1 = ((merge_max_1 == -CUDART_INF_F) ? 0.0f : merge_max_1);
                        float numerator_1[8] = {0};
                        float denominator_1 = 0.0f;
                        float shard_values_2[8] = {0};
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_2[(0) + 3]))
                            : "r"(scratch_addr + partial_offset_1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_2[(4) + 3]))
                            : "r"(scratch_addr + (partial_offset_1 + 16)));
                        float weight_1 = approx_exp2(maxima_1[0] - safe_merge_max_1);
                        #pragma unroll
                        for (int element_2 = 0; element_2 < 8; element_2++) {
                            numerator_1[element_2] = __fmaf_rn(shard_values_2[element_2], weight_1, numerator_1[element_2]);
                        }
                        denominator_1 = __fmaf_rn(sums_1[0], weight_1, denominator_1);
                        float shard_values_1_1[8] = {0};
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_1[(0) + 3]))
                            : "r"(scratch_addr + (81936 + partial_offset_1)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_1[(4) + 3]))
                            : "r"(scratch_addr + (81936 + partial_offset_1 + 16)));
                        float weight_2_1 = approx_exp2(maxima_1[1] - safe_merge_max_1);
                        #pragma unroll
                        for (int element_3 = 0; element_3 < 8; element_3++) {
                            numerator_1[element_3] = __fmaf_rn(shard_values_1_1[element_3], weight_2_1, numerator_1[element_3]);
                        }
                        denominator_1 = __fmaf_rn(sums_1[1], weight_2_1, denominator_1);
                        float _rcp_5 = __frcp_rn(denominator_1);
                        float inverse_1 = ((denominator_1 > 0.0f) ? _rcp_5 : 0.0f);
                        {
                            const float2 _prescale2_6 = {inverse_1, inverse_1};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 4; _ps++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(&numerator_1[0])[_ps], _prescale2_6);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                numerator_1[0 + _ps] *= inverse_1;
                            #endif
                            __half2 _pk[4];
                            _pk[0] = __floats2half2_rn(numerator_1[0 + 0], numerator_1[0 + 1]);
                            _pk[1] = __floats2half2_rn(numerator_1[0 + 2], numerator_1[0 + 3]);
                            _pk[2] = __floats2half2_rn(numerator_1[0 + 4], numerator_1[0 + 5]);
                            _pk[3] = __floats2half2_rn(numerator_1[0 + 6], numerator_1[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__half*)(output + (merge_row_1 * 128 + merge_col_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp == 2) {
        { // load_warp_main
            unsigned int partition_0_2 = (unsigned int)blockIdx.x;
            unsigned int head_1_2 = (unsigned int)blockIdx.y;
            unsigned int batch_2_2 = (unsigned int)blockIdx.z;
            unsigned int issue_begin = partition_0_2 * partition_tokens;
            unsigned int lane_3 = (unsigned int)lane;
            unsigned int q_plane = batch_2_2 * heads + head_1_2;
            unsigned int k_plane = batch_2_2 * 2 * heads + head_1_2;
            unsigned int v_plane = (batch_2_2 * 2 + 1) * heads + head_1_2;
            unsigned int load_stage = 0;
            unsigned int load_phase = 0;
            unsigned int length_2 = (unsigned int)sequence_lengths[batch_2_2];
            unsigned int partition_begin_2 = partition_0_2 * partition_tokens;
            unsigned int _min_0_2 = ((length_2) < (partition_begin_2 + partition_tokens) ? (length_2) : (partition_begin_2 + partition_tokens));
            unsigned int partition_end_2 = ((_min_0_2) > (partition_begin_2) ? (_min_0_2) : (partition_begin_2));
            unsigned int partition_length_2 = partition_end_2 - partition_begin_2;
            unsigned int tile_count_2 = (partition_length_2 + 63) / 64;
            unsigned int tail_2 = partition_length_2 - (tile_count_2 - 1) * 64;
            #pragma unroll 1
            for (unsigned int tile_3 = 2; tile_3 < tile_count_2; tile_3++) {
                unsigned int token = partition_begin_2 + tile_3 * 64;
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, load_phase);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                    tma_3d_gmem2smem(kv_smem_addr + load_stage * 16384, (&KV), 0, token, k_plane, kv_full_addr + (load_stage) * 8);
                    tma_3d_gmem2smem(kv_smem_addr + load_stage * 16384 + 8192, (&KV), 64, token, k_plane, kv_full_addr + (load_stage) * 8);
                }
                load_stage += 1;
                if (load_stage == 4) { load_stage = 0; load_phase ^= 1; }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, load_phase);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                    tma_3d_gmem2smem(kv_smem_addr + load_stage * 16384, (&KV), 0, token, v_plane, kv_full_addr + (load_stage) * 8);
                    tma_3d_gmem2smem(kv_smem_addr + load_stage * 16384 + 8192, (&KV), 64, token, v_plane, kv_full_addr + (load_stage) * 8);
                }
                load_stage += 1;
                if (load_stage == 4) { load_stage = 0; load_phase ^= 1; }
            }
            #pragma unroll
            for (int stage = 0; stage < 4; stage++) {
                if ((unsigned int)stage >= tile_count_2 * 2) {
                    mbarrier_wait(kv_full_addr + (stage) * 8, 0);
                }
            }
            unsigned int rank_2 = (unsigned int)cta_rank;
            if (rank_2 == 0) {
                #pragma unroll 1
                for (unsigned int live_group_3 = 0; live_group_3 < ratio / 4; live_group_3++) {
                    mbarrier_wait_cluster_hint(inbox_full_addr + (live_group_3) * 8, 0, 10000000);
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                unsigned int merge_col_2 = lane_3 % 16 * 8;
                unsigned int group_row_2 = (batch_2_2 * heads + head_1_2) * ratio;
                #pragma unroll 1
                for (unsigned int group_2 = 0; group_2 < (ratio + 7) / 8; group_2++) {
                    unsigned int local_row_2 = group_2 * 8 + 4 + lane_3 / 16;
                    if (local_row_2 < ratio) {
                        unsigned int merge_row_2 = group_row_2 + local_row_2;
                        unsigned int source_lane_2 = merge_col_2 / 64 * 16 + local_row_2;
                        unsigned int partial_offset_2 = source_lane_2 * 256 + merge_col_2 % 64 * 4;
                        unsigned int stats_offset_2 = local_row_2 * 16;
                        float maxima_2[2] = {0};
                        float sums_2[2] = {0};
                        float record_2[4] = {0};
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&record_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&record_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&record_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&record_2[(0) + 3]))
                            : "r"(scratch_addr + (8192 + stats_offset_2)));
                        maxima_2[0] = record_2[0];
                        sums_2[0] = record_2[1];
                        float record_0_2[4] = {0};
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&record_0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&record_0_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&record_0_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&record_0_2[(0) + 3]))
                            : "r"(scratch_addr + (90128 + stats_offset_2)));
                        maxima_2[1] = record_0_2[0];
                        sums_2[1] = record_0_2[1];
                        float merge_max_2 = -CUDART_INF_F;
                        merge_max_2 = max_noftz(merge_max_2, maxima_2[0]);
                        merge_max_2 = max_noftz(merge_max_2, maxima_2[1]);
                        float safe_merge_max_2 = ((merge_max_2 == -CUDART_INF_F) ? 0.0f : merge_max_2);
                        float numerator_2[8] = {0};
                        float denominator_2 = 0.0f;
                        float shard_values_3[8] = {0};
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_3[(0) + 3]))
                            : "r"(scratch_addr + partial_offset_2));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_3[(4) + 3]))
                            : "r"(scratch_addr + (partial_offset_2 + 16)));
                        float weight_3 = approx_exp2(maxima_2[0] - safe_merge_max_2);
                        #pragma unroll
                        for (int element_4 = 0; element_4 < 8; element_4++) {
                            numerator_2[element_4] = __fmaf_rn(shard_values_3[element_4], weight_3, numerator_2[element_4]);
                        }
                        denominator_2 = __fmaf_rn(sums_2[0], weight_3, denominator_2);
                        float shard_values_1_2[8] = {0};
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_2[(0) + 3]))
                            : "r"(scratch_addr + (81936 + partial_offset_2)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_2[(4) + 3]))
                            : "r"(scratch_addr + (81936 + partial_offset_2 + 16)));
                        float weight_2_2 = approx_exp2(maxima_2[1] - safe_merge_max_2);
                        #pragma unroll
                        for (int element_5 = 0; element_5 < 8; element_5++) {
                            numerator_2[element_5] = __fmaf_rn(shard_values_1_2[element_5], weight_2_2, numerator_2[element_5]);
                        }
                        denominator_2 = __fmaf_rn(sums_2[1], weight_2_2, denominator_2);
                        float _rcp_1 = __frcp_rn(denominator_2);
                        float inverse_2 = ((denominator_2 > 0.0f) ? _rcp_1 : 0.0f);
                        {
                            const float2 _prescale2_2 = {inverse_2, inverse_2};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 4; _ps++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(&numerator_2[0])[_ps], _prescale2_2);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                numerator_2[0 + _ps] *= inverse_2;
                            #endif
                            __half2 _pk[4];
                            _pk[0] = __floats2half2_rn(numerator_2[0 + 0], numerator_2[0 + 1]);
                            _pk[1] = __floats2half2_rn(numerator_2[0 + 2], numerator_2[0 + 3]);
                            _pk[2] = __floats2half2_rn(numerator_2[0 + 4], numerator_2[0 + 5]);
                            _pk[3] = __floats2half2_rn(numerator_2[0 + 6], numerator_2[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__half*)(output + (merge_row_2 * 128 + merge_col_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
        }
    }
    // ---- Role: aux ----
    if (warp == 3) {
        { // aux_main
            unsigned int partition_0_3 = (unsigned int)blockIdx.x;
            unsigned int head_1_3 = (unsigned int)blockIdx.y;
            unsigned int batch_2_3 = (unsigned int)blockIdx.z;
            unsigned int length_3 = (unsigned int)sequence_lengths[batch_2_3];
            unsigned int partition_begin_3 = partition_0_3 * partition_tokens;
            unsigned int _min_0_3 = ((length_3) < (partition_begin_3 + partition_tokens) ? (length_3) : (partition_begin_3 + partition_tokens));
            unsigned int partition_end_3 = ((_min_0_3) > (partition_begin_3) ? (_min_0_3) : (partition_begin_3));
            unsigned int partition_length_3 = partition_end_3 - partition_begin_3;
            unsigned int tile_count_3 = (partition_length_3 + 63) / 64;
            unsigned int tail_3 = partition_length_3 - (tile_count_3 - 1) * 64;
            unsigned int lane_4 = (unsigned int)lane;
            unsigned int rank_3 = (unsigned int)cta_rank;
            if (rank_3 == 0) {
                #pragma unroll 1
                for (unsigned int live_group_4 = 0; live_group_4 < ratio / 4; live_group_4++) {
                    mbarrier_wait_cluster_hint(inbox_full_addr + (live_group_4) * 8, 0, 10000000);
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                unsigned int merge_col_3 = lane_4 % 16 * 8;
                unsigned int group_row_3 = (batch_2_3 * heads + head_1_3) * ratio;
                #pragma unroll 1
                for (unsigned int group_3 = 0; group_3 < (ratio + 7) / 8; group_3++) {
                    unsigned int local_row_3 = group_3 * 8 + 6 + lane_4 / 16;
                    if (local_row_3 < ratio) {
                        unsigned int merge_row_3 = group_row_3 + local_row_3;
                        unsigned int source_lane_3 = merge_col_3 / 64 * 16 + local_row_3;
                        unsigned int partial_offset_3 = source_lane_3 * 256 + merge_col_3 % 64 * 4;
                        unsigned int stats_offset_3 = local_row_3 * 16;
                        float maxima_3[2] = {0};
                        float sums_3[2] = {0};
                        float record_3[4] = {0};
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&record_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&record_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&record_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&record_3[(0) + 3]))
                            : "r"(scratch_addr + (8192 + stats_offset_3)));
                        maxima_3[0] = record_3[0];
                        sums_3[0] = record_3[1];
                        float record_0_3[4] = {0};
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&record_0_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&record_0_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&record_0_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&record_0_3[(0) + 3]))
                            : "r"(scratch_addr + (90128 + stats_offset_3)));
                        maxima_3[1] = record_0_3[0];
                        sums_3[1] = record_0_3[1];
                        float merge_max_3 = -CUDART_INF_F;
                        merge_max_3 = max_noftz(merge_max_3, maxima_3[0]);
                        merge_max_3 = max_noftz(merge_max_3, maxima_3[1]);
                        float safe_merge_max_3 = ((merge_max_3 == -CUDART_INF_F) ? 0.0f : merge_max_3);
                        float numerator_3[8] = {0};
                        float denominator_3 = 0.0f;
                        float shard_values_4[8] = {0};
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_4[(0) + 3]))
                            : "r"(scratch_addr + partial_offset_3));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_4[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_4[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_4[(4) + 3]))
                            : "r"(scratch_addr + (partial_offset_3 + 16)));
                        float weight_4 = approx_exp2(maxima_3[0] - safe_merge_max_3);
                        #pragma unroll
                        for (int element_6 = 0; element_6 < 8; element_6++) {
                            numerator_3[element_6] = __fmaf_rn(shard_values_4[element_6], weight_4, numerator_3[element_6]);
                        }
                        denominator_3 = __fmaf_rn(sums_3[0], weight_4, denominator_3);
                        float shard_values_1_3[8] = {0};
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_3[(0) + 3]))
                            : "r"(scratch_addr + (81936 + partial_offset_3)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&shard_values_1_3[(4) + 3]))
                            : "r"(scratch_addr + (81936 + partial_offset_3 + 16)));
                        float weight_2_3 = approx_exp2(maxima_3[1] - safe_merge_max_3);
                        #pragma unroll
                        for (int element_7 = 0; element_7 < 8; element_7++) {
                            numerator_3[element_7] = __fmaf_rn(shard_values_1_3[element_7], weight_2_3, numerator_3[element_7]);
                        }
                        denominator_3 = __fmaf_rn(sums_3[1], weight_2_3, denominator_3);
                        float _rcp_1_1 = __frcp_rn(denominator_3);
                        float inverse_3 = ((denominator_3 > 0.0f) ? _rcp_1_1 : 0.0f);
                        {
                            const float2 _prescale2_2 = {inverse_3, inverse_3};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 4; _ps++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(&numerator_3[0])[_ps], _prescale2_2);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                numerator_3[0 + _ps] *= inverse_3;
                            #endif
                            __half2 _pk[4];
                            _pk[0] = __floats2half2_rn(numerator_3[0 + 0], numerator_3[0 + 1]);
                            _pk[1] = __floats2half2_rn(numerator_3[0 + 2], numerator_3[0 + 3]);
                            _pk[2] = __floats2half2_rn(numerator_3[0 + 4], numerator_3[0 + 5]);
                            _pk[3] = __floats2half2_rn(numerator_3[0 + 6], numerator_3[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__half*)(output + (merge_row_3 * 128 + merge_col_3)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
