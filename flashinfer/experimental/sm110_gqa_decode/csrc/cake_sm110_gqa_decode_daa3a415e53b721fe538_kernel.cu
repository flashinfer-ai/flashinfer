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

#include "cake_sm110_gqa_decode_device_abi.cuh"

#define TMEM_NCOLS 256
#define TMEM_SCORES_OFFSET 0
#define TMEM_OUTPUT_OFFSET 128
#define NUM_KV_PIPE_STAGES 2
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_KV_SMEM_OFF 17408
#define SMEM_KV_SMEM_STAGE_BYTES 16384
#define SMEM_KV_SMEM_STRIDE 16384
#define SMEM_V_SMEM_OFF 17408
#define SMEM_V_SMEM_STAGE_BYTES 16384
#define SMEM_V_SMEM_STRIDE 16384
#define SMEM_SCALE_SMEM_OFF 50176
#define SMEM_SCALE_SMEM_STAGE_BYTES 768
#define SMEM_SCALE_SMEM_STRIDE 768
#define SMEM_TOTAL 50944
#define THREADS 384
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>
#include "cake_sm110_gqa_decode_device_helpers.cuh"
extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_sm110_gqa_decode_daa3a415e53b721fe538(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __half* __restrict__ O, int* __restrict__ sequence_lengths, float softmax_scale_log2)
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
    #define kv_empty_addr (mbar_base + 24)
    #define s_full_addr (mbar_base + 40)
    #define p_full_addr (mbar_base + 48)
    #define corr_sig_addr (mbar_base + 56)
    #define corr_done_addr (mbar_base + 64)
    #define o_full_addr (mbar_base + 72)
    #define tile_done_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __half* q_smem = reinterpret_cast<__half*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __half* kv_smem = reinterpret_cast<__half*>(smem_raw + 17408);
    const int kv_smem_addr = smem + 17408;
    __half* v_smem = reinterpret_cast<__half*>(smem_raw + 17408);
    const int v_smem_addr = smem + 17408;
    float* scale_smem = reinterpret_cast<float*>(smem_raw + 50176);
    const int scale_smem_addr = smem + 50176;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 11 barriers)
    // Mbarriers at smem_raw[0..88)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // kv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // p_full: 1 barriers, init_count=256
            mbarrier_init(smem + 48, 256);
            // corr_sig: 1 barriers, init_count=128
            mbarrier_init(smem + 56, 128);
            // corr_done: 1 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // tile_done: 1 barriers, init_count=128
            mbarrier_init(smem + 80, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 88);
    if (warp == 0) {
        int _tmem_hold = smem + 88;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_output = taddr + 128;
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            int tile = blockIdx.x;
            int batch = tile / 8;
            int kv_head = tile - batch * 8;
            int seqlen_kv = sequence_lengths[batch];
            int num_n_blocks = (seqlen_kv + 64 - 1) / 64;
            const int warp_in_role = warp;
            const int tmem_row_origin = warp_in_role * 32;
            const int logical_row_origin = warp_in_role * 16;
            int my_row = (unsigned int)logical_row_origin + lane % 16;
            int col_half = lane / 16;
            int row_valid = ((my_row < 4) ? 1 : 0);
            float row_max = -CUDART_INF_F;
            float row_sum = 0.0f;
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_corr_done_0 = 0;
            #pragma unroll 1
            for (int n_block = 0; n_block < num_n_blocks; n_block++) {
                mbarrier_wait(s_full_addr, _phase_s_full_0);
                _phase_s_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int valid_cols = seqlen_kv - n_block * 64;
                if (valid_cols > 64) {
                    valid_cols = 64;
                }
                if (row_valid == 0) {
                    valid_cols = 0;
                }
                int score_addr = taddr + (unsigned int)(tmem_row_origin << 16);
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 32;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                    : "r"(score_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int half_valid = valid_cols - col_half * 32;
                if (half_valid < 0) {
                    half_valid = 0;
                }
                if (half_valid > 32) {
                    half_valid = 32;
                }
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
                    uint32_t _thresh_mask_0;
                    {
                        int _lim_1 = half_valid;
                        if (_lim_1 <= 0) { _thresh_mask_0 = 0u; }
                        else if (_lim_1 >= 32) { _thresh_mask_0 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_thresh_mask_0) : "r"(_lim_1));
                        }
                    }
                    if (!(_thresh_mask_0 & (1u << 0))) _tmem_load_0[0] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 1))) _tmem_load_0[1] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 2))) _tmem_load_0[2] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 3))) _tmem_load_0[3] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 4))) _tmem_load_0[4] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 5))) _tmem_load_0[5] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 6))) _tmem_load_0[6] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 7))) _tmem_load_0[7] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 8))) _tmem_load_0[8] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 9))) _tmem_load_0[9] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 10))) _tmem_load_0[10] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 11))) _tmem_load_0[11] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 12))) _tmem_load_0[12] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 13))) _tmem_load_0[13] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 14))) _tmem_load_0[14] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 15))) _tmem_load_0[15] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 16))) _tmem_load_0[16] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 17))) _tmem_load_0[17] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 18))) _tmem_load_0[18] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 19))) _tmem_load_0[19] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 20))) _tmem_load_0[20] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 21))) _tmem_load_0[21] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 22))) _tmem_load_0[22] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 23))) _tmem_load_0[23] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 24))) _tmem_load_0[24] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 25))) _tmem_load_0[25] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 26))) _tmem_load_0[26] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 27))) _tmem_load_0[27] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 28))) _tmem_load_0[28] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 29))) _tmem_load_0[29] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 30))) _tmem_load_0[30] = -CUDART_INF_F;
                    if (!(_thresh_mask_0 & (1u << 31))) _tmem_load_0[31] = -CUDART_INF_F;
                }
                float2 _reg_reduce_max2_2 = {-CAKE_INF, -CAKE_INF};
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[0], _tmem_load_0[1]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[2], _tmem_load_0[3]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[4], _tmem_load_0[5]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[6], _tmem_load_0[7]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[8], _tmem_load_0[9]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[10], _tmem_load_0[11]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[12], _tmem_load_0[13]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[14], _tmem_load_0[15]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[16], _tmem_load_0[17]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[18], _tmem_load_0[19]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[20], _tmem_load_0[21]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[22], _tmem_load_0[23]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[24], _tmem_load_0[25]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[26], _tmem_load_0[27]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[28], _tmem_load_0[29]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[30], _tmem_load_0[31]));
                float tile_max = row_max_reduce(_reg_reduce_max2_2);
                if (half_valid <= 0) {
                    tile_max = -CUDART_INF_F;
                }
                float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, tile_max, 16);
                tile_max = max_noftz(tile_max, _shfl_xor_sync_3);
                float new_max;
                asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(tile_max), "f"(row_max));
                float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                float new_max_scaled = safe_max * softmax_scale_log2;
                float acc_scale_log2 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                float acc_scale;
                float selected_max;
                if (acc_scale_log2 >= -8.0f) {
                    selected_max = row_max;
                    safe_max = ((row_max == -CUDART_INF_F) ? 0.0f : row_max);
                    acc_scale = 1.0f;
                    new_max_scaled = safe_max * softmax_scale_log2;
                } else {
                    selected_max = new_max;
                    float _exp2_4 = approx_exp2(acc_scale_log2);
                    acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_4 : 1.0f);
                }
                row_max = selected_max;
                if (col_half == 0) {
                    scale_smem[my_row] = acc_scale;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(corr_sig_addr);
                float score_bias = ((valid_cols > 0) ? -new_max_scaled : -CUDART_INF_F);
                const float2 _fma_b2_5 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_6 = {score_bias, score_bias};
                float2 _fma_pair_7 = fma_f32x2(make_float2(_tmem_load_0[0], _tmem_load_0[1]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[0] = _fma_pair_7.x;
                _tmem_load_0[1] = _fma_pair_7.y;
                float2 _fma_pair_8 = fma_f32x2(make_float2(_tmem_load_0[2], _tmem_load_0[3]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[2] = _fma_pair_8.x;
                _tmem_load_0[3] = _fma_pair_8.y;
                float2 _fma_pair_9 = fma_f32x2(make_float2(_tmem_load_0[4], _tmem_load_0[5]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[4] = _fma_pair_9.x;
                _tmem_load_0[5] = _fma_pair_9.y;
                float2 _fma_pair_10 = fma_f32x2(make_float2(_tmem_load_0[6], _tmem_load_0[7]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[6] = _fma_pair_10.x;
                _tmem_load_0[7] = _fma_pair_10.y;
                float2 _fma_pair_11 = fma_f32x2(make_float2(_tmem_load_0[8], _tmem_load_0[9]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[8] = _fma_pair_11.x;
                _tmem_load_0[9] = _fma_pair_11.y;
                float2 _fma_pair_12 = fma_f32x2(make_float2(_tmem_load_0[10], _tmem_load_0[11]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[10] = _fma_pair_12.x;
                _tmem_load_0[11] = _fma_pair_12.y;
                float2 _fma_pair_13 = fma_f32x2(make_float2(_tmem_load_0[12], _tmem_load_0[13]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[12] = _fma_pair_13.x;
                _tmem_load_0[13] = _fma_pair_13.y;
                float2 _fma_pair_14 = fma_f32x2(make_float2(_tmem_load_0[14], _tmem_load_0[15]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[14] = _fma_pair_14.x;
                _tmem_load_0[15] = _fma_pair_14.y;
                float2 _fma_pair_15 = fma_f32x2(make_float2(_tmem_load_0[16], _tmem_load_0[17]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[16] = _fma_pair_15.x;
                _tmem_load_0[17] = _fma_pair_15.y;
                float2 _fma_pair_16 = fma_f32x2(make_float2(_tmem_load_0[18], _tmem_load_0[19]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[18] = _fma_pair_16.x;
                _tmem_load_0[19] = _fma_pair_16.y;
                float2 _fma_pair_17 = fma_f32x2(make_float2(_tmem_load_0[20], _tmem_load_0[21]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[20] = _fma_pair_17.x;
                _tmem_load_0[21] = _fma_pair_17.y;
                float2 _fma_pair_18 = fma_f32x2(make_float2(_tmem_load_0[22], _tmem_load_0[23]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[22] = _fma_pair_18.x;
                _tmem_load_0[23] = _fma_pair_18.y;
                float2 _fma_pair_19 = fma_f32x2(make_float2(_tmem_load_0[24], _tmem_load_0[25]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[24] = _fma_pair_19.x;
                _tmem_load_0[25] = _fma_pair_19.y;
                float2 _fma_pair_20 = fma_f32x2(make_float2(_tmem_load_0[26], _tmem_load_0[27]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[26] = _fma_pair_20.x;
                _tmem_load_0[27] = _fma_pair_20.y;
                float2 _fma_pair_21 = fma_f32x2(make_float2(_tmem_load_0[28], _tmem_load_0[29]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[28] = _fma_pair_21.x;
                _tmem_load_0[29] = _fma_pair_21.y;
                float2 _fma_pair_22 = fma_f32x2(make_float2(_tmem_load_0[30], _tmem_load_0[31]), _fma_b2_5, _fma_c2_6);
                _tmem_load_0[30] = _fma_pair_22.x;
                _tmem_load_0[31] = _fma_pair_22.y;
                #pragma unroll
                for (int _le = 0; _le < 32; _le++) {
                    _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                }
                float2 _reg_reduce_sum2_23 = make_float2(0.0f, 0.0f);
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[4], _tmem_load_0[5]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[6], _tmem_load_0[7]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[8], _tmem_load_0[9]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[10], _tmem_load_0[11]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[12], _tmem_load_0[13]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[14], _tmem_load_0[15]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[16], _tmem_load_0[17]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[18], _tmem_load_0[19]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[20], _tmem_load_0[21]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[22], _tmem_load_0[23]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[24], _tmem_load_0[25]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[26], _tmem_load_0[27]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[28], _tmem_load_0[29]));
                _reg_reduce_sum2_23 = add_f32x2(_reg_reduce_sum2_23, make_float2(_tmem_load_0[30], _tmem_load_0[31]));
                float block_half = _reg_reduce_sum2_23.x + _reg_reduce_sum2_23.y;
                float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, block_half, 16);
                float block_sum = block_half + _shfl_xor_sync_24;
                unsigned int packed_p[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    packed_p[_lp] = *(uint32_t*)&_h2;
                }
                int p_addr = taddr + 32 + (unsigned int)(tmem_row_origin << 16);
                asm volatile(
                    "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                    " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[15])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_full_addr);
                mbarrier_wait(corr_done_addr, _phase_corr_done_0);
                _phase_corr_done_0 ^= 1;
                row_sum = __fmaf_rn(row_sum, acc_scale, block_sum);
            }
            if (col_half == 0) {
                scale_smem[64 + my_row] = row_sum;
                scale_smem[128 + my_row] = row_max;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(corr_sig_addr);
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            int tile_1 = blockIdx.x;
            int batch_1 = tile_1 / 8;
            int kv_head_1 = tile_1 - batch_1 * 8;
            int seqlen_kv_1 = sequence_lengths[batch_1];
            int num_n_blocks_1 = (seqlen_kv_1 + 64 - 1) / 64;
            const int warp_in_role_1 = warp - 4;
            const int tmem_row_origin_1 = warp_in_role_1 * 32;
            const int logical_row_origin_1 = warp_in_role_1 * 16;
            int my_row_1 = (unsigned int)logical_row_origin_1 + lane % 16;
            int col_half_1 = lane / 16;
            int row_addr = tmem_row_origin_1 << 16;
            mbarrier_arrive(p_full_addr);
            unsigned int _phase_corr_sig_0 = 0;
            mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
            _phase_corr_sig_0 ^= 1;
            mbarrier_arrive(corr_done_addr);
            #pragma unroll 1
            for (int _n_block = 1; _n_block < num_n_blocks_1; _n_block++) {
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float acc_scale_1 = scale_smem[my_row_1];
                int _warp_vote_0 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                if (_warp_vote_0 != 0) {
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                        : "r"(taddr + 128 + (unsigned int)row_addr));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                        : "r"(taddr + 128 + (unsigned int)row_addr + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_1 = {acc_scale_1, acc_scale_1};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_1);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++) {
                        _tmem_load_1[_ls] = _tmem_load_1[_ls] * acc_scale_1;
                    }
                    #endif
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                        " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                        :: "r"(taddr + 128 + (unsigned int)row_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[63])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_full_addr);
                mbarrier_arrive(corr_done_addr);
            }
            unsigned int _phase_o_full_0 = 0;
            mbarrier_wait(o_full_addr, _phase_o_full_0);
            _phase_o_full_0 ^= 1;
            mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
            _phase_corr_sig_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float final_sum = scale_smem[64 + my_row_1];
            float _rcp_2 = approx_rcp(final_sum);
            float inv_sum = ((final_sum > 0.0f && final_sum == final_sum) ? _rcp_2 : 0.0f);
            float _tmem_load_2[64];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                : "r"(taddr + 128 + (unsigned int)row_addr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[63]))
                : "r"(taddr + 128 + (unsigned int)row_addr + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int col_base = col_half_1 * 64;
            if (my_row_1 < 4) {
                int q_head = kv_head_1 * 4 + my_row_1;
                int output_row = (batch_1 * 32 + q_head) * 128;
                #pragma unroll
                for (int offset = 0; offset < 64; offset += 8) {
                    {
                        const float2 _prescale2_3 = {inv_sum, inv_sum};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(&_tmem_load_2[offset])[_ps], _prescale2_3);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            _tmem_load_2[offset + _ps] *= inv_sum;
                        #endif
                        __half2 _pk[4];
                        _pk[0] = __floats2half2_rn(_tmem_load_2[offset + 0], _tmem_load_2[offset + 1]);
                        _pk[1] = __floats2half2_rn(_tmem_load_2[offset + 2], _tmem_load_2[offset + 3]);
                        _pk[2] = __floats2half2_rn(_tmem_load_2[offset + 4], _tmem_load_2[offset + 5]);
                        _pk[3] = __floats2half2_rn(_tmem_load_2[offset + 6], _tmem_load_2[offset + 7]);
                        *reinterpret_cast<uint4*>(&((__half*)(O + (output_row + col_base + offset)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tile_done_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            int tile_2 = blockIdx.x;
            int batch_2 = tile_2 / 8;
            int kv_head_2 = tile_2 - batch_2 * 8;
            int seqlen_kv_2 = sequence_lengths[batch_2];
            int num_n_blocks_2 = (seqlen_kv_2 + 64 - 1) / 64;
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 0;
            int first_pv = 1;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            unsigned int _phase_p_full_0 = 0;
            #pragma unroll 1
            for (int n_block_1 = 0; n_block_1 < num_n_blocks_2; n_block_1++) {
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int k_stage = kv_stage;
                kv_stage += 1;
                if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
                uint32_t _elected_0 = elect_sync();
                int _smem_desc_lo_1 = ((q_smem_addr) >> 4) & 0x3FFF;
                int _smem_desc_lo_2 = (((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 1024;
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
                    :: "r"(_smem_desc_lo_1), "r"(_smem_desc_lo_2), "r"(tmem_scores), "r"(0), "r"(_elected_0) : "memory");
                tcgen05_commit_leader(s_full_addr, _elected_0);
                tcgen05_commit_leader(kv_empty_addr + (k_stage) * 8, _elected_0);
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                unsigned int v_stage = kv_stage;
                kv_stage += 1;
                if (kv_stage == 2) { kv_stage = 0; kv_phase ^= 1; }
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    int _smem_desc_lo_3 = ((((v_smem_addr) >> 4) & 0x3FFF) | 0x2000000) + (v_stage) * 1024;
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
                    :: "r"(tmem_output), "r"(_smem_desc_lo_3), "r"(tmem_scores + 32), "r"(((first_pv) ? 0 : 1)) : "memory");
                    if (n_block_1 + 1 == num_n_blocks_2) {
                        tcgen05_commit(o_full_addr);
                    }
                    tcgen05_commit(kv_empty_addr + (v_stage) * 8);
                }
                first_pv = 0;
            }
            unsigned int _phase_tile_done_0 = 0;
            mbarrier_wait(tile_done_addr, _phase_tile_done_0);
            _phase_tile_done_0 ^= 1;
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
        }
    }
    // ---- Role: load_warp ----
    if (warp == 9) {
        { // load_warp_main
            int tile_3 = blockIdx.x;
            int batch_3 = tile_3 / 8;
            int kv_head_3 = tile_3 - batch_3 * 8;
            int seqlen_kv_3 = sequence_lengths[batch_3];
            int num_n_blocks_3 = (seqlen_kv_3 + 64 - 1) / 64;
            unsigned int load_stage = 0;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 16384);
                tma_5d_gmem2smem(q_smem_addr, (&Q), 0, 0, 0, kv_head_3, batch_3, q_full_addr);
                tma_5d_gmem2smem(q_smem_addr + 8192, (&Q), 64, 0, 0, kv_head_3, batch_3, q_full_addr);
            }
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (int n_block_2 = 0; n_block_2 < num_n_blocks_3; n_block_2++) {
                int token_base = n_block_2 * 64;
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384, (&K), 0, token_base, kv_head_3, batch_3, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384 + 8192, (&K), 64, token_base, kv_head_3, batch_3, kv_full_addr + (load_stage) * 8);
                }
                load_stage += 1;
                if (load_stage == 2) { load_stage = 0; _phase_kv_empty ^= 1; }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384, (&V), 0, token_base, kv_head_3, batch_3, kv_full_addr + (load_stage) * 8);
                    tma_4d_gmem2smem(kv_smem_addr + load_stage * 16384 + 8192, (&V), 64, token_base, kv_head_3, batch_3, kv_full_addr + (load_stage) * 8);
                }
                load_stage += 1;
                if (load_stage == 2) { load_stage = 0; _phase_kv_empty ^= 1; }
            }
        }
    }
    // ---- Role: empty ----
    if (warp >= 10 && warp <= 11) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
