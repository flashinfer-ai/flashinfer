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

#define TMEM_NCOLS 128
#define TMEM_OUTPUT_OFFSET 0
#define TMEM_SCORES_OFFSET 64
#define NUM_KV_PIPE_STAGES 3
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_KV_SMEM_OFF 17408
#define SMEM_KV_SMEM_STAGE_BYTES 8192
#define SMEM_KV_SMEM_STRIDE 8192
#define SMEM_V_LOW_OFF 17408
#define SMEM_V_LOW_STAGE_BYTES 4096
#define SMEM_V_LOW_STRIDE 8192
#define SMEM_V_HIGH_OFF 21504
#define SMEM_V_HIGH_STAGE_BYTES 4096
#define SMEM_V_HIGH_STRIDE 8192
#define SMEM_TOTAL 41984
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>
#include "cake_sm110_gqa_decode_device_helpers.cuh"
extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_sm110_gqa_decode_d8f05d2bfbbde5459902(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, float* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, unsigned int* __restrict__ completed, __half* __restrict__ O, int* __restrict__ sequence_lengths, float softmax_scale_log2)
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
    #define kv_full_addr (mbar_base + 0)
    #define kv_empty_addr (mbar_base + 24)
    #define s_full_addr (mbar_base + 48)
    #define padding_ready_addr (mbar_base + 56)
    #define p_full_addr (mbar_base + 64)
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
    __half* v_low = reinterpret_cast<__half*>(smem_raw + 17408);
    const int v_low_addr = smem + 17408;
    __half* v_high = reinterpret_cast<__half*>(smem_raw + 21504);
    const int v_high_addr = smem + 21504;

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 11 barriers)
    // Mbarriers at smem_raw[0..88)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // kv_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // padding_ready: 1 barriers, init_count=96
            mbarrier_init(smem + 56, 96);
            // p_full: 1 barriers, init_count=32
            mbarrier_init(smem + 64, 32);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // tile_done: 1 barriers, init_count=32
            mbarrier_init(smem + 80, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel pre-TMEM ops
    if (warp == 2) {
        if (elect_sync()) {
            int early_tile = (int)blockIdx.y;
            int early_batch = early_tile / 8;
            int early_head = early_tile - early_batch * 8;
            mbarrier_arrive_expect_tx(kv_full_addr, 24576);
            tma_4d_gmem2smem(kv_smem_addr, (&K), 0, 0, early_head, early_batch, kv_full_addr);
            tma_4d_gmem2smem(kv_smem_addr + 4096, (&K), 64, 0, early_head, early_batch, kv_full_addr);
            tma_5d_gmem2smem(q_smem_addr, (&Q), 0, 0, 0, early_head, early_batch, kv_full_addr);
            tma_5d_gmem2smem(q_smem_addr + 8192, (&Q), 64, 0, 0, early_head, early_batch, kv_full_addr);
            mbarrier_arrive_expect_tx(kv_full_addr + 8, 8192);
            tma_4d_gmem2smem(kv_smem_addr + 8192, (&V), 0, 0, early_head, early_batch, kv_full_addr + 8);
            tma_4d_gmem2smem(kv_smem_addr + 8192 + 4096, (&V), 64, 0, early_head, early_batch, kv_full_addr + 8);
        }
    }

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 88);
    if (warp == 0) {
        int _tmem_hold = smem + 88;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_output = taddr;
    const int tmem_scores = taddr + 64;

    // ---- Role: softmax ----
    if (warp == 0) {
        { // softmax_main
            int tile = (int)blockIdx.y;
            int batch = tile / 8;
            int kv_head = tile - batch * 8;
            int seqlen_kv = sequence_lengths[batch];
            int total_blocks = (seqlen_kv + 32 - 1) / 32;
            int first_block = 0;
            int num_n_blocks = total_blocks;
            if (num_n_blocks < 1) {
                num_n_blocks = 1;
                first_block = total_blocks;
            }
            const int warp_in_role = warp;
            const int tmem_row_origin = warp_in_role * 32;
            const int logical_row_origin = warp_in_role * 16;
            int my_row = (unsigned int)logical_row_origin + lane % 16;
            int col_half = lane / 16;
            int row_valid = ((my_row < 4) ? 1 : 0);
            float row_max = -CUDART_INF_F;
            float row_sum = 0.0f;
            unsigned int _phase_s_full_0 = 0;
            #pragma unroll 1
            for (int n_block = 0; n_block < num_n_blocks; n_block++) {
                mbarrier_wait(s_full_addr, _phase_s_full_0);
                _phase_s_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int valid_cols = seqlen_kv - (first_block + n_block) * 32;
                if (valid_cols > 32) {
                    valid_cols = 32;
                }
                if (row_valid == 0) {
                    valid_cols = 0;
                }
                int score_addr = taddr + 64 + (unsigned int)(tmem_row_origin << 16);
                float _tmem_load_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                    : "r"(score_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int half_valid = valid_cols - col_half * 16;
                if (half_valid < 0) {
                    half_valid = 0;
                }
                if (half_valid > 16) {
                    half_valid = 16;
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
                } else if (half_valid < 16) {
                    uint32_t _thresh_mask_0;
                    {
                        int _lim_1 = half_valid;
                        if (_lim_1 <= 0) { _thresh_mask_0 = 0u; }
                        else if (_lim_1 >= 16) { _thresh_mask_0 = ((1u << 16) - 1u); }
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
                }
                float tile_max = _tmem_load_0[0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    tile_max = max_noftz(tile_max, _tmem_load_0[_lr]);
                }
                if (half_valid <= 0) {
                    tile_max = -CUDART_INF_F;
                }
                float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, tile_max, 16);
                tile_max = max_noftz(tile_max, _shfl_xor_sync_2);
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
                    float _exp2_3 = approx_exp2(acc_scale_log2);
                    acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_3 : 1.0f);
                }
                row_max = selected_max;
                float score_bias = ((valid_cols > 0) ? -new_max_scaled : -CUDART_INF_F);
                const float2 _fma_b2_4 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_5 = {score_bias, score_bias};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_4, _fma_c2_5);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                }
                float block_half = 0.0f;
                #pragma unroll
                for (int _lr = 0; _lr < 16; _lr++) {
                    block_half += _tmem_load_0[_lr];
                }
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, block_half, 16);
                float block_sum = block_half + _shfl_xor_sync_6;
                unsigned int packed_p[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    packed_p[_lp] = *(uint32_t*)&_h2;
                }
                int p_addr = taddr + 96 + (unsigned int)(tmem_row_origin << 16);
                asm volatile(
                    "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                    " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[7])));
                asm volatile(
                    "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                    " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(taddr + 96 + (unsigned int)(tmem_row_origin + 16 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[7])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale < 1.0f);
                if (n_block > 0 && _vote_0 != 0) {
                    #pragma unroll
                    for (int output_half = 0; output_half < 2; output_half++) {
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 32;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                            : "r"(taddr + (unsigned int)(tmem_row_origin + output_half * 16 << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #if __CUDA_ARCH__ >= 1000
                        const float2 _scale2_7 = {acc_scale, acc_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_7);
                        #else
                        #pragma unroll
                        for (int _ls = 0; _ls < 32; _ls++) {
                            _tmem_load_1[_ls] = _tmem_load_1[_ls] * acc_scale;
                        }
                        #endif
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x32bx2.x32.b32"
                            " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(taddr + (unsigned int)(tmem_row_origin + output_half * 16 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[31])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                row_sum = __fmaf_rn(row_sum, acc_scale, block_sum);
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_full_addr);
            }
            unsigned int _phase_o_full_0 = 0;
            mbarrier_wait(o_full_addr, _phase_o_full_0);
            _phase_o_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int q_head = kv_head * 4 + my_row;
            float _rcp_8 = approx_rcp(row_sum);
            float inv_sum = ((row_sum > 0.0f && row_sum == row_sum) ? _rcp_8 : 0.0f);
            int output_row = (batch * 32 + q_head) * 128;
            #pragma unroll
            for (int output_half_1 = 0; output_half_1 < 2; output_half_1++) {
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 32;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                    : "r"(taddr + (unsigned int)(tmem_row_origin + output_half_1 * 16 << 16)));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col_base = output_half_1 * 64 + col_half * 32;
                if (my_row < 4) {
                    #pragma unroll
                    for (int offset = 0; offset < 32; offset += 8) {
                        {
                            const float2 _prescale2_9 = {inv_sum, inv_sum};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 4; _ps++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(&_tmem_load_2[offset])[_ps], _prescale2_9);
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
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tile_done_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 1) {
        { // mma_warp_main
            int tile_1 = (int)blockIdx.y;
            int batch_1 = tile_1 / 8;
            int kv_head_1 = tile_1 - batch_1 * 8;
            int seqlen_kv_1 = sequence_lengths[batch_1];
            int total_blocks_1 = (seqlen_kv_1 + 32 - 1) / 32;
            int first_block_1 = 0;
            int num_n_blocks_1 = total_blocks_1;
            if (num_n_blocks_1 < 1) {
                num_n_blocks_1 = 1;
                first_block_1 = total_blocks_1;
            }
            unsigned int inactive_p[8];
            inactive_p[0] = 0;
            inactive_p[1] = 0;
            inactive_p[2] = 0;
            inactive_p[3] = 0;
            inactive_p[4] = 0;
            inactive_p[5] = 0;
            inactive_p[6] = 0;
            inactive_p[7] = 0;
            const int inactive_row_origin = (int)warp * 32;
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(taddr + 96 + (unsigned int)(inactive_row_origin << 16)), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[7])));
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(taddr + 96 + (unsigned int)(inactive_row_origin + 16 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[7])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(padding_ready_addr);
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 0;
            int first_pv = 1;
            unsigned int _phase_padding_ready_0 = 0;
            mbarrier_wait(padding_ready_addr, _phase_padding_ready_0);
            _phase_padding_ready_0 ^= 1;
            unsigned int _phase_p_full_0 = 0;
            #pragma unroll 1
            for (int n_block_1 = 0; n_block_1 < num_n_blocks_1; n_block_1++) {
                mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int k_stage = kv_stage;
                kv_stage += 1;
                if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                uint32_t _elected_0 = elect_sync();
                int _smem_desc_lo_1 = ((q_smem_addr) >> 4) & 0x3FFF;
                int _smem_desc_lo_2 = (((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 512;
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
                    "mov.b32 id, 67633168;\n\t"
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
                    "add.u32 blo, blo, 250;\n\t"
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
                if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    int _smem_desc_lo_3 = ((((v_low_addr) >> 4) & 0x3FFF) | 0x1000000) + (v_stage) * 512;
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
                    "mov.b32 id, 68222992;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_smem_desc_lo_3), "r"(tmem_scores + 32), "r"(((first_pv) ? 0 : 1)) : "memory");
                    int _smem_desc_lo_4 = ((((v_high_addr) >> 4) & 0x3FFF) | 0x1000000) + (v_stage) * 512;
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
                    "mov.b32 id, 68222992;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_output + (1048576))), "r"(_smem_desc_lo_4), "r"(tmem_scores + 1048608), "r"(((first_pv) ? 0 : 1)) : "memory");
                    if (n_block_1 + 1 == num_n_blocks_1) {
                        tcgen05_commit(o_full_addr);
                    }
                    tcgen05_commit(kv_empty_addr + (v_stage) * 8);
                }
                first_pv = 0;
            }
            unsigned int _phase_tile_done_0 = 0;
            mbarrier_wait(tile_done_addr, _phase_tile_done_0);
            _phase_tile_done_0 ^= 1;
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        }
    }
    // ---- Role: load_warp ----
    if (warp == 2) {
        { // load_warp_main
            int tile_2 = (int)blockIdx.y;
            int batch_2 = tile_2 / 8;
            int kv_head_2 = tile_2 - batch_2 * 8;
            int seqlen_kv_2 = sequence_lengths[batch_2];
            int total_blocks_2 = (seqlen_kv_2 + 32 - 1) / 32;
            int first_block_2 = 0;
            int num_n_blocks_2 = total_blocks_2;
            if (num_n_blocks_2 < 1) {
                num_n_blocks_2 = 1;
                first_block_2 = total_blocks_2;
            }
            unsigned int inactive_p_1[8];
            inactive_p_1[0] = 0;
            inactive_p_1[1] = 0;
            inactive_p_1[2] = 0;
            inactive_p_1[3] = 0;
            inactive_p_1[4] = 0;
            inactive_p_1[5] = 0;
            inactive_p_1[6] = 0;
            inactive_p_1[7] = 0;
            const int inactive_row_origin_1 = (int)warp * 32;
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(taddr + 96 + (unsigned int)(inactive_row_origin_1 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[7])));
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(taddr + 96 + (unsigned int)(inactive_row_origin_1 + 16 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[7])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(padding_ready_addr);
            unsigned int load_stage = 0;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (int n_block_2 = 0; n_block_2 < num_n_blocks_2; n_block_2++) {
                int block = first_block_2 + n_block_2;
                int token_base = block * 32;
                int hinted = ((block >= total_blocks_2 - 32 && block < total_blocks_2) ? 1 : 0);
                int issue = 1;
                issue = ((n_block_2 > 0) ? 1 : 0);
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (issue != 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 8192);
                        if (hinted != 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(kv_smem_addr + load_stage * 8192), "l"((&K)), "r"(0), "r"(token_base), "r"(kv_head_2), "r"(batch_2),
                                   "r"(kv_full_addr + (load_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(kv_smem_addr + load_stage * 8192 + 4096), "l"((&K)), "r"(64), "r"(token_base), "r"(kv_head_2), "r"(batch_2),
                                   "r"(kv_full_addr + (load_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 8192, (&K), 0, token_base, kv_head_2, batch_2, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 8192 + 4096, (&K), 64, token_base, kv_head_2, batch_2, kv_full_addr + (load_stage) * 8);
                        }
                    }
                }
                load_stage += 1;
                if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (issue != 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 8192);
                        if (hinted != 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(kv_smem_addr + load_stage * 8192), "l"((&V)), "r"(0), "r"(token_base), "r"(kv_head_2), "r"(batch_2),
                                   "r"(kv_full_addr + (load_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(kv_smem_addr + load_stage * 8192 + 4096), "l"((&V)), "r"(64), "r"(token_base), "r"(kv_head_2), "r"(batch_2),
                                   "r"(kv_full_addr + (load_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 8192, (&V), 0, token_base, kv_head_2, batch_2, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 8192 + 4096, (&V), 64, token_base, kv_head_2, batch_2, kv_full_addr + (load_stage) * 8);
                        }
                    }
                }
                load_stage += 1;
                if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
            }
        }
    }
    // ---- Role: aux_warp ----
    if (warp == 3) {
        { // aux_warp_main
            int tile_3 = (int)blockIdx.y;
            int batch_3 = tile_3 / 8;
            int kv_head_3 = tile_3 - batch_3 * 8;
            int seqlen_kv_3 = sequence_lengths[batch_3];
            int total_blocks_3 = (seqlen_kv_3 + 32 - 1) / 32;
            int first_block_3 = 0;
            int num_n_blocks_3 = total_blocks_3;
            if (num_n_blocks_3 < 1) {
                num_n_blocks_3 = 1;
                first_block_3 = total_blocks_3;
            }
            unsigned int inactive_p_2[8];
            inactive_p_2[0] = 0;
            inactive_p_2[1] = 0;
            inactive_p_2[2] = 0;
            inactive_p_2[3] = 0;
            inactive_p_2[4] = 0;
            inactive_p_2[5] = 0;
            inactive_p_2[6] = 0;
            inactive_p_2[7] = 0;
            const int inactive_row_origin_2 = (int)warp * 32;
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(taddr + 96 + (unsigned int)(inactive_row_origin_2 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[7])));
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(taddr + 96 + (unsigned int)(inactive_row_origin_2 + 16 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[7])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(padding_ready_addr);
        }
    }

    // Cleanup
}

} // extern "C"
