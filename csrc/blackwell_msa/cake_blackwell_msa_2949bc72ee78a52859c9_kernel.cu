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
#include "cake_blackwell_msa_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 96
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 16
#define TMEM_TMEM_O0_OFFSET 32
#define TMEM_TMEM_O1_OFFSET 48
#define TMEM_TMEM_STATS0_OFFSET 64
#define TMEM_TMEM_STATS1_OFFSET 80
#define NUM_DECODE_KV_STAGES 4
#define SMEM_SMEM_CORR0_OFF 1024
#define SMEM_SMEM_CORR0_STAGE_BYTES 64
#define SMEM_SMEM_CORR0_STRIDE 64
#define SMEM_SMEM_CORR1_OFF 1088
#define SMEM_SMEM_CORR1_STAGE_BYTES 64
#define SMEM_SMEM_CORR1_STRIDE 64
#define SMEM_SMEM_EXCH0_OFF 1152
#define SMEM_SMEM_EXCH0_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_STRIDE 256
#define SMEM_SMEM_EXCH1_OFF 1408
#define SMEM_SMEM_EXCH1_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_STRIDE 256
#define SMEM_SMEM_QT_OFF 1664
#define SMEM_SMEM_QT_STAGE_BYTES 4096
#define SMEM_SMEM_QT_STRIDE 4096
#define SMEM_SMEM_KV_OFF 6144
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 6144
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_KV_FP8_OFF 151168
#define SMEM_SMEM_KV_FP8_STAGE_BYTES 16384
#define SMEM_SMEM_KV_FP8_STRIDE 16384
#define SMEM_SMEM_P0_OFF 137216
#define SMEM_SMEM_P0_STAGE_BYTES 4096
#define SMEM_SMEM_P0_STRIDE 4096
#define SMEM_SMEM_P1_OFF 141312
#define SMEM_SMEM_P1_STAGE_BYTES 4096
#define SMEM_SMEM_P1_STRIDE 4096
#define SMEM_SMEM_PAGE_INDICES_OFF 145408
#define SMEM_SMEM_PAGE_INDICES_STAGE_BYTES 2048
#define SMEM_SMEM_PAGE_INDICES_STRIDE 2048
#define SMEM_DECODE_ROW_MAX_OFF 150144
#define SMEM_DECODE_ROW_MAX_STAGE_BYTES 512
#define SMEM_DECODE_ROW_MAX_STRIDE 512
#define SMEM_DECODE_ROW_SUM_OFF 150656
#define SMEM_DECODE_ROW_SUM_STAGE_BYTES 512
#define SMEM_DECODE_ROW_SUM_STRIDE 512
#define SMEM_TOTAL 216704
#define THREADS 512

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_blackwell_msa_2949bc72ee78a52859c9(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, float* __restrict__ msa_lse, int* __restrict__ kv_indices, int* __restrict__ kv_indptr, int* __restrict__ task_kind, int* __restrict__ task_request, int* __restrict__ task_kv_head, int num_requests, float softmax_scale_log2, int msa_max_pages)
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
    #define q_full_addr (mbar_base + 0)
    #define kv_full_addr (mbar_base + 8)
    #define kv_src_full_addr (mbar_base + 40)
    #define kv_empty_addr (mbar_base + 72)
    #define kv_raw_empty_addr (mbar_base + 104)
    #define s_full_addr (mbar_base + 136)
    #define p_full_addr (mbar_base + 152)
    #define corr_sig_addr (mbar_base + 168)
    #define softmax_done_addr (mbar_base + 184)
    #define corr_done_addr (mbar_base + 200)
    #define o_full_addr (mbar_base + 216)
    #define decode_done_addr (mbar_base + 232)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* smem_corr0 = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_CORR0_OFF);
    const int smem_corr0_addr = smem + SMEM_SMEM_CORR0_OFF;
    float* smem_corr1 = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_CORR1_OFF);
    const int smem_corr1_addr = smem + SMEM_SMEM_CORR1_OFF;
    float* smem_exch0 = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_EXCH0_OFF);
    const int smem_exch0_addr = smem + SMEM_SMEM_EXCH0_OFF;
    float* smem_exch1 = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_EXCH1_OFF);
    const int smem_exch1_addr = smem + SMEM_SMEM_EXCH1_OFF;
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_QT_OFF);
    const int smem_qt_addr = smem + SMEM_SMEM_QT_OFF;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_KV_OFF);
    const int smem_kv_addr = smem + SMEM_SMEM_KV_OFF;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V_OFF);
    const int smem_v_addr = smem + SMEM_SMEM_V_OFF;
    uint8_t* smem_kv_fp8 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_KV_FP8_OFF);
    const int smem_kv_fp8_addr = smem + SMEM_SMEM_KV_FP8_OFF;
    __nv_bfloat16* smem_p0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_P0_OFF);
    const int smem_p0_addr = smem + SMEM_SMEM_P0_OFF;
    __nv_bfloat16* smem_p1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_P1_OFF);
    const int smem_p1_addr = smem + SMEM_SMEM_P1_OFF;
    int* smem_page_indices = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_PAGE_INDICES_OFF);
    const int smem_page_indices_addr = smem + SMEM_SMEM_PAGE_INDICES_OFF;
    float* decode_row_max = reinterpret_cast<float*>(smem_raw + SMEM_DECODE_ROW_MAX_OFF);
    const int decode_row_max_addr = smem + SMEM_DECODE_ROW_MAX_OFF;
    float* decode_row_sum = reinterpret_cast<float*>(smem_raw + SMEM_DECODE_ROW_SUM_OFF);
    const int decode_row_sum_addr = smem + SMEM_DECODE_ROW_SUM_OFF;

    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 30 barriers)
    // Mbarriers at smem_raw[0..240)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'decode_kv' ---
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // kv_src_full: 4 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            // kv_raw_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 152, 256);
            mbarrier_init(smem + 160, 256);
            // corr_sig: 2 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            // softmax_done: 2 barriers, init_count=128
            mbarrier_init(smem + 184, 128);
            mbarrier_init(smem + 192, 128);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            mbarrier_init(smem + 208, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // decode_done: 1 barriers, init_count=128
            mbarrier_init(smem + 232, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 240);
    if (warp == 0) {
        int _tmem_hold = smem + 240;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_s1 = taddr + 16;
    const int tmem_tmem_o0 = taddr + 32;
    const int tmem_tmem_o1 = taddr + 48;
    const int tmem_tmem_stats0 = taddr + 64;
    const int tmem_tmem_stats1 = taddr + 80;

    // ---- Role: softmax ----
    if (warp <= 7) {
        { // softmax_main
            int is_wg1 = ((warp >= 4) ? 1 : 0);
            int group_size = 16;
            const int tmem_row_base_v = warp % 4 * 32;
            int my_tmem_s = taddr + (unsigned int)(((is_wg1 != 0) ? 16 : 0)) + (unsigned int)(tmem_row_base_v << 16);
            int my_tmem_stats = taddr + (unsigned int)(((is_wg1 != 0) ? 80 : 64)) + (unsigned int)(tmem_row_base_v << 16);
            const int warp_in_wg = warp % 4;
            const int wg_tid = warp_in_wg * 32 + lane;
            float* my_exch_ptr = ((is_wg1 != 0) ? (smem_exch1) : (smem_exch0));
            float* my_corr_ptr = ((is_wg1 != 0) ? (smem_corr1) : (smem_corr0));
            unsigned int* base = ((is_wg1 != 0) ? (reinterpret_cast<unsigned int*>(smem_p1)) : (reinterpret_cast<unsigned int*>(smem_p0)));
            int direct_request = 0;
            direct_request = blockIdx.x;
            int kv_tile_begin = 0;
            int kv_tile_end = 16;
            int num_n_blocks = kv_tile_end - kv_tile_begin;
            int num_pairs = num_n_blocks / 2;
            const int row_state_base = warp * 16;
            #pragma unroll
            for (int c = 0; c < 16; c++) {
                decode_row_max[row_state_base + c] = -CAKE_INF;
                decode_row_sum[row_state_base + c] = 0.0f;
            }
            int max_decode_pairs = 8;
            unsigned int _phase_s_full_1 = 0;
            unsigned int _phase_s_full_0 = 0;
            #pragma unroll 1
            for (int pair = 0; pair < max_decode_pairs; pair++) {
                if (num_pairs <= pair) {
                    break;
                }
                if (is_wg1 != 0) {
                    mbarrier_wait(s_full_addr + 8, _phase_s_full_1);
                    _phase_s_full_1 ^= 1;
                } else {
                    mbarrier_wait(s_full_addr, _phase_s_full_0);
                    _phase_s_full_0 ^= 1;
                }
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], my_tmem_s);
                int valid_cols = smem_page_indices[pair * 2 + is_wg1];
                int token_in_block = warp_in_wg * 32 + lane;
                if (token_in_block >= valid_cols) {
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 16; c_1++) {
                        _tmem_load_0[c_1] = -CAKE_INF;
                    }
                }
                float partial_max[16];
                #pragma unroll
                for (int c_2 = 0; c_2 < 16; c_2++) {
                    partial_max[c_2] = _tmem_load_0[c_2];
                }
                #pragma unroll
                for (int c_3 = 0; c_3 < 16; c_3++) {
                    float _warp_reduce_0 = partial_max[c_3];
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                    partial_max[c_3] = _warp_reduce_0;
                }
                if (lane < 16) {
                    my_exch_ptr[warp_in_wg * 16 + lane] = partial_max[lane];
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                float tile_max[16];
                if (lane < 16) {
                    float _max_0 = max_noftz(my_exch_ptr[lane], my_exch_ptr[16 + lane]);
                    float _max_1 = max_noftz(my_exch_ptr[32 + lane], my_exch_ptr[48 + lane]);
                    float _max_2 = max_noftz(_max_0, _max_1);
                    tile_max[lane] = _max_2;
                }
                #pragma unroll
                for (int c_4 = 0; c_4 < 16; c_4++) {
                    float _shfl_0 = __shfl_sync(0xFFFFFFFF, tile_max[c_4], c_4);
                    tile_max[c_4] = _shfl_0;
                }
                float acc_scale[16];
                #pragma unroll
                for (int c_5 = 0; c_5 < 16; c_5++) {
                    float old_max = decode_row_max[row_state_base + c_5];
                    float _max_3 = max_noftz(old_max, tile_max[c_5]);
                    float new_max = _max_3;
                    decode_row_max[row_state_base + c_5] = new_max;
                    float delta = softmax_scale_log2 * (old_max - new_max);
                    float _exp2_0 = approx_exp2(delta);
                    acc_scale[c_5] = ((old_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                }
                tmem_st_x16_f32(my_tmem_stats, acc_scale);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (is_wg1 != 0) {
                    mbarrier_arrive(corr_sig_addr + 8);
                } else {
                    mbarrier_arrive(corr_sig_addr);
                }
                float exp_vals[16];
                #pragma unroll
                for (int c_6 = 0; c_6 < 16; c_6++) {
                    float new_max_1 = decode_row_max[row_state_base + c_6];
                    float safe_max = ((new_max_1 == -CAKE_INF) ? 0.0f : new_max_1);
                    float max_scaled = safe_max * softmax_scale_log2;
                    float _exp2_1 = approx_exp2(_tmem_load_0[c_6] * softmax_scale_log2 - max_scaled);
                    exp_vals[c_6] = _exp2_1;
                }
                float warp_sum[16];
                #pragma unroll
                for (int c_7 = 0; c_7 < 16; c_7++) {
                    warp_sum[c_7] = exp_vals[c_7];
                }
                #pragma unroll
                for (int c_8 = 0; c_8 < 16; c_8++) {
                    float _warp_reduce_1 = warp_sum[c_8];
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                    warp_sum[c_8] = _warp_reduce_1;
                }
                #pragma unroll
                for (int c_9 = 0; c_9 < 16; c_9++) {
                    float old_sum = decode_row_sum[row_state_base + c_9];
                    decode_row_sum[row_state_base + c_9] = old_sum * acc_scale[c_9] + warp_sum[c_9];
                }
                #pragma unroll
                for (int h = 0; h < 16; h++) {
                    {
                        __nv_bfloat16 _bval_0 = __float2bfloat16_rn(exp_vals[h]);
                        uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                        const void* _ptr_0 = reinterpret_cast<const void*>((reinterpret_cast<uint8_t*>(base) + (wg_tid % 64 / 64 * 2048 + (wg_tid / 64 * 16 + h) * 128 + wg_tid % 64 % 64 * 2 ^ (wg_tid % 64 / 64 * 2048 + (wg_tid / 64 * 16 + h) * 128 + wg_tid % 64 % 64 * 2 >> 7 & 7) << 4)));
                        uint64_t _addr64_0;
                        asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_addr64_0) : "l"(_ptr_0));
                        uint32_t _addr_0;
                        asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_addr_0) : "l"(_addr64_0));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                    }
                }
                asm volatile("fence.proxy.async;");
                if (is_wg1 != 0) {
                    mbarrier_arrive(p_full_addr + 8);
                } else {
                    mbarrier_arrive(p_full_addr);
                }
            }
            if (is_wg1 != 0) {
                asm volatile("barrier.sync 9, 128;" ::: "memory");
            } else {
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (lane < 16) {
                my_exch_ptr[warp_in_wg * 16 + lane] = decode_row_sum[row_state_base + lane];
            }
            if (is_wg1 != 0) {
                asm volatile("barrier.sync 9, 128;" ::: "memory");
            } else {
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            float total_sum[16];
            if (lane < 16) {
                total_sum[lane] = my_exch_ptr[lane] + my_exch_ptr[16 + lane] + my_exch_ptr[32 + lane] + my_exch_ptr[48 + lane];
            }
            if (is_wg1 != 0) {
                asm volatile("barrier.sync 9, 128;" ::: "memory");
            } else {
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (warp_in_wg == 0 && lane < 16) {
                my_corr_ptr[lane] = total_sum[lane];
                my_exch_ptr[lane] = decode_row_max[row_state_base + lane];
            }
            if (is_wg1 != 0) {
                asm volatile("barrier.sync 9, 128;" ::: "memory");
            } else {
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (is_wg1 != 0) {
                mbarrier_arrive(softmax_done_addr + 8);
            } else {
                mbarrier_arrive(softmax_done_addr);
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        { // correction_main
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            int d_idx = warp % 4 * 32 + lane;
            int request = 0;
            int kv_head = 0;
            request = blockIdx.x;
            kv_head = blockIdx.y;
            int kv_tile_begin_1 = 0;
            int kv_tile_end_1 = 16;
            int qo_begin = request;
            int num_n_blocks_1 = kv_tile_end_1 - kv_tile_begin_1;
            int group_size_1 = 16;
            int num_pairs_1 = num_n_blocks_1 / 2;
            int max_decode_pairs_1 = 8;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_corr_sig_1 = 0;
            #pragma unroll 1
            for (int pair_1 = 0; pair_1 < max_decode_pairs_1; pair_1++) {
                if (num_pairs_1 <= pair_1) {
                    break;
                }
                mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                _phase_corr_sig_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], taddr + 64 + (unsigned int)corr_row);
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], taddr + 32 + (unsigned int)corr_row);
                #pragma unroll
                for (int h_1 = 0; h_1 < 16; h_1++) {
                    _tmem_load_2[h_1] = _tmem_load_2[h_1] * _tmem_load_1[h_1];
                }
                tmem_st_x16_f32(taddr + 32 + (unsigned int)corr_row, _tmem_load_2);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                mbarrier_wait(corr_sig_addr + 8, _phase_corr_sig_1);
                _phase_corr_sig_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], taddr + 80 + (unsigned int)corr_row);
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], taddr + 48 + (unsigned int)corr_row);
                #pragma unroll
                for (int h_2 = 0; h_2 < 16; h_2++) {
                    _tmem_load_4[h_2] = _tmem_load_4[h_2] * _tmem_load_3[h_2];
                }
                tmem_st_x16_f32(taddr + 48 + (unsigned int)corr_row, _tmem_load_4);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_addr + 8);
            }
            unsigned int _phase_softmax_done_0 = 0;
            mbarrier_wait(softmax_done_addr, _phase_softmax_done_0);
            _phase_softmax_done_0 ^= 1;
            unsigned int _phase_softmax_done_1 = 0;
            mbarrier_wait(softmax_done_addr + 8, _phase_softmax_done_1);
            _phase_softmax_done_1 ^= 1;
            float scale0[16];
            float scale1[16];
            float final_sum[16];
            float final_max[16];
            #pragma unroll
            for (int c_10 = 0; c_10 < 16; c_10++) {
                float _shfl_1 = __shfl_sync(0xFFFFFFFF, smem_exch0[c_10], c_10);
                float _shfl_2 = __shfl_sync(0xFFFFFFFF, smem_exch1[c_10], c_10);
                float _shfl_3 = __shfl_sync(0xFFFFFFFF, smem_corr0[c_10], c_10);
                float _shfl_4 = __shfl_sync(0xFFFFFFFF, smem_corr1[c_10], c_10);
                float _max_4 = max_noftz(_shfl_1, _shfl_2);
                float fm = _max_4;
                final_max[c_10] = fm;
                float d0 = ((_shfl_1 == -CAKE_INF) ? 0.0f : softmax_scale_log2 * (_shfl_1 - fm));
                float d1 = ((_shfl_2 == -CAKE_INF) ? 0.0f : softmax_scale_log2 * (_shfl_2 - fm));
                float _exp2_2 = approx_exp2(d0);
                scale0[c_10] = _exp2_2;
                float _exp2_3 = approx_exp2(d1);
                scale1[c_10] = _exp2_3;
                final_sum[c_10] = _shfl_3 * scale0[c_10] + _shfl_4 * scale1[c_10];
            }
            unsigned int _phase_o_full_0 = 0;
            mbarrier_wait(o_full_addr, _phase_o_full_0);
            _phase_o_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float inv_sum[16];
            #pragma unroll
            for (int h_3 = 0; h_3 < 16; h_3++) {
                float _rcp_0 = approx_rcp(final_sum[h_3]);
                inv_sum[h_3] = ((final_sum[h_3] > 0.0f) ? _rcp_0 : 0.0f);
            }
            float _tmem_load_5[16];
            tmem_ld_x16(&_tmem_load_5[0], taddr + 32 + (unsigned int)corr_row);
            float _tmem_load_6[16];
            tmem_ld_x16(&_tmem_load_6[0], taddr + 48 + (unsigned int)corr_row);
            #pragma unroll
            for (int h_4 = 0; h_4 < 16; h_4++) {
                if (group_size_1 > h_4) {
                    float merged = _tmem_load_5[h_4] * scale0[h_4] + _tmem_load_6[h_4] * scale1[h_4];
                    int q_row = qo_begin * 64 + kv_head * group_size_1 + h_4;
                    int out_idx = q_row * 128 + d_idx;
                    if (d_idx == 0) {
                        float natural_lse = -CAKE_INF;
                        if (final_sum[h_4] > 0.0f) {
                            float _log2_0;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(final_sum[h_4]));
                            natural_lse = final_max[h_4] * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f;
                        }
                        *(reinterpret_cast<float*>(msa_lse + q_row) + (0)) = natural_lse;
                    }
                    *(reinterpret_cast<__nv_bfloat16*>(O + out_idx) + (0)) = __float2bfloat16_rn(merged * inv_sum[h_4]);
                }
            }
            mbarrier_arrive(decode_done_addr);
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            int direct_request_1 = 0;
            direct_request_1 = blockIdx.x;
            int kv_tile_begin_2 = 0;
            int kv_tile_end_2 = 16;
            int num_n_blocks_2 = kv_tile_end_2 - kv_tile_begin_2;
            int num_pairs_2 = num_n_blocks_2 / 2;
            int inst0_stage = 0;
            int first_pv0 = 1;
            int first_pv1 = 1;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            mbarrier_wait(kv_full_addr, 0);
            int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 2048);
            int _mma_b_lo_0 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
            {
                uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134481040, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134481040, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134481040, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134481040, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 122U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134481040, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134481040, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134481040, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134481040, 1);
                }
            }
            elect_commit(s_full_addr);
            elect_commit(kv_empty_addr);
            int max_decode_pairs_2 = 8;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            #pragma unroll 1
            for (int pair_2 = 0; pair_2 < max_decode_pairs_2 - 1; pair_2++) {
                if (pair_2 >= num_pairs_2 - 1) {
                    break;
                }
                int s0 = inst0_stage;
                int s1 = (inst0_stage + 1) % 4;
                int s0_next = (inst0_stage + 2) % 4;
                mbarrier_wait(kv_full_addr + (s1) * 8, 0);
                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1) * 2048);
                int _mma_b_lo_1 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134481040, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134481040, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134481040, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134481040, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 122U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134481040, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134481040, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134481040, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134481040, 1);
                    }
                }
                elect_commit(s_full_addr + 8);
                elect_commit(kv_empty_addr + (s1) * 8);
                mbarrier_wait(kv_full_addr + (s0) * 8, 1);
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0) * 2048);
                int _mma_b_lo_2 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
                {
                    uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                    uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134513808, ((first_pv0) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134513808, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134513808, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134513808, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 122U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134513808, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134513808, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134513808, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134513808, 1);
                    }
                }
                elect_commit(kv_empty_addr + (s0) * 8);
                mbarrier_wait(kv_full_addr + (s0_next) * 8, 0);
                int _mma_a_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s0_next) * 2048);
                {
                    uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                    uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134481040, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134481040, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134481040, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134481040, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 122U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134481040, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134481040, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134481040, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_s0, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134481040, 1);
                    }
                }
                elect_commit(s_full_addr);
                elect_commit(kv_empty_addr + (s0_next) * 8);
                mbarrier_wait(kv_full_addr + (s1) * 8, 1);
                mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                _phase_p_full_1 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1) * 2048);
                int _mma_b_lo_4 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
                {
                    uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                    uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134513808, ((first_pv1) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134513808, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134513808, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134513808, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_4, 122U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134513808, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134513808, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134513808, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_4, 128U);
                    incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134513808, 1);
                    }
                }
                elect_commit(kv_empty_addr + (s1) * 8);
                inst0_stage = s0_next;
                first_pv0 = 0;
                first_pv1 = 0;
            }
            int s0_last = inst0_stage;
            int s1_last = (inst0_stage + 1) % 4;
            mbarrier_wait(kv_full_addr + (s1_last) * 8, 0);
            int _mma_a_lo_5 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1_last) * 2048);
            {
                uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134481040, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134481040, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134481040, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134481040, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 1018U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 122U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134481040, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134481040, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134481040, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_s1, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134481040, 1);
                }
            }
            elect_commit(s_full_addr + 8);
            elect_commit(kv_empty_addr + (s1_last) * 8);
            mbarrier_wait(kv_full_addr + (s0_last) * 8, 1);
            mbarrier_wait(p_full_addr, _phase_p_full_0);
            _phase_p_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_last) * 2048);
            int _mma_b_lo_6 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
            {
                uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_6);
                uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_6);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134513808, ((first_pv0) ? 0 : 1));
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134513808, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134513808, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134513808, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 122U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134513808, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134513808, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134513808, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134513808, 1);
                }
            }
            elect_commit(kv_empty_addr + (s0_last) * 8);
            mbarrier_wait(kv_full_addr + (s1_last) * 8, 1);
            mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
            _phase_p_full_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_last) * 2048);
            int _mma_b_lo_7 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
            {
                uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_7);
                uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134513808, ((first_pv1) ? 0 : 1));
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134513808, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134513808, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134513808, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 122U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134513808, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134513808, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134513808, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 128U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16(tmem_tmem_o1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134513808, 1);
                }
            }
            elect_commit(kv_empty_addr + (s1_last) * 8);
            elect_commit(o_full_addr);
            unsigned int _phase_decode_done_0 = 0;
            mbarrier_wait(decode_done_addr, _phase_decode_done_0);
            _phase_decode_done_0 ^= 1;
        }
    }
    // ---- Role: producer ----
    if (warp == 13) {
        { // producer_main
            int direct_request_2 = 0;
            int kv_head_1 = 0;
            direct_request_2 = blockIdx.x;
            kv_head_1 = blockIdx.y;
            int q_tile = 0;
            int kv_tile_begin_3 = 0;
            int kv_tile_end_3 = 16;
            int qo_begin_1 = direct_request_2;
            int num_n_blocks_3 = kv_tile_end_3 - kv_tile_begin_3;
            int group_size_2 = 16;
            if (elect_sync()) {
                int q_row_1 = qo_begin_1 * 64 + kv_head_1 * group_size_2 + q_tile * 16;
                mbarrier_arrive_expect_tx(q_full_addr, 4096);
                tma_3d_gmem2smem(smem_qt_addr, (&Q), 0, q_row_1, 0, q_full_addr);
            }
            int native_num_n_blocks = num_n_blocks_3;
            if (elect_sync()) {
                int native_kv_stage = 0;
                int native_kv_phase = 1;
                int native_prefill = ((native_num_n_blocks < 4) ? native_num_n_blocks : 4);
                #pragma unroll
                for (int native_ni = 0; native_ni < 4; native_ni++) {
                    if (native_prefill <= native_ni) {
                        break;
                    }
                    int native_n_block = kv_tile_end_3 - 1 - native_ni;
                    int native_pg0 = 0;
                    int native_pg1 = 0;
                    int msa_token_base = 0;
                    int msa_page_head = 0;
                    int msa_valid_cols = 128;
                    int batch = direct_request_2;
                    int query_in_batch = direct_request_2 - batch;
                    int selected_block = task_kind[(kv_head_1 * num_requests + direct_request_2) * 16 + native_n_block];
                    int kv_len = task_kv_head[batch];
                    {
                        kv_len = kv_indptr[batch + 1] - kv_indptr[batch];
                    }
                    int valid_cols_1 = 0;
                    if (selected_block >= 0) {
                        int block_start = selected_block * 128;
                        valid_cols_1 = kv_len - block_start;
                        if (valid_cols_1 > 128) {
                            valid_cols_1 = 128;
                        }
                        if (valid_cols_1 < 0) {
                            valid_cols_1 = 0;
                        }
                        {
                            int query_position = kv_len - 1 + query_in_batch;
                            int causal_cols = query_position - block_start + 1;
                            if (valid_cols_1 > causal_cols) {
                                valid_cols_1 = causal_cols;
                            }
                            if (valid_cols_1 < 0) {
                                valid_cols_1 = 0;
                            }
                        }
                    }
                    int token_base = 0;
                    int page_head = 0;
                    {
                        int safe_block = ((selected_block >= 0) ? selected_block : 0);
                        token_base = kv_indptr[batch] + safe_block * 128;
                        page_head = kv_head_1;
                    }
                    msa_token_base = token_base;
                    msa_page_head = page_head;
                    msa_valid_cols = valid_cols_1;
                    smem_page_indices[native_ni] = msa_valid_cols;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_wait(kv_raw_empty_addr + (native_kv_stage) * 8, native_kv_phase);
                    mbarrier_arrive_expect_tx(kv_src_full_addr + (native_kv_stage) * 8, 16384);
                    int native_dst = smem_kv_addr + (unsigned int)(native_kv_stage * 32768);
                    int token0 = msa_token_base;
                    int token1 = msa_token_base + 64;
                    tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_kv_stage * 16384), (&K), 0, token0, msa_page_head, kv_src_full_addr + (native_kv_stage) * 8);
                    tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_kv_stage * 16384) + 8192, (&K), 0, token1, msa_page_head, kv_src_full_addr + (native_kv_stage) * 8);
                    native_kv_stage += 1;
                    if (native_kv_stage == 4) { native_kv_stage = 0; native_kv_phase ^= 1; }
                }
                #pragma unroll 1
                for (int native_ni_1 = 0; native_ni_1 < 16; native_ni_1++) {
                    if (native_num_n_blocks <= native_ni_1) {
                        break;
                    }
                    int native_stage = native_ni_1 % 4;
                    int native_n_block_1 = kv_tile_end_3 - 1 - native_ni_1;
                    int native_pg0_1 = 0;
                    int native_pg1_1 = 0;
                    int msa_token_base_1 = 0;
                    int msa_page_head_1 = 0;
                    int msa_valid_cols_1 = 128;
                    int batch_1 = direct_request_2;
                    int query_in_batch_1 = direct_request_2 - batch_1;
                    int selected_block_1 = task_kind[(kv_head_1 * num_requests + direct_request_2) * 16 + native_n_block_1];
                    int kv_len_1 = task_kv_head[batch_1];
                    {
                        kv_len_1 = kv_indptr[batch_1 + 1] - kv_indptr[batch_1];
                    }
                    int valid_cols_2 = 0;
                    if (selected_block_1 >= 0) {
                        int block_start_1 = selected_block_1 * 128;
                        valid_cols_2 = kv_len_1 - block_start_1;
                        if (valid_cols_2 > 128) {
                            valid_cols_2 = 128;
                        }
                        if (valid_cols_2 < 0) {
                            valid_cols_2 = 0;
                        }
                        {
                            int query_position_1 = kv_len_1 - 1 + query_in_batch_1;
                            int causal_cols_1 = query_position_1 - block_start_1 + 1;
                            if (valid_cols_2 > causal_cols_1) {
                                valid_cols_2 = causal_cols_1;
                            }
                            if (valid_cols_2 < 0) {
                                valid_cols_2 = 0;
                            }
                        }
                    }
                    int token_base_1 = 0;
                    int page_head_1 = 0;
                    {
                        int safe_block_1 = ((selected_block_1 >= 0) ? selected_block_1 : 0);
                        token_base_1 = kv_indptr[batch_1] + safe_block_1 * 128;
                        page_head_1 = kv_head_1;
                    }
                    msa_token_base_1 = token_base_1;
                    msa_page_head_1 = page_head_1;
                    msa_valid_cols_1 = valid_cols_2;
                    mbarrier_wait(kv_raw_empty_addr + (native_stage) * 8, 0);
                    mbarrier_arrive_expect_tx(kv_src_full_addr + (native_stage) * 8, 16384);
                    int native_dst_1 = smem_kv_addr + (unsigned int)(native_stage * 32768);
                    int token0_1 = msa_token_base_1;
                    int token1_1 = msa_token_base_1 + 64;
                    tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_stage * 16384), (&V), 0, token0_1, msa_page_head_1, kv_src_full_addr + (native_stage) * 8);
                    tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_stage * 16384) + 8192, (&V), 0, token1_1, msa_page_head_1, kv_src_full_addr + (native_stage) * 8);
                    int native_next_ni = native_ni_1 + 4;
                    if (native_next_ni < native_num_n_blocks) {
                        int native_next_n = kv_tile_end_3 - 1 - native_next_ni;
                        int native_npg0 = 0;
                        int native_npg1 = 0;
                        int msa_next_token_base = 0;
                        int msa_next_page_head = 0;
                        int msa_next_valid_cols = 128;
                        int batch_0 = direct_request_2;
                        int query_in_batch_1_1 = direct_request_2 - batch_0;
                        int selected_block_2 = task_kind[(kv_head_1 * num_requests + direct_request_2) * 16 + native_next_n];
                        int kv_len_3 = task_kv_head[batch_0];
                        {
                            kv_len_3 = kv_indptr[batch_0 + 1] - kv_indptr[batch_0];
                        }
                        int valid_cols_4 = 0;
                        if (selected_block_2 >= 0) {
                            int block_start_2 = selected_block_2 * 128;
                            valid_cols_4 = kv_len_3 - block_start_2;
                            if (valid_cols_4 > 128) {
                                valid_cols_4 = 128;
                            }
                            if (valid_cols_4 < 0) {
                                valid_cols_4 = 0;
                            }
                            {
                                int query_position_2 = kv_len_3 - 1 + query_in_batch_1_1;
                                int causal_cols_2 = query_position_2 - block_start_2 + 1;
                                if (valid_cols_4 > causal_cols_2) {
                                    valid_cols_4 = causal_cols_2;
                                }
                                if (valid_cols_4 < 0) {
                                    valid_cols_4 = 0;
                                }
                            }
                        }
                        int token_base_5 = 0;
                        int page_head_6 = 0;
                        {
                            int safe_block_2 = ((selected_block_2 >= 0) ? selected_block_2 : 0);
                            token_base_5 = kv_indptr[batch_0] + safe_block_2 * 128;
                            page_head_6 = kv_head_1;
                        }
                        msa_next_token_base = token_base_5;
                        msa_next_page_head = page_head_6;
                        msa_next_valid_cols = valid_cols_4;
                        smem_page_indices[native_next_ni] = msa_next_valid_cols;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_wait(kv_raw_empty_addr + (native_stage) * 8, 1);
                        mbarrier_arrive_expect_tx(kv_src_full_addr + (native_stage) * 8, 16384);
                        int native_kdst = smem_kv_addr + (unsigned int)(native_stage * 32768);
                        int token0_7 = msa_next_token_base;
                        int token1_8 = msa_next_token_base + 64;
                        tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_stage * 16384), (&K), 0, token0_7, msa_next_page_head, kv_src_full_addr + (native_stage) * 8);
                        tma_3d_gmem2smem(smem_kv_fp8_addr + (unsigned int)(native_stage * 16384) + 8192, (&K), 0, token1_8, msa_next_page_head, kv_src_full_addr + (native_stage) * 8);
                    }
                }
            }
        }
    }
    // ---- Role: producer_aux ----
    if (warp >= 14 && warp <= 15) {
        { // producer_aux_main
            int num_n_blocks_4 = 16;
            int prefill = ((num_n_blocks_4 < 4) ? num_n_blocks_4 : 4);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                if (prefill <= ni) {
                    break;
                }
                mbarrier_wait(kv_src_full_addr + (ni) * 8, 0);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_wait(kv_empty_addr + (ni) * 8, 1);
                {
                    const char* _src_ptr = smem_raw + (smem_kv_fp8_addr + (unsigned int)(ni * 16384) - smem);
                    char* _dst_ptr = smem_raw + (smem_kv_addr + (unsigned int)(ni * 32768) - smem);
                    const int _tid = (int)threadIdx.x - (14) * 32;
                    uint64_t _src_buf[16];
                    #pragma unroll
                    for (int _outer = 0; _outer < 2; ++_outer) {
                        #pragma unroll
                        for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 64) {
                            int _off = _base + _tid;
                            _src_buf[(_base / 64) % 16] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                        }
                        #pragma unroll
                        for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 64) {
                            int _off = _base + _tid;
                            uint64_t _src64 = _src_buf[(_base / 64) % 16];
                            uint32_t _out_x16x2[4];
                            #pragma unroll
                            for (int _cv = 0; _cv < 4; ++_cv) {
                                uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                _out_x16x2[_cv] = cvt_bf16x2_e4m3x2_v(_e4m3x2);
                            }
                            uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                            int _elt = _off * 8;
                            int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                            int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                            int _swz_off = _byte_off ^ ((_row % 8) * 16);
                            *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile("barrier.sync 10, 64;" ::: "memory");
                if (warp == 14) {
                    if (elect_sync()) {
                        mbarrier_arrive(kv_full_addr + (ni) * 8);
                        mbarrier_arrive(kv_raw_empty_addr + (ni) * 8);
                    }
                }
            }
            #pragma unroll 1
            for (int ni_1 = 0; ni_1 < 16; ni_1++) {
                if (num_n_blocks_4 <= ni_1) {
                    break;
                }
                int stage = ni_1 % 4;
                mbarrier_wait(kv_src_full_addr + (stage) * 8, 1);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_wait(kv_empty_addr + (stage) * 8, 0);
                {
                    const char* _src_ptr = smem_raw + (smem_kv_fp8_addr + (unsigned int)(stage * 16384) - smem);
                    char* _dst_ptr = smem_raw + (smem_kv_addr + (unsigned int)(stage * 32768) - smem);
                    const int _tid = (int)threadIdx.x - (14) * 32;
                    uint64_t _src_buf[16];
                    #pragma unroll
                    for (int _outer = 0; _outer < 2; ++_outer) {
                        #pragma unroll
                        for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 64) {
                            int _off = _base + _tid;
                            _src_buf[(_base / 64) % 16] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                        }
                        #pragma unroll
                        for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 64) {
                            int _off = _base + _tid;
                            uint64_t _src64 = _src_buf[(_base / 64) % 16];
                            uint32_t _out_x16x2[4];
                            #pragma unroll
                            for (int _cv = 0; _cv < 4; ++_cv) {
                                uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                _out_x16x2[_cv] = cvt_bf16x2_e4m3x2_v(_e4m3x2);
                            }
                            uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                            int _elt = _off * 8;
                            int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                            int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                            int _swz_off = _byte_off ^ ((_row % 8) * 16);
                            *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                asm volatile("barrier.sync 10, 64;" ::: "memory");
                if (warp == 14) {
                    if (elect_sync()) {
                        mbarrier_arrive(kv_full_addr + (stage) * 8);
                        mbarrier_arrive(kv_raw_empty_addr + (stage) * 8);
                    }
                }
                int next_ni = ni_1 + 4;
                if (next_ni < num_n_blocks_4) {
                    mbarrier_wait(kv_src_full_addr + (stage) * 8, 0);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_wait(kv_empty_addr + (stage) * 8, 1);
                    {
                        const char* _src_ptr = smem_raw + (smem_kv_fp8_addr + (unsigned int)(stage * 16384) - smem);
                        char* _dst_ptr = smem_raw + (smem_kv_addr + (unsigned int)(stage * 32768) - smem);
                        const int _tid = (int)threadIdx.x - (14) * 32;
                        uint64_t _src_buf[16];
                        #pragma unroll
                        for (int _outer = 0; _outer < 2; ++_outer) {
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 64) {
                                int _off = _base + _tid;
                                _src_buf[(_base / 64) % 16] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                            }
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 64) {
                                int _off = _base + _tid;
                                uint64_t _src64 = _src_buf[(_base / 64) % 16];
                                uint32_t _out_x16x2[4];
                                #pragma unroll
                                for (int _cv = 0; _cv < 4; ++_cv) {
                                    uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                    _out_x16x2[_cv] = cvt_bf16x2_e4m3x2_v(_e4m3x2);
                                }
                                uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                                int _elt = _off * 8;
                                int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                                int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                                int _swz_off = _byte_off ^ ((_row % 8) * 16);
                                *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    asm volatile("barrier.sync 10, 64;" ::: "memory");
                    if (warp == 14) {
                        if (elect_sync()) {
                            mbarrier_arrive(kv_full_addr + (stage) * 8);
                            mbarrier_arrive(kv_raw_empty_addr + (stage) * 8);
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
