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
#include "cake_dsv4_nvfp4_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsv4_nvfp4_d6fad161fd02521b6d50(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ lse_out, int num_heads, int num_splits, int heads_per_cta, float lse_scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y * heads_per_cta + warp;
    if (warp < heads_per_cta && head_idx < num_heads) {
        int stat_base = (query_idx * num_heads + head_idx) * num_splits;
        int last_split = num_splits - 1;
        float local_lse = -CAKE_INF;
        if (lane < num_splits) {
            local_lse = partial_lse[stat_base + lane];
        }
        float _warp_reduce_0 = local_lse;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
        float global_max = _warp_reduce_0;
        float local_weight = 0.0f;
        if (lane < num_splits && local_lse != -CAKE_INF) {
            float _exp2_0 = approx_exp2(local_lse - global_max);
            local_weight = _exp2_0;
        }
        float _warp_reduce_1 = local_weight;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        float global_sum = _warp_reduce_1;
        float normalized_weight = 0.0f;
        if (global_sum > 0.0f) {
            float _rcp_0 = approx_rcp(global_sum);
            normalized_weight = local_weight * _rcp_0;
        }
        int partial_base = stat_base * 512 + lane * 8;
        float acc_lo[8];
        float acc_hi[8];
        acc_lo[0] = 0.0f;
        acc_hi[0] = 0.0f;
        acc_lo[1] = 0.0f;
        acc_hi[1] = 0.0f;
        acc_lo[2] = 0.0f;
        acc_hi[2] = 0.0f;
        acc_lo[3] = 0.0f;
        acc_hi[3] = 0.0f;
        acc_lo[4] = 0.0f;
        acc_hi[4] = 0.0f;
        acc_lo[5] = 0.0f;
        acc_hi[5] = 0.0f;
        acc_lo[6] = 0.0f;
        acc_hi[6] = 0.0f;
        acc_lo[7] = 0.0f;
        acc_hi[7] = 0.0f;
        for (int split_group = 0; split_group < num_splits; split_group += 4) {
            int _min_0 = ((split_group) < (last_split) ? (split_group) : (last_split));
            int split_k = _min_0;
            int row_k = partial_base + split_k * 512;
            float _vec_load_0[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_O + row_k + 0);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_1[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(partial_O + (row_k + 256) + 0);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                    }
                }
            }
            int _min_1 = ((split_group + 1) < (last_split) ? (split_group + 1) : (last_split));
            int split_k_0 = _min_1;
            int row_k_1 = partial_base + split_k_0 * 512;
            float _vec_load_2[8];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(partial_O + row_k_1 + 0);
                uint4 _vld_2[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                        (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_3[8];
            {
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(partial_O + (row_k_1 + 256) + 0);
                uint4 _vld_3[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_3[_blk] = _vptr_3[_blk];
                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                        (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                    }
                }
            }
            int _min_2 = ((split_group + 2) < (last_split) ? (split_group + 2) : (last_split));
            int split_k_2 = _min_2;
            int row_k_3 = partial_base + split_k_2 * 512;
            float _vec_load_4[8];
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(partial_O + row_k_3 + 0);
                uint4 _vld_4[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_4[_blk] = _vptr_4[_blk];
                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                        (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_5[8];
            {
                const uint4* _vptr_5 = reinterpret_cast<const uint4*>(partial_O + (row_k_3 + 256) + 0);
                uint4 _vld_5[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_5[_blk] = _vptr_5[_blk];
                    uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                        (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                    }
                }
            }
            int _min_3 = ((split_group + 3) < (last_split) ? (split_group + 3) : (last_split));
            int split_k_4 = _min_3;
            int row_k_5 = partial_base + split_k_4 * 512;
            float _vec_load_6[8];
            {
                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(partial_O + row_k_5 + 0);
                uint4 _vld_6[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_6[_blk] = _vptr_6[_blk];
                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                        (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_7[8];
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(partial_O + (row_k_5 + 256) + 0);
                uint4 _vld_7[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_7[_blk] = _vptr_7[_blk];
                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                        (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                    }
                }
            }
            int _min_4 = ((split_group) < (last_split) ? (split_group) : (last_split));
            float _shfl_0 = __shfl_sync(0xFFFFFFFF, normalized_weight, _min_4);
            float weight_k = _shfl_0;
            if (last_split < split_group) {
                weight_k = 0.0f;
            }
            acc_lo[0] = acc_lo[0] + weight_k * _vec_load_0[0];
            acc_hi[0] = acc_hi[0] + weight_k * _vec_load_1[0];
            acc_lo[1] = acc_lo[1] + weight_k * _vec_load_0[1];
            acc_hi[1] = acc_hi[1] + weight_k * _vec_load_1[1];
            acc_lo[2] = acc_lo[2] + weight_k * _vec_load_0[2];
            acc_hi[2] = acc_hi[2] + weight_k * _vec_load_1[2];
            acc_lo[3] = acc_lo[3] + weight_k * _vec_load_0[3];
            acc_hi[3] = acc_hi[3] + weight_k * _vec_load_1[3];
            acc_lo[4] = acc_lo[4] + weight_k * _vec_load_0[4];
            acc_hi[4] = acc_hi[4] + weight_k * _vec_load_1[4];
            acc_lo[5] = acc_lo[5] + weight_k * _vec_load_0[5];
            acc_hi[5] = acc_hi[5] + weight_k * _vec_load_1[5];
            acc_lo[6] = acc_lo[6] + weight_k * _vec_load_0[6];
            acc_hi[6] = acc_hi[6] + weight_k * _vec_load_1[6];
            acc_lo[7] = acc_lo[7] + weight_k * _vec_load_0[7];
            acc_hi[7] = acc_hi[7] + weight_k * _vec_load_1[7];
            int _min_5 = ((split_group + 1) < (last_split) ? (split_group + 1) : (last_split));
            float _shfl_1 = __shfl_sync(0xFFFFFFFF, normalized_weight, _min_5);
            float weight_k_6 = _shfl_1;
            if (last_split < split_group + 1) {
                weight_k_6 = 0.0f;
            }
            acc_lo[0] = acc_lo[0] + weight_k_6 * _vec_load_2[0];
            acc_hi[0] = acc_hi[0] + weight_k_6 * _vec_load_3[0];
            acc_lo[1] = acc_lo[1] + weight_k_6 * _vec_load_2[1];
            acc_hi[1] = acc_hi[1] + weight_k_6 * _vec_load_3[1];
            acc_lo[2] = acc_lo[2] + weight_k_6 * _vec_load_2[2];
            acc_hi[2] = acc_hi[2] + weight_k_6 * _vec_load_3[2];
            acc_lo[3] = acc_lo[3] + weight_k_6 * _vec_load_2[3];
            acc_hi[3] = acc_hi[3] + weight_k_6 * _vec_load_3[3];
            acc_lo[4] = acc_lo[4] + weight_k_6 * _vec_load_2[4];
            acc_hi[4] = acc_hi[4] + weight_k_6 * _vec_load_3[4];
            acc_lo[5] = acc_lo[5] + weight_k_6 * _vec_load_2[5];
            acc_hi[5] = acc_hi[5] + weight_k_6 * _vec_load_3[5];
            acc_lo[6] = acc_lo[6] + weight_k_6 * _vec_load_2[6];
            acc_hi[6] = acc_hi[6] + weight_k_6 * _vec_load_3[6];
            acc_lo[7] = acc_lo[7] + weight_k_6 * _vec_load_2[7];
            acc_hi[7] = acc_hi[7] + weight_k_6 * _vec_load_3[7];
            int _min_6 = ((split_group + 2) < (last_split) ? (split_group + 2) : (last_split));
            float _shfl_2 = __shfl_sync(0xFFFFFFFF, normalized_weight, _min_6);
            float weight_k_7 = _shfl_2;
            if (last_split < split_group + 2) {
                weight_k_7 = 0.0f;
            }
            acc_lo[0] = acc_lo[0] + weight_k_7 * _vec_load_4[0];
            acc_hi[0] = acc_hi[0] + weight_k_7 * _vec_load_5[0];
            acc_lo[1] = acc_lo[1] + weight_k_7 * _vec_load_4[1];
            acc_hi[1] = acc_hi[1] + weight_k_7 * _vec_load_5[1];
            acc_lo[2] = acc_lo[2] + weight_k_7 * _vec_load_4[2];
            acc_hi[2] = acc_hi[2] + weight_k_7 * _vec_load_5[2];
            acc_lo[3] = acc_lo[3] + weight_k_7 * _vec_load_4[3];
            acc_hi[3] = acc_hi[3] + weight_k_7 * _vec_load_5[3];
            acc_lo[4] = acc_lo[4] + weight_k_7 * _vec_load_4[4];
            acc_hi[4] = acc_hi[4] + weight_k_7 * _vec_load_5[4];
            acc_lo[5] = acc_lo[5] + weight_k_7 * _vec_load_4[5];
            acc_hi[5] = acc_hi[5] + weight_k_7 * _vec_load_5[5];
            acc_lo[6] = acc_lo[6] + weight_k_7 * _vec_load_4[6];
            acc_hi[6] = acc_hi[6] + weight_k_7 * _vec_load_5[6];
            acc_lo[7] = acc_lo[7] + weight_k_7 * _vec_load_4[7];
            acc_hi[7] = acc_hi[7] + weight_k_7 * _vec_load_5[7];
            int _min_7 = ((split_group + 3) < (last_split) ? (split_group + 3) : (last_split));
            float _shfl_3 = __shfl_sync(0xFFFFFFFF, normalized_weight, _min_7);
            float weight_k_8 = _shfl_3;
            if (last_split < split_group + 3) {
                weight_k_8 = 0.0f;
            }
            acc_lo[0] = acc_lo[0] + weight_k_8 * _vec_load_6[0];
            acc_hi[0] = acc_hi[0] + weight_k_8 * _vec_load_7[0];
            acc_lo[1] = acc_lo[1] + weight_k_8 * _vec_load_6[1];
            acc_hi[1] = acc_hi[1] + weight_k_8 * _vec_load_7[1];
            acc_lo[2] = acc_lo[2] + weight_k_8 * _vec_load_6[2];
            acc_hi[2] = acc_hi[2] + weight_k_8 * _vec_load_7[2];
            acc_lo[3] = acc_lo[3] + weight_k_8 * _vec_load_6[3];
            acc_hi[3] = acc_hi[3] + weight_k_8 * _vec_load_7[3];
            acc_lo[4] = acc_lo[4] + weight_k_8 * _vec_load_6[4];
            acc_hi[4] = acc_hi[4] + weight_k_8 * _vec_load_7[4];
            acc_lo[5] = acc_lo[5] + weight_k_8 * _vec_load_6[5];
            acc_hi[5] = acc_hi[5] + weight_k_8 * _vec_load_7[5];
            acc_lo[6] = acc_lo[6] + weight_k_8 * _vec_load_6[6];
            acc_hi[6] = acc_hi[6] + weight_k_8 * _vec_load_7[6];
            acc_lo[7] = acc_lo[7] + weight_k_8 * _vec_load_6[7];
            acc_hi[7] = acc_hi[7] + weight_k_8 * _vec_load_7[7];
        }
        int output_base = (query_idx * num_heads + head_idx) * 512 + lane * 8;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(acc_lo[0 + 0], acc_lo[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(acc_lo[0 + 2], acc_lo[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(acc_lo[0 + 4], acc_lo[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(acc_lo[0 + 6], acc_lo[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + output_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(acc_hi[0 + 0], acc_hi[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(acc_hi[0 + 2], acc_hi[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(acc_hi[0 + 4], acc_hi[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(acc_hi[0 + 6], acc_hi[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (output_base + 256)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        if (lane == 0) {
            float _log2_0;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(global_sum));
            lse_out[query_idx * num_heads + head_idx] = ((global_sum > 0.0f) ? (global_max + _log2_0) * lse_scale : -CAKE_INF);
        }
    }
}

} // extern "C"
