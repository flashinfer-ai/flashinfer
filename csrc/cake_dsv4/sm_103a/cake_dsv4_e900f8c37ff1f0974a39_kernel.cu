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
#include "cake_dsv4_device_common.cuh"
// The common preamble's fallback helper this module's helper library does not provide; the shared
// header carries another member's library definition of the name (kept translation-unit-local).
__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}


#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_dsv4_e900f8c37ff1f0974a39(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, int num_heads, int num_splits)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int stat_base = (query_idx * num_heads + head_idx) * num_splits;
    int partial_base = stat_base * 512;
    int output_base = (query_idx * num_heads + head_idx) * 512;
    int d_base = tid * 4;
    float acc[4];
    #pragma unroll
    for (int elem = 0; elem < 4; elem++) {
        acc[elem] = 0.0f;
    }
    if (num_splits == 4) {
        float _vec_load_0[4];
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + d_base) + 0);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                (&_vec_load_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
            }
        }
        float _vec_load_1[4];
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 512 + d_base) + 0);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                (&_vec_load_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
            }
        }
        float _vec_load_2[4];
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 1024 + d_base) + 0);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&_vec_load_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
        float _vec_load_3[4];
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 1536 + d_base) + 0);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&_vec_load_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
        float lse0 = partial_lse[stat_base];
        float lse1 = partial_lse[stat_base + 1];
        float lse2 = partial_lse[stat_base + 2];
        float lse3 = partial_lse[stat_base + 3];
        float _max_0 = max_noftz(lse0, lse1);
        float _max_1 = max_noftz(_max_0, lse2);
        float _max_2 = max_noftz(_max_1, lse3);
        float global_max = _max_2;
        float _exp2_0 = approx_exp2(lse0 - global_max);
        float w0 = ((lse0 == -CAKE_INF) ? 0.0f : _exp2_0);
        float _exp2_1 = approx_exp2(lse1 - global_max);
        float w1 = ((lse1 == -CAKE_INF) ? 0.0f : _exp2_1);
        float _exp2_2 = approx_exp2(lse2 - global_max);
        float w2 = ((lse2 == -CAKE_INF) ? 0.0f : _exp2_2);
        float _exp2_3 = approx_exp2(lse3 - global_max);
        float w3 = ((lse3 == -CAKE_INF) ? 0.0f : _exp2_3);
        float global_sum = w0 + w2 + (w1 + w3);
        float _rcp_0 = approx_rcp(global_sum);
        float inv_sum = _rcp_0;
        float weight0 = ((global_sum > 0.0f) ? w0 * inv_sum : 0.0f);
        float weight1 = ((global_sum > 0.0f) ? w1 * inv_sum : 0.0f);
        float weight2 = ((global_sum > 0.0f) ? w2 * inv_sum : 0.0f);
        float weight3 = ((global_sum > 0.0f) ? w3 * inv_sum : 0.0f);
        #pragma unroll
        for (int elem_1 = 0; elem_1 < 4; elem_1++) {
            acc[elem_1] = acc[elem_1] + weight0 * _vec_load_0[elem_1];
            acc[elem_1] = acc[elem_1] + weight1 * _vec_load_1[elem_1];
            acc[elem_1] = acc[elem_1] + weight2 * _vec_load_2[elem_1];
            acc[elem_1] = acc[elem_1] + weight3 * _vec_load_3[elem_1];
        }
    } else if (num_splits == 3) {
        float _vec_load_4[4];
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + d_base) + 0);
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                (&_vec_load_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
            }
        }
        float _vec_load_5[4];
        {
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 512 + d_base) + 0);
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                (&_vec_load_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
            }
        }
        float _vec_load_6[4];
        {
            uint2 _vld_6;
            _vld_6 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 1024 + d_base) + 0);
            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_6[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                (&_vec_load_6[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
            }
        }
        float lse0_1 = partial_lse[stat_base];
        float lse1_1 = partial_lse[stat_base + 1];
        float lse2_1 = partial_lse[stat_base + 2];
        float _max_3 = max_noftz(lse0_1, lse1_1);
        float _max_4 = max_noftz(_max_3, lse2_1);
        float global_max_1 = _max_4;
        float _exp2_4 = approx_exp2(lse0_1 - global_max_1);
        float w0_1 = ((lse0_1 == -CAKE_INF) ? 0.0f : _exp2_4);
        float _exp2_5 = approx_exp2(lse1_1 - global_max_1);
        float w1_1 = ((lse1_1 == -CAKE_INF) ? 0.0f : _exp2_5);
        float _exp2_6 = approx_exp2(lse2_1 - global_max_1);
        float w2_1 = ((lse2_1 == -CAKE_INF) ? 0.0f : _exp2_6);
        float global_sum_1 = w0_1 + w2_1 + w1_1;
        float _rcp_1 = approx_rcp(global_sum_1);
        float inv_sum_1 = _rcp_1;
        float weight0_1 = ((global_sum_1 > 0.0f) ? w0_1 * inv_sum_1 : 0.0f);
        float weight1_1 = ((global_sum_1 > 0.0f) ? w1_1 * inv_sum_1 : 0.0f);
        float weight2_1 = ((global_sum_1 > 0.0f) ? w2_1 * inv_sum_1 : 0.0f);
        #pragma unroll
        for (int elem_2 = 0; elem_2 < 4; elem_2++) {
            acc[elem_2] = acc[elem_2] + weight0_1 * _vec_load_4[elem_2];
            acc[elem_2] = acc[elem_2] + weight1_1 * _vec_load_5[elem_2];
            acc[elem_2] = acc[elem_2] + weight2_1 * _vec_load_6[elem_2];
        }
    } else {
        if (num_splits == 5) {
            float _vec_load_7[4];
            {
                uint2 _vld_7;
                _vld_7 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + d_base) + 0);
                uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_vec_load_7[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                    (&_vec_load_7[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                }
            }
            float _vec_load_8[4];
            {
                uint2 _vld_8;
                _vld_8 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 512 + d_base) + 0);
                uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_vec_load_8[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                    (&_vec_load_8[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                }
            }
            float _vec_load_9[4];
            {
                uint2 _vld_9;
                _vld_9 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 1024 + d_base) + 0);
                uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_vec_load_9[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                    (&_vec_load_9[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                }
            }
            float _vec_load_10[4];
            {
                uint2 _vld_10;
                _vld_10 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 1536 + d_base) + 0);
                uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_vec_load_10[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                    (&_vec_load_10[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                }
            }
            float _vec_load_11[4];
            {
                uint2 _vld_11;
                _vld_11 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + 2048 + d_base) + 0);
                uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_vec_load_11[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
                    (&_vec_load_11[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
                }
            }
            float lse0_2 = partial_lse[stat_base];
            float lse1_2 = partial_lse[stat_base + 1];
            float lse2_2 = partial_lse[stat_base + 2];
            float lse3_1 = partial_lse[stat_base + 3];
            float lse4 = partial_lse[stat_base + 4];
            float _max_5 = max_noftz(lse0_2, lse1_2);
            float _max_6 = max_noftz(_max_5, lse2_2);
            float _max_7 = max_noftz(_max_6, lse3_1);
            float _max_8 = max_noftz(_max_7, lse4);
            float global_max_2 = _max_8;
            float _exp2_7 = approx_exp2(lse0_2 - global_max_2);
            float w0_2 = ((lse0_2 == -CAKE_INF) ? 0.0f : _exp2_7);
            float _exp2_8 = approx_exp2(lse1_2 - global_max_2);
            float w1_2 = ((lse1_2 == -CAKE_INF) ? 0.0f : _exp2_8);
            float _exp2_9 = approx_exp2(lse2_2 - global_max_2);
            float w2_2 = ((lse2_2 == -CAKE_INF) ? 0.0f : _exp2_9);
            float _exp2_10 = approx_exp2(lse3_1 - global_max_2);
            float w3_1 = ((lse3_1 == -CAKE_INF) ? 0.0f : _exp2_10);
            float _exp2_11 = approx_exp2(lse4 - global_max_2);
            float w4 = ((lse4 == -CAKE_INF) ? 0.0f : _exp2_11);
            float global_sum_2 = w0_2 + w4 + w2_2 + (w1_2 + w3_1);
            float _rcp_2 = approx_rcp(global_sum_2);
            float inv_sum_2 = _rcp_2;
            float weight0_2 = ((global_sum_2 > 0.0f) ? w0_2 * inv_sum_2 : 0.0f);
            float weight1_2 = ((global_sum_2 > 0.0f) ? w1_2 * inv_sum_2 : 0.0f);
            float weight2_2 = ((global_sum_2 > 0.0f) ? w2_2 * inv_sum_2 : 0.0f);
            float weight3_1 = ((global_sum_2 > 0.0f) ? w3_1 * inv_sum_2 : 0.0f);
            float weight4 = ((global_sum_2 > 0.0f) ? w4 * inv_sum_2 : 0.0f);
            #pragma unroll
            for (int elem_3 = 0; elem_3 < 4; elem_3++) {
                acc[elem_3] = acc[elem_3] + weight0_2 * _vec_load_7[elem_3];
                acc[elem_3] = acc[elem_3] + weight1_2 * _vec_load_8[elem_3];
                acc[elem_3] = acc[elem_3] + weight2_2 * _vec_load_9[elem_3];
                acc[elem_3] = acc[elem_3] + weight3_1 * _vec_load_10[elem_3];
                acc[elem_3] = acc[elem_3] + weight4 * _vec_load_11[elem_3];
            }
        } else {
            float local_lse = -CAKE_INF;
            if (lane < num_splits) {
                local_lse = partial_lse[stat_base + lane];
            }
            float _warp_reduce_0 = local_lse;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
            float global_max_3 = _warp_reduce_0;
            float local_weight = 0.0f;
            if (lane < num_splits) {
                float _exp2_12 = approx_exp2(local_lse - global_max_3);
                local_weight = ((local_lse == -CAKE_INF) ? 0.0f : _exp2_12);
            }
            float _warp_reduce_1 = local_weight;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
            float global_sum_3 = _warp_reduce_1;
            float _rcp_3 = approx_rcp(global_sum_3);
            float normalized_weight = ((global_sum_3 > 0.0f) ? local_weight * _rcp_3 : 0.0f);
            #pragma unroll 5
            for (int split = 0; split < num_splits; split++) {
                float _vec_load_12[4];
                {
                    uint2 _vld_12;
                    _vld_12 = *reinterpret_cast<const uint2*>(partial_O + (partial_base + split * 512 + d_base) + 0);
                    uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&_vec_load_12[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                        (&_vec_load_12[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                    }
                }
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, normalized_weight, split);
                float weight = _shfl_0;
                #pragma unroll
                for (int elem_4 = 0; elem_4 < 4; elem_4++) {
                    acc[elem_4] = acc[elem_4] + weight * _vec_load_12[elem_4];
                }
            }
        }
    }
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O + (output_base + d_base)))[0]) = _pk2;
    }
}

} // extern "C"
