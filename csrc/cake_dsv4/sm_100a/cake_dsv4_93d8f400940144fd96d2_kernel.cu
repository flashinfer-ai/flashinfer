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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128
#define USE_PDL 0

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_dsv4_93d8f400940144fd96d2(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, int num_q_heads, int num_split)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    const int wg_dummy = 0;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int stat_base = (batch_idx * num_q_heads + head_idx) * 5;
    float local_m = -CAKE_INF;
    if (lane < 5) {
        local_m = partial_lse[stat_base + lane];
    }
    float _warp_reduce_0 = local_m;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
    float global_max = _warp_reduce_0;
    float local_w = 0.0f;
    if (lane < 5) {
        float _exp2_0 = approx_exp2(local_m - global_max);
        local_w = ((local_m == -CAKE_INF) ? 0.0f : _exp2_0);
    }
    float _warp_reduce_1 = local_w;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    float global_sum = _warp_reduce_1;
    float _rcp_0 = approx_rcp(global_sum);
    float inv_sum = ((global_sum > 0.0f) ? _rcp_0 : 0.0f);
    float local_weight = local_w * inv_sum;
    int po_head_base = stat_base * 512;
    int o_head_base = (batch_idx * num_q_heads + head_idx) * 512;
    float acc[4];
    int d_base = tid * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        acc[e] = 0.0f;
    }
    #pragma unroll
    for (int s = 0; s < 5; s++) {
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, local_weight, s);
        float split_weight = _shfl_0;
        float _vec_load_0[4];
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(partial_O + (po_head_base + s * 512 + d_base) + 0);
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                (&_vec_load_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
            }
        }
        #pragma unroll
        for (int e_1 = 0; e_1 < 4; e_1++) {
            acc[e_1] = acc[e_1] + split_weight * _vec_load_0[e_1];
        }
    }
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(acc[0 + 0], acc[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(acc[0 + 2], acc[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O + (o_head_base + d_base)))[0]) = _pk2;
    }
}

} // extern "C"
