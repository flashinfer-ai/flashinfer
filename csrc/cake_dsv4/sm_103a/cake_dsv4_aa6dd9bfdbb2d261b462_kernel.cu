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
#define USE_PDL 0

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_dsv4_aa6dd9bfdbb2d261b462(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, int num_q_heads, int num_split)
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
    int stat_base = (batch_idx * num_q_heads + head_idx) * 4;
    float lane_lse[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        lane_lse[s] = partial_lse[stat_base + s];
    }
    float global_max = -CAKE_INF;
    #pragma unroll
    for (int s_1 = 0; s_1 < 4; s_1++) {
        float _max_0 = max_noftz(global_max, lane_lse[s_1]);
        global_max = _max_0;
    }
    float lane_w[4];
    float lane_sum[4];
    #pragma unroll
    for (int s_2 = 0; s_2 < 4; s_2++) {
        float _exp2_0 = approx_exp2(lane_lse[s_2] - global_max);
        lane_w[s_2] = ((lane_lse[s_2] == -CAKE_INF) ? 0.0f : _exp2_0);
        lane_sum[s_2] = lane_w[s_2];
    }
    lane_sum[0] = lane_sum[0] + lane_sum[2];
    lane_sum[1] = lane_sum[1] + lane_sum[3];
    lane_sum[0] = lane_sum[0] + lane_sum[1];
    float global_sum = lane_sum[0];
    float _rcp_0 = approx_rcp(global_sum);
    float inv_sum = ((global_sum > 0.0f) ? _rcp_0 : 0.0f);
    #pragma unroll
    for (int s_3 = 0; s_3 < 4; s_3++) {
        lane_w[s_3] = lane_w[s_3] * inv_sum;
    }
    int po_head_base = stat_base * 512;
    int o_head_base = (batch_idx * num_q_heads + head_idx) * 512;
    float acc[4];
    int d_base = tid * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        acc[e] = 0.0f;
    }
    #pragma unroll
    for (int s_4 = 0; s_4 < 4; s_4++) {
        float split_weight = lane_w[s_4];
        float _vec_load_0[4];
        {
            uint2 _vld_0;
            _vld_0 = *reinterpret_cast<const uint2*>(partial_O + (po_head_base + s_4 * 512 + d_base) + 0);
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
